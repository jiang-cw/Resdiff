import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import config

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim =config.latent_dim):
        super(Encoder, self).__init__()
        channels = [16, 32, 64]
        num_res_blocks = 2
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != len(channels)-1:
                layers.append(DownSampleBlock(channels[i+1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)
       
    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim = config.latent_dim):
        super(Decoder, self).__init__()
        channels = [64, 32, 16]
        num_res_blocks = 2

        in_channels = channels[0]
        layers = [nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(UpSampleBlock(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AAE(nn.Module):
    def __init__(self):
        super(AAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, AV45):
        encoded_AV45 = self.encoder(AV45)
        decoded_AV45 = self.decoder(encoded_AV45)
        return  decoded_AV45

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

class Discriminator(nn.Module):
    def __init__(self, image_channels = 1, num_filters_last=16, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv3d(image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv3d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4, 2, 1, bias=False),
                nn.BatchNorm3d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv3d(num_filters_last * num_filters_mult, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def L2_norm(x):
    x = x.reshape(1,64,-1)
    out = torch.div(x,torch.pow(x, 2).sum(dim=1,keepdim=True).sqrt())
    out.requires_grad_()
    return out

class clsBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel=3,stride=1,padding=1,do_norm=True,do_relu=True,relufactor=0):
        super(clsBlock,self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride, padding, padding_mode="zeros")
        if do_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = None
        if do_relu:
            if relufactor == 0: 
                self.relu=nn.ReLU(inplace=True)
            else:
                self.relu=nn.LeakyReLU(relufactor,inplace=True)
        else:
            self.relu=None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x=self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Classifier(nn.Module): 
    def __init__(self, in_channels=1, features=[16, 32, 64, 64, 64]):
        super(Classifier,self).__init__()
        self.linear=nn.Linear(4096 ,1)
        self.max_pool=nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding =(1, 1, 1))
        self.average_pool=nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding =(1, 1, 1))
        self.layer_1=clsBlock(in_channels, features[0], 3, 1, 1)
        self.layer_2=clsBlock(features[0], features[1], 3, 1, 1)
        self.layer_3=clsBlock(features[1], features[2], 3, 1, 1)
        self.layer_4=clsBlock(features[2], features[3], 3, 1, 1)
        self.layer_5=clsBlock(features[3], features[4], 3, 1, 1)

    def forward(self, x):
        layer_1=self.layer_1(x)
        layer_1_0=self.max_pool(layer_1)
        layer_2=self.layer_2(layer_1_0)
        layer_2_0=self.max_pool(layer_2)
        layer_3=self.layer_3(layer_2_0)
        layer_3_0=self.max_pool(layer_3)
        layer_4=self.layer_4(layer_3_0)
        layer_4_0=self.max_pool(layer_4)
        layer_5=self.layer_5(layer_4_0)
        layer_5_0=self.average_pool(layer_5)
        #final_layer=L2_norm(layer_5_0)
        final_layer = layer_5_0.reshape(layer_5_0.shape[0],-1)
        pred = self.linear(final_layer)
        return pred

if __name__ == "__main__":
    x = torch.randn((1, 1, 160, 192, 160))
    model = Classifier()
    #model = Discriminator()
    preds = model(x)
    print(preds.shape)
    #print(preds)