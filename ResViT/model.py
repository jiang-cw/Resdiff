import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
import config
import collections
from itertools import repeat
import functools
from torch.nn import init

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(3, "_pair")

class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(channel_compression, self).__init__()
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, img_size, input_dim, old = 1):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        grid_size = config.grid
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1],img_size[2] // 16 // grid_size[2])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16, patch_size[2] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (img_size[2] // patch_size_real[2])
        in_channels = 512
        #Learnable patch embeddings
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,out_channels=config.hidden_size,kernel_size=patch_size,stride=patch_size)
        #learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        print(x.shape)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.num_layers):
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, img_size, vis, in_channels=1, old = 1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size,input_dim=in_channels,old = old)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(dim),nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ART_block(nn.Module):
    def __init__(self, input_dim, img_size, transformer = None):
        super(ART_block, self).__init__()
        self.transformer = transformer
        ngf = 64
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm3d
        padding_type = 'reflect'
        if self.transformer:
            # Downsample
            model = [nn.Conv3d(ngf * 4, ngf * 8, kernel_size=3,stride=2, padding=1, bias=use_bias),norm_layer(ngf * 8),nn.ReLU(True)]
            model += [nn.Conv3d(ngf * 8, 512, kernel_size=3,stride=2, padding=1, bias=use_bias), norm_layer(512), nn.ReLU(True)]
            setattr(self, 'downsample', nn.Sequential(*model))

            #Patch embedings
            self.embeddings = Embeddings(img_size=img_size, input_dim=input_dim)

            # Upsampling block
            model = [nn.ConvTranspose3d(config.hidden_size, ngf * 8,kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(ngf * 8),nn.ReLU(True)]
            model += [nn.ConvTranspose3d(ngf * 8, ngf * 4,kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(ngf * 4),nn.ReLU(True)]
            setattr(self, 'upsample', nn.Sequential(*model))

            #Channel compression
            self.cc = channel_compression(ngf * 8, ngf * 4)

        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        if self.transformer:
            # downsample
            down_sampled = self.downsample(x)
            # embed
            embedding_output = self.embeddings(down_sampled)
            # feed to transformer
            transformer_out, attn_weights = self.transformer(embedding_output)
            B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w, d= 16, 16, 8#int(np.cbrt(n_patch)), int(np.cbrt(n_patch)),int(np.cbrt(n_patch))
            transformer_out = transformer_out.permute(0, 2, 1)
            transformer_out = transformer_out.contiguous().view(B, hidden, h, w, d)
            # upsample transformer output
            transformer_out = self.upsample(transformer_out)
            # concat transformer output and resnet output
            x = torch.cat([transformer_out, x], dim=1)
            # channel compression
            x = self.cc(x)
        # residual CNN
        x = self.residual_cnn(x)
        return x

########Generator############
class ResViT(nn.Module):
    def __init__(self, input_dim=1, img_size=[256,256,128], output_dim=1, vis=False):
        super(ResViT, self).__init__()
        self.transformer_encoder = Encoder(vis)
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm3d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad3d(3),nn.Conv3d(input_dim, ngf, kernel_size=7, padding=0,bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,stride=2, padding=1, bias=use_bias),norm_layer(ngf * mult * 2),nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,stride=2, padding=1, bias=use_bias),norm_layer(ngf * mult * 2),nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ###################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(input_dim, img_size, transformer = self.transformer_encoder)
        self.art_2 = ART_block(input_dim, img_size, transformer=None)
        self.art_3 = ART_block(input_dim, img_size, transformer=None)
        self.art_4 = ART_block(input_dim, img_size, transformer=None)
        self.art_5 = ART_block(input_dim, img_size, transformer=None)
        self.art_6 = ART_block(input_dim, img_size, transformer = self.transformer_encoder)
        self.art_7 = ART_block(input_dim, img_size, transformer=None)
        self.art_8 = ART_block(input_dim, img_size, transformer=None)
        self.art_9 = ART_block(input_dim, img_size, transformer=None)
        ###########################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad3d(3)]
        model += [nn.Conv3d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################
    def forward(self, x):
        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        #Information Bottleneck
        x = self.art_1(x)
        x = self.art_2(x)
        x = self.art_3(x)
        x = self.art_4(x)
        x = self.art_5(x)
        x = self.art_6(x)
        x = self.art_7(x)
        x = self.art_8(x)
        x = self.art_9(x)

        #decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x

########Discriminator############
class Discriminator(nn.Module):
    def __init__(self, input_nc=2, ndf=32, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm3d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

if __name__ == "__main__":
    x = torch.randn((1, 2, 128, 128, 128))
   # model = ResViT()
    model = Discriminator()
    preds = model(x)
    print(preds.shape)

