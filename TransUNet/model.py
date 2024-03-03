import torch
import torch.nn as nn
import numpy as np
import config
import functools
import torch.nn.functional as F
import copy
import math
from torch.nn.modules.utils import _triple
from collections import OrderedDict

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([4, 3, 0, 1, 2])
    return torch.from_numpy(weights)

class StdConv3d(nn.Conv3d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class PreActBottleneck3D(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)  # Change the input channels of conv1 to cmid
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y

class ResNetV3(nn.Module):
    def __init__(self, block_units, width_factor):
        super(ResNetV3,self).__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv3d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck3D(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck3D(cin=width * 4, cout=width * 4, cmid=width)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck3D(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck3D(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck3D(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck3D(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, d, h, w = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(d / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, f"x {x.size()} should {right_size}"
                feat = torch.zeros((b, x.size()[1], right_size, h, w), device=x.device)
                feat[:, :, 0:x.size()[2], :, :] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features

class Discriminator(nn.Module):
    def __init__(self, input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
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

class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer_num_heads
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

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer_dropout_rate)

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
    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _triple(img_size)

        if config.grid is not None:
            grid_size = config.grid
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1], img_size[2] // 16 // grid_size[2])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16, patch_size[2] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (img_size[2] // patch_size_real[2])  
            self.hybrid = True
        else:
            patch_size = _triple(config.patches_size)
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])* (img_size[2] // patch_size[1]) 
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV3(block_units=config.resnet_num_layers, width_factor=config.resnet_width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer_dropout_rate)


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp()
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
    def __init__(self, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=not (use_batchnorm),)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm3d(out_channels)
        super(Conv3dReLU, self).__init__(conv, bn, relu)

class DecoderBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup3D(nn.Module):
    def __init__(self):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if config.n_skip != 0:
            skip_channels = config.skip_channels
            for i in range(4 - config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock3D(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w, d = 16, 16, 8#8, 8, 8
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w, d)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[config.n_skip - 1- i] if (i < config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class TransUNet(nn.Module):
    def __init__(self, img_size=[256,256,128], num_classes=3, zero_head=False, vis=False):
        super(TransUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.transformer = Transformer(img_size, vis)
        self.decoder = DecoderCup3D()
        self.out_layer = nn.Conv3d(config.decoder_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        out = self.out_layer(x)
        return out

if __name__ == "__main__":
    #x = torch.randn((1, 1, 40, 48, 40))
    y = torch.randn((1, 1, 128, 128, 128))
    # t = torch.randint(1000, (1,))
    model = TransUNet()
    #model = Discriminator()
    preds = model(y)
    print(preds.shape)
    #print(preds)