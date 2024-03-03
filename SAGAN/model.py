import torch
import torch.nn as nn
import torch.nn.functional as F
import config

def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def custom_cond(value, threshold, default_value):
    return value if value >= threshold else default_value

def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = weights.size()
    w_mat = weights.view(-1, w_shape[-1]) 
    u = torch.nn.Parameter(torch.randn(1, w_shape[-1]), requires_grad=False)
    u_ = u.clone()
    
    for _ in range(num_iters):
        v_ = _l2normalize(torch.matmul(u_, w_mat.t()))
        u_ = _l2normalize(torch.matmul(v_, w_mat))
    
    sigma = torch.squeeze(torch.matmul(torch.matmul(v_, w_mat), u_.t()))
    w_bar = w_mat / sigma

    if update_collection is None:
        with torch.no_grad():
            u.copy_(u_)
            w_bar = w_bar.view(w_shape)
    else:
        w_bar = w_bar.view(w_shape)
        if update_collection != 'NO_OPS':
            raise NotImplementedError("Update collection not directly supported in PyTorch.")

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def snconv3d(input_dim, output_dim, size, stride, sn_iters=1, update_collection=None):
    conv = nn.Conv3d(input_dim, output_dim, size, stride, padding= (size - 1) // 2)
    w = conv.weight
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv.weight.data = w_bar
    return conv

def snconv3d_tranpose(input_dim, output_dim, size, stride, sn_iters=1, update_collection=None):
    conv_transpose = nn.ConvTranspose3d(input_dim, output_dim.shape[1], size, stride, padding=1)
    w = conv_transpose.weight
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv_transpose.weight.data = w_bar
    return conv_transpose

def snconv3d_1x1(input_dim, output_dim, sn_iters=1, sn=True, update_collection=None):
    conv = nn.Conv3d(input_dim, output_dim, kernel_size=1)
    w = conv.weight
    if sn:
        w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        conv.weight.data = w_bar
    return conv

class sn_attention(nn.Module):
    def __init__(self, in_channels, sn=True, final_layer=False, update_collection=None):
        super(sn_attention, self).__init__()
        self.sn = sn
        self.final_layer = final_layer
        self.update_collection = update_collection
        self.theta = snconv3d_1x1(input_dim = in_channels, sn=self.sn, output_dim=(in_channels//8), update_collection=self.update_collection)
        self.phi = snconv3d_1x1(input_dim = in_channels, sn=self.sn, output_dim=(in_channels//8), update_collection=self.update_collection)
        self.g = snconv3d_1x1(input_dim = in_channels, sn=self.sn, output_dim=(in_channels // 2 or 16), update_collection=self.update_collection)
        self.conv = snconv3d_1x1(input_dim = in_channels // 2 or 16, sn=self.sn, output_dim=in_channels, update_collection=self.update_collection)
    
    def forward(self, x):
        batch_size, num_channels, height, width, depth = x.size()
        if not batch_size:
            batch_size = 1
        location_num = height * width * depth
        if self.final_layer:
            downsampled = location_num // (64 ** 3)
            stride = 64
        else:
            downsampled = location_num // 8
            stride = 2
        # theta path
        theta = self.theta(x)
        theta = theta.view(batch_size, location_num, (num_channels // 8 or 4))
        
        # phi path
        phi = self.phi(x)
        phi = F.max_pool3d(phi, kernel_size=(2, 2, 2), stride=stride)
        phi = phi.view(batch_size, downsampled, (num_channels // 8 or 4))

        attn = torch.matmul(theta, phi.transpose(1, 2))
        attn = F.softmax(attn, dim=2)

        # g path
        g = self.g(x)
        g = F.max_pool3d(g, kernel_size=(2, 2, 2), stride=stride)
        g = g.view(batch_size, downsampled, (num_channels // 2 or 16))

        attn_g = torch.matmul(attn, g)
        attn_g = attn_g.view(batch_size, num_channels // 2 or 16, height, width, depth)
        sigma = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
        attn_g = self.conv(attn_g)
        
        if self.final_layer:
            self.sigma = None
            return attn_g
        else:
            return (x + sigma * attn_g) / (1 + sigma)

class SAGAN(nn.Module):
    def __init__(self, input_channels = 1, output_channels = 1, attn=True):
        super(SAGAN, self).__init__()
        self.output_channels = output_channels
        self.attn = attn
        self._prob = 0.5
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder_layer(self, size, inputs, output, bn=True, use_dropout=False):
        layers = []
        layers.append(nn.Conv3d(inputs, output, kernel_size=size, stride=2, padding = 1))
        if bn:
            layers.append(nn.BatchNorm3d(output))
        if use_dropout:
            layers.append(nn.Dropout(p=self._prob))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _build_encoder(self):
        encoder = nn.Sequential()
        encoder.add_module('l1', self._build_encoder_layer(4, inputs = 1, output=32, bn=False))
        encoder.add_module('l2', self._build_encoder_layer(4, 32, 64))
        encoder.add_module('l3', self._build_encoder_layer(4, 64, 128))
        encoder.add_module('l4', self._build_encoder_layer(4, 128, 256))
        if self.attn:
            attn_module = sn_attention(256, sn=True).to(config.device)
            encoder.add_module('encoder_attention', attn_module)
        else:
            encoder.add_module('encoder_attention', encoder[4][-1])
        encoder.add_module('l6', self._build_encoder_layer(4, 256, 256))
        # encoder.add_module('l7', self._build_encoder_layer(3, 512, 512))
        return encoder

    def _build_decoder_layer(self, inputs, output, size, stride, padding = 1, use_dropout=False):
        layers = []
        layers.append(nn.ConvTranspose3d(inputs, output, kernel_size=size, stride=stride, padding= padding))
        layers.append(nn.BatchNorm3d(output))
        if use_dropout:
            layers.append(nn.Dropout(p=self._prob))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _build_decoder(self):
        decoder = nn.Sequential()
        decoder.add_module('dl1', self._build_decoder_layer(256, 256, 4, 2, padding = 1, use_dropout=False))
        # decoder.add_module('dl2', self._build_decoder_layer(1024, 512, 4, 2, use_dropout=True))
        decoder.add_module('dl4', self._build_decoder_layer(512, 128, 4, 2))
        decoder.add_module('dl5', self._build_decoder_layer(256, 64, 4, 2))
        if self.attn:
            attn_module = sn_attention(64, sn=True).to(config.device)
            decoder.add_module('decoder_attention', attn_module)
        else:
            decoder.add_module('decoder_attention', decoder[4][-1])
        decoder.add_module('dl6', self._build_decoder_layer(128, 32, 4, 2))
        decoder.add_module('dl7', self._build_decoder_layer(64, 1, 4, 2))
        
        final_layer = nn.Sequential(
            snconv3d(1, 1, size=3, stride=1),
            sn_attention(1, sn=True, final_layer=True),
            nn.ReLU()
        )
        decoder.add_module('final', final_layer)
        return decoder

    def forward(self, x):
        encoder_outputs = []
        encoder_children = list(self._encoder.children())
        flag = 0
        for i, layer in enumerate(encoder_children):
            x = layer(x)
            if ('sn_attention' not in layer._get_name()):
                encoder_outputs.append(x)

        for i, layer in enumerate(self._decoder.children()):
            if i < 6:
                if i == 0:
                    x = layer(x)
                elif 'sn_attention' in layer._get_name():
                    flag = 1
                    x = layer(x)
                else:
                    if flag !=0:
                        i = i -1 
                    concat_input = torch.cat([x, encoder_outputs[-i-1]], dim=1)
                    x = layer(concat_input)
        return x

class disc_layer(nn.Module):
    def __init__(self, input_dim, output_dim, bn=True, use_dropout=False):
        super(disc_layer, self).__init__()
        self.bn = bn
        self.layer = snconv3d(input_dim, output_dim, size=4, stride=2)
        self.norm = nn.BatchNorm3d(output_dim)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.layer(x)
        if self.bn:
            x = self.norm(x)
        x = self.leakyrelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channel=2):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.fmap = []
        self.layer_1 = disc_layer(self.in_channel, 64, bn=False)
        self.layer_2 = disc_layer(64, 128)
        self.layer_3 = disc_layer(128, 256)
        self.layer_attn = sn_attention(256, sn=True)
        self.layer_4 = disc_layer(256, 512)
        self.layer_5 = nn.Sequential(snconv3d_1x1(512, 1, sn=True),nn.BatchNorm3d(1),nn.Sigmoid())

    def forward(self, x):
        fmap = []
        layer_1 = self.layer_1(x)
        fmap.append(layer_1)
        layer_2 = self.layer_2(layer_1)
        fmap.append(layer_2)
        layer_3 = self.layer_3(layer_2)
        fmap.append(layer_3)
        layer_attn = self.layer_attn(layer_3)
        fmap.append(layer_attn)
        layer_4 = self.layer_4(layer_attn)
        fmap.append(layer_4)
        layer_5 = self.layer_5(layer_4)
        fmap.append(layer_5)
        return layer_5

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

if __name__ == "__main__":
    x = torch.randn((1, 1, 128, 128, 128))
    model = SCGAN()
    #discriminator = Discriminator()
    output = model(x)
    print(output.shape)


