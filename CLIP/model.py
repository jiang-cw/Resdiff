import math
import torch
from torch import nn,einsum
import numpy as np
from torch.nn import init
from torch.nn import functional as F
import config
from inspect import isfunction
from abc import abstractmethod
from numbers import Number
from tqdm import tqdm
from dataset import *
import clip
from einops import rearrange, repeat, reduce, pack, unpack

def l2norm(t):
    return F.normalize(t, dim = -1)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# Unet
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=16, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=16):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwd, bncyxz -> bnhwdyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, depth, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, depth, height, width, depth)

        out = torch.einsum("bnhwdyxz, bncyxz -> bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))

        return out + input

class CrossAttention(nn.Module):
    def __init__(self, x_channel, y_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, x_channel)
        self.q = nn.Linear(x_channel, x_channel)
        self.k = nn.Linear(y_channel, y_channel)
        self.v = nn.Linear(y_channel, y_channel)
        self.out = nn.Linear(x_channel, x_channel)

    def forward(self, input, y):
        b, n, H, W, D = input.shape
        x = self.norm(input)
        x = x.reshape(b, n, H*W*D).permute(0, 2, 1)
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        attn = torch.matmul(q,k.T) / math.sqrt(n)
        attn = torch.softmax(attn, -1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1).reshape(b, n, H, W, D)
        return out

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=16, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=1,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=(8,),
        res_blocks=1,
        dropout=0,
        with_time_emb=True,
        image_size=128,
        num_classes = 2
    ):
        super().__init__()
        if with_time_emb:
            time_dim = inner_channel * 4
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, time_dim),
                Swish(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.label_emb = nn.Embedding(num_classes, time_dim)
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = []
        now_res = image_size
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            for _ in range(0, res_blocks):
                channel_mult = inner_channel * channel_mults[ind]
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
            downs.append(Downsample(channel_mult,channel_mult))
            pre_channel = channel_mult
            now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(channel_mult, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=True),
            CrossAttention(channel_mult,channel_mult),
            ResnetBlocWithAttn(channel_mult, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=False)
        ])
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            ups.append(Upsample(pre_channel,pre_channel))
            now_res = now_res*2
            for _ in range(0, res_blocks):
                channel_mult = inner_channel * channel_mults[ind]
                ups.append(ResnetBlocWithAttn(
                    channel_mult+feat_channels.pop(), channel_mult//2, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult//2
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, y = None):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                feats.append(x)
            else:
                x = layer(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x,y)

        for layer in self.ups:

            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class Diffusion:
    def __init__(self, noise_steps=15, min_noise_level=0.0001, etas_end=1, kappa= 0.1, power=1):
        #self.sqrt_etas = self.get_named_eta_schedule(noise_steps,min_noise_level,etas_end,kappa,power)
        self.etas = self.get_named_eta_schedule(noise_steps,min_noise_level,etas_end,kappa,power) #self.sqrt_etas **2
        self.sqrt_etas = np.sqrt(self.etas)
        #print(self.etas)
        self.kappa = kappa
        self.noise_steps = noise_steps
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(self.posterior_variance[1], self.posterior_variance[1:])

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

    # def get_named_eta_schedule(self,noise_steps,min_noise_level,etas_end,kappa,power):
    #     etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
    #     increaser = math.exp(1/((noise_steps-1))*math.log(etas_end/etas_start))
    #     base = np.ones([noise_steps, ]) * increaser
    #     power_timestep = np.linspace(0, 1, noise_steps, endpoint=True)**power
    #     power_timestep *= (noise_steps-1)
    #     sqrt_etas = np.power(base, power_timestep) * etas_start
    #     return sqrt_etas

    def get_named_eta_schedule(self,noise_steps,min_noise_level,etas_end,kappa,power):
        power_timestep = np.linspace(min_noise_level, etas_end, noise_steps)
        return power_timestep

    def _scale_input(self, inputs, t):
        std = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
        inputs_norm = inputs / std
        return inputs_norm

    def noise_images(self, x_start, y, t, noise=None):
        noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        return posterior_mean 
    
    def p_mean_variance(self, model, x_t, y, t, cond, clip_denoised=True):
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        #FDG_output = model(self._scale_input(x_t, t), t)
        FDG_output = model(x_t, t, cond)
        def process_xstart(x):
            if clip_denoised:
                return x.clamp(0, 1)
            return x
        pred_xstart = process_xstart(FDG_output)
        model_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def prior_sample(self, y, noise=None):
        #Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)
        if noise is None:
            noise = torch.randn_like(y)
        t = torch.tensor([self.noise_steps-1,] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise
    
    def p_sample(self, model, x, y, t, cond, clip_denoised=True):
        mean, log_variance = self.p_mean_variance(model, x, y, t, cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
        return sample

    def sample(self, model, y, cond, clip_denoised=True):
        # generating noise
        noise = torch.randn_like(y)
        y_sample = self.prior_sample(y, noise)
        indices = list(range(self.noise_steps))[::-1]
        for i in tqdm(indices):
            t = torch.tensor([i] * y.shape[0], device=config.device)
            with torch.no_grad():
                y_sample = self.p_sample(model, y_sample, y, t, cond, clip_denoised=clip_denoised)
        return y_sample

#CLIP
def get_optim_params(model_name: str):
    if model_name in ['ViT-B/32', 'ViT-B/16']:
        return ['visual.transformer.resblocks.11.attn.in_proj_weight',
                'visual.transformer.resblocks.11.attn.in_proj_bias',
                'visual.transformer.resblocks.11.attn.out_proj.weight',
                'visual.transformer.resblocks.11.attn.out_proj.bias',
                'visual.transformer.resblocks.11.ln_1.weight',
                'visual.transformer.resblocks.11.ln_1.bias',
                'visual.transformer.resblocks.11.mlp.c_fc.weight',
                'visual.transformer.resblocks.11.mlp.c_fc.bias',
                'visual.transformer.resblocks.11.mlp.c_proj.weight',
                'visual.transformer.resblocks.11.mlp.c_proj.bias',
                'visual.transformer.resblocks.11.ln_2.weight',
                'visual.transformer.resblocks.11.ln_2.bias',
                'visual.ln_post.weight',
                'visual.ln_post.bias',
                'visual.proj',
                'transformer.resblocks.11.attn.in_proj_weight',
                'transformer.resblocks.11.attn.in_proj_bias',
                'transformer.resblocks.11.attn.out_proj.weight',
                'transformer.resblocks.11.attn.out_proj.bias',
                'transformer.resblocks.11.ln_1.weight',
                'transformer.resblocks.11.ln_1.bias',
                'transformer.resblocks.11.mlp.c_fc.weight',
                'transformer.resblocks.11.mlp.c_fc.bias',
                'transformer.resblocks.11.mlp.c_proj.weight',
                'transformer.resblocks.11.mlp.c_proj.bias',
                'transformer.resblocks.11.ln_2.weight',
                'transformer.resblocks.11.ln_2.bias',
                'ln_final.weight',
                'ln_final.bias',
                'text_projection']
    elif model_name in ['ViT-L/14', 'ViT-L/14@336px']:
        return ['visual.transformer.resblocks.23.attn.in_proj_weight',
                'visual.transformer.resblocks.23.attn.in_proj_bias',
                'visual.transformer.resblocks.23.attn.out_proj.weight',
                'visual.transformer.resblocks.23.attn.out_proj.bias',
                'visual.transformer.resblocks.23.ln_1.weight',
                'visual.transformer.resblocks.23.ln_1.bias',
                'visual.transformer.resblocks.23.mlp.c_fc.weight',
                'visual.transformer.resblocks.23.mlp.c_fc.bias',
                'visual.transformer.resblocks.23.mlp.c_proj.weight',
                'visual.transformer.resblocks.23.mlp.c_proj.bias',
                'visual.transformer.resblocks.23.ln_2.weight',
                'visual.transformer.resblocks.23.ln_2.bias',
                'visual.ln_post.weight',
                'visual.ln_post.bias',
                'visual.proj',
                'transformer.resblocks.11.attn.in_proj_weight',
                'transformer.resblocks.11.attn.in_proj_bias',
                'transformer.resblocks.11.attn.out_proj.weight',
                'transformer.resblocks.11.attn.out_proj.bias',
                'transformer.resblocks.11.ln_1.weight',
                'transformer.resblocks.11.ln_1.bias',
                'transformer.resblocks.11.mlp.c_fc.weight',
                'transformer.resblocks.11.mlp.c_fc.bias',
                'transformer.resblocks.11.mlp.c_proj.weight',
                'transformer.resblocks.11.mlp.c_proj.bias',
                'transformer.resblocks.11.ln_2.weight',
                'transformer.resblocks.11.ln_2.bias',
                'ln_final.weight',
                'ln_final.bias',
                'text_projection']
    else:
        print(f"no {model_name}")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model, self.preprocess = clip.load('ViT-B/32', 'cpu')

        optim_params = get_optim_params('ViT-B/32')

        for name, param in self.model.named_parameters():
            if name not in optim_params:
                param.requires_grad = False

    def forward(self, text):
        text_features = self.model.encode_text(text)
        return text_features

if __name__ == "__main__":
    #x = torch.randn((1, 1, 40, 48, 40))
    # diffusion =  Diffusion()
    # FDG_path = config.whole_Tau+"/002S6695.nii" #+t1_name[0]
    # FDG = nifti_to_numpy(FDG_path)
    # FDG = np.expand_dims(FDG, axis=0)
    # FDG = np.expand_dims(FDG, axis=1)
    # FDG = torch.tensor(FDG)
    # FDG = FDG.to(config.device)
    
    # t1_path = config.whole_t1+"/002S6695.nii" #+t1_name[0]
    # t1 = nifti_to_numpy(t1_path)
    # t1 = np.expand_dims(t1, axis=0)
    # t1 = np.expand_dims(t1, axis=1)
    # t1 = torch.tensor(t1)
    # t1 = t1.to(config.device)
    # #y = torch.randn((1, 1, 96, 128, 96))

    # image = nib.load(config.path)

    # indices = list(range(diffusion.noise_steps))[::-1]
    # for i in tqdm(indices):
    #     t = torch.tensor([i] * FDG.shape[0], device=config.device)
    #     noise = torch.randn_like(FDG)
    #     FDG_t = diffusion.noise_images(FDG, t1, t, noise)
    #     FDG_out = FDG_t.detach().cpu().numpy()
    #     FDG_out = np.squeeze(FDG_out)
    #     FDG_out = FDG_out.astype(np.float32)
    #     FDG_out=nib.Nifti1Image(FDG_out,image.affine)
    #     nib.save(FDG_out,"data/Tau_view/"+str(i)+"_002S6695")
    
    # t1 = diffusion.prior_sample(t1, noise)
    # t1 = t1.detach().cpu().numpy()
    # t1 = np.squeeze(t1)
    # t1 = t1.astype(np.float32)
    # t1=nib.Nifti1Image(t1,image.affine)
    # nib.save(t1,"data/t1_view/"+str(i)+"_002S6695")

    x = torch.randn((1, 1, 128, 128, 128))
    t = torch.randint(15, (1,))
    net = Net()
    Tau_text = f"Synthesize an Tau-PET scan for a 65-year-old female subject diagnoised as AD with Mini-Mental Status Examination of 30"
    y = net(clip.tokenize(Tau_text))
    print(y.shape)
    model = UNet()
    #model = Discriminator()
    #print(model)

    #model = Discriminator()
    preds = model(x,t,y)
    print(preds.shape)
    #print(preds)