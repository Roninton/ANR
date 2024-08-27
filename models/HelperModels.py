import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
import einops
from models import register
from functools import partial

import sys
sys.path.extend(["../","./"])
from utils import *


@register("MLP")
class MLP(nn.Module):
    def __init__(self
                 , in_channels
                 , out_channels
                 , hidden_channels=3
                 , num_layers=2
                 , dropout=0.0
                 , batchnorm=False
                 , last_layer="none"  # "sigmoid" "relu" "softmax" ""log_softmax"" "none"
                 , **kwargs):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        self.last_layer = last_layer
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

        self.channels = out_channels
        self.out_dim = out_channels
        self.in_dim = in_channels
        self.self_condition = False
        self.learned_sinusoidal_cond = False

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, time=None, x_self_cond=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        if self.last_layer == "sigmoid":
            x = F.sigmoid(x)
        elif self.last_layer == "relu":
            x = F.relu(x)
        elif self.last_layer == "softmax":
            x = F.softmax(x, dim=-1)
        elif self.last_layer == "log_softmax":
            x = F.log_softmax(x, dim=-1)
        else:  # do nothing
            pass
        return x

# img tokenizer
@register("ImgTokenizer")
class ImgTokenizer(nn.Module):
    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)
        n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * ((input_size[1] + padding[1] * 2)  // patch_size[1])
        self.posemb = nn.Parameter(torch.randn(n_patches, dim))

    def token_shape(self):
        return self.posemb.shape
    
    def forward(self, x):
        p = self.patch_size
        # print(x.shape)
        # unfold -> 将x按照p进行分块，每一块为(c,p,p)的展平，并且按照strid移动，总共需要移动的次数即l
        x = F.unfold(x, p, stride=p, padding=self.padding) # (B, C * p * p, L)
        # print("tokenizer unfold:",x.shape)
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape,self.prefc(x).shape,self.posemb.unsqueeze(0).shape)
        x = self.prefc(x) + self.posemb.unsqueeze(0)
        # print(x.shape)
        # exit(0)
        return x

# shapenet tokenizer
@register('nvs_tokenizer')
class NvsTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * (img_channels + 3 + 3), dim)
        self.grid_shape = ((input_size[0] + padding[0] * 2) // patch_size[0],
                           (input_size[1] + padding[1] * 2) // patch_size[1])

    def forward(self, data):
        imgs = data['support_imgs']
        B = imgs.shape[0]
        H, W = imgs.shape[-2:]
        rays_o, rays_d = poses_to_rays(data['support_poses'], H, W, data['support_focals'])
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b n c h w')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b n c h w')

        x = torch.cat([imgs, rays_o, rays_d], dim=2)
        x = einops.rearrange(x, 'b n d h w -> (b n) d h w')
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = einops.rearrange(x, '(b n) ppd l -> b (n l) ppd', b=B)

        x = self.prefc(x)
        return x



# sinusoidal positional embeds
@register("LearnedSinusoidalPosEmb")
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # n*1 @ 1*n = n*n, emb = [1,....,10e-4]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

@register("LearnedSinusoidalPosEmb")
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi  # 2*pi * x @ w, shape = d*d
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

@register("MultifeatureSinusoidalPosEmb")
class MultifeatureSinusoidalPosEmb(nn.Module):
    def __init__(self, dim = 16, feat_len=1):
        super().__init__()
        self.dim = dim
        self.feat_len = feat_len
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb[None, ...].repeat(self.feat_len,1)
        self.emb = torch.nn.Parameter(emb,requires_grad = False)

    def forward(self, x):
        x = x[..., None]
        emb = x * self.emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print("emb",emb.shape)
        out = torch.cat((x, emb), dim=-1)
        out = out.flatten(-2)
        return out

@register("LearnedMultifeatureSinusoidalPosEmb")
class LearnedMultifeatureSinusoidalPosEmb(nn.Module):
    def __init__(self, dim = 16, feat_len=1):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.feat_len = feat_len
        self.weights = nn.Parameter(torch.randn(feat_len,half_dim))

    def forward(self, x):
        x = x[...,None]
        freqs = x * self.weights * 2 * math.pi  # 2*pi * x @ w, shape = d*d
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        fouriered = fouriered.flatten(-2)
        return fouriered

# small helper modules
@register("Residual")
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

@register("WeightStandardizedConv2d")
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

@register("LayerNorm")
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

@register("PreNorm")
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kw_args):
        return self.fn(self.norm(x),**kw_args)

@register("Block")
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

@register("ResnetBlock")
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

@register("Attention")
class Attention(nn.Module):
    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None, **kw_args):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1) # split to 2 block
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

@register("Attention2D")
class Attention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda time: rearrange(time, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

@register("LinearAttention")
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda time: rearrange(time, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

@register("FeedForward")
class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kw_args):
        return self.net(x)


@register("Sine")
class Sine(nn.Module):
  """Applies a scaled sine transform to input: out = sin(w0 * in)."""

  def __init__(self, w0: float = 1.):
    """Constructor.

    Args:
      w0 (float): Scale factor in sine activation (omega_0 factor from SIREN).
    """
    super().__init__()
    self.w0 = w0

  def forward(self, x):
    return torch.sin(self.w0 * x)