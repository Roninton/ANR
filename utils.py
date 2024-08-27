import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from einops import rearrange, reduce
from functools import partial
from torch import einsum
import os
import shutil

import numpy as np

### config related ###
def config_copy(config_path,copy_root):
    filename = os.path.split(config_path)[-1]
    copy_path = os.path.join(copy_root,filename)
    source_path = os.path.abspath(config_path)
    target_path = os.path.abspath(copy_path)
    shutil.copyfile(source_path, target_path)

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

### HypoNet Util ###
def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb)

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

def init_w(shape):
    weight = torch.empty(shape[1], shape[0])
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    return weight.t().detach()

### make_coord_grid ###
def make_coord_grid(shape, range, device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return grid

### sample function ###
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

### helpers functions ###
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(time, *args, **kwargs):
    return time


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(time):
    return (time + 1) * 0.5


# gaussian diffusion trainer class
def extract(a, time, x_shape):
    b, *_ = time.shape
    out = a.gather(-1, time)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))



## NeRF volume rendering
def poses_to_rays(poses, image_h, image_w, focal):
    """
        Pose columns are: 3 camera axes specified in world coordinate + 1 camera position.
        Camera: x-axis right, y-axis up, z-axis inward.
        Focal is in pixel-scale.

        Args:
            poses: (... 3 4)
            focal: (... 2)
        Returns:
            rays_o, rays_d: shape (... image_h image_w 3)
    """
    device = poses.device
    bshape = poses.shape[:-2]
    poses = poses.view(-1, 3, 4)
    focal = focal.view(-1, 2)
    bsize = poses.shape[0]

    x, y = torch.meshgrid(torch.arange(image_w, device=device), torch.arange(image_h, device=device), indexing='xy') # h w
    x, y = x + 0.5, y + 0.5 # modified to + 0.5
    x, y = x.unsqueeze(0), y.unsqueeze(0) # h w -> 1 h w
    focal = focal.unsqueeze(1).unsqueeze(1) # b 2 -> b 1 1 2
    dirs = torch.stack([
        (x - image_w / 2) / focal[..., 0],
        -(y - image_h / 2) / focal[..., 1],
        -torch.ones(bsize, image_h, image_w, device=device)
    ], dim=-1) # b h w 3

    poses = poses.unsqueeze(1).unsqueeze(1) # b 3 4 -> b 1 1 3 4
    rays_o = poses[..., -1].repeat(1, image_h, image_w, 1) # b h w 3
    rays_d = (dirs.unsqueeze(-2) * poses[..., :3]).sum(dim=-1) # b h w 3

    rays_o = rays_o.view(*bshape, *rays_o.shape[1:])
    rays_d = rays_d.view(*bshape, *rays_d.shape[1:])
    return rays_o, rays_d

def adaptive_sample_rays(rays_o, rays_d, gt, n_sample):
    B = rays_o.shape[0]
    inds = []
    fg_n_sample = n_sample // 2
    for i in range(B):
        fg = ((gt[i].min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()
        if fg_n_sample <= len(fg):
            fg = np.random.choice(fg, fg_n_sample, replace=False)
        else:
            fg = np.concatenate([fg, np.random.choice(fg, fg_n_sample - len(fg), replace=True)], axis=0)
        rd = np.random.choice(rays_o.shape[1], n_sample - fg_n_sample, replace=False)
        inds.append(np.concatenate([fg, rd], axis=0))

    def subselect(x, inds):
        t = torch.empty(B, len(inds[0]), 3, dtype=x.dtype, device=x.device)
        for i in range(B):
            t[i] = x[i][inds[i], :]
        return t
    return subselect(rays_o, inds), subselect(rays_d, inds), subselect(gt, inds)

def square_sample_rays(rays_o, rays_d, gt, resolution=32, H=128,W=128, sample_type='downsample',downsample=2,max_extra_sample_times=3):
    def shift2index(centers_B, resolution=48, H=128, stride=1):

        B = centers_B.shape[0]
        x = range(-resolution // 2, resolution // 2)
        y = range(-resolution // 2, resolution // 2)
        index = np.meshgrid(x, y)

        index = np.array(index)
        if stride != 1:
            start_x, start_y = np.random.choice(np.array(range(stride)),size=(2))
            index = index[:,start_x::stride,start_y::stride]

            centers_B = centers_B[:,:,start_x::stride,start_y::stride]

        index_B = einops.repeat(index, 'c h w -> repeat c h w', repeat=B)

        index_final = centers_B + index_B
        index_final = index_final[:, 0, :, :] * H + index_final[:, 1, :, :]
        inds = einops.rearrange(index_final, 'b h w -> b (h w)')

        return inds

    def ratio_chcek(gt, ratio=0.2):
        fg = ((gt[0].min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()

        return fg.shape[0] > gt.shape[1] * ratio
    
    def range_check(centers):
        amin, amax = -1, 1
        bmin, bmax = -1, 1
        flag = (np.min(centers - [amin, bmin], axis=1) >= 0).all() and (np.max(centers - [amax, bmax], axis=1) <= 0).all()
        return flag

    def subselect(x, inds, B=1):
        t = torch.empty(B, len(inds[0]), 3, dtype=x.dtype, device=x.device)
        for i in range(B):
            t[i] = x[i][inds[i], :]
        return t


    B = rays_o.shape[0]

    if sample_type == 'random':
        centers = np.random.randint(low=resolution//2, high=H-resolution//2-1, size=(B,2))
        centers_B = einops.repeat(centers, 'b c -> b c h w', h=resolution, w=resolution)
        inds = shift2index(centers_B,resolution=resolution, H=H)
        return subselect(rays_o, inds, B=B), subselect(rays_d, inds,B=B), subselect(gt,inds,B=B)

    elif sample_type == 'center':
        centers = []
        # times = 0
        for i in range(B):
            fg = ((gt[i].min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()
            flag = False
            times = 0
            while flag == False:
                fg_choose = np.random.choice(fg, size=(1))
                times += 1
                center_H = fg_choose//H
                center_W = np.mod(fg_choose,H)
                flag = center_H > resolution//2 and center_H < H-resolution//2 and center_W > resolution//2 and center_W < W-resolution//2
                # if times != 1:
                    # print('times:', times)
                if times > 3:
                    print('more than 3')
                    center_H, center_W= np.array([H//2]), np.array([W//2])
                    break
            centers.append(np.array([center_H, center_W]))

        centers = np.array(centers)[:,:,0]
        centers_B = einops.repeat(centers, 'b c -> b c h w', h=resolution, w=resolution)
        inds = shift2index(centers_B, resolution=resolution, H=H)
        return subselect(rays_o, inds, B=B), subselect(rays_d, inds, B=B), subselect(gt, inds, B=B)


    elif sample_type == 'gaussian':
        mean = np.array((0,0))
        cov = np.eye(2)*0.25
        size = (1,)
        times = 0
        inds_all = []
        for i in range(B):
            ratio_accepeted = False
            # fg = ((gt.min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()
            while ratio_accepeted == False and times <= B + max_extra_sample_times:
                range_accepted = False
                while range_accepted == False:
                    centers = np.random.multivariate_normal(mean, cov, size)
                    times += 1
                    range_accepted = range_check(centers)
                centers = (centers + 1) / 2 * (H-resolution) + resolution//2
                centers = centers.astype('int')

                centers_B = einops.repeat(centers, 'b c -> b c h w', h=resolution, w=resolution)
                inds = shift2index(centers_B,resolution=resolution, H=H)

                gt_sub = subselect(torch.unsqueeze(gt[i],dim=0), inds, B=1)
                ratio_accepeted = ratio_chcek(gt_sub,ratio=0.2)

            inds_all.append(inds[0])



        print('resample time: ', times - B)

        return subselect(rays_o, inds_all, B=B), subselect(rays_d, inds_all, B=B), subselect(gt, inds_all, B=B)

    elif sample_type == 'downsample':
        centers = np.array([H//2, W//2])
        centers_B = einops.repeat(centers, 'c -> b c h w', b =B, h=resolution, w=resolution)

        inds = shift2index(centers_B, resolution=resolution, H=H, stride=downsample) # b h w, 按照行列依次选择的index(int),index[i]>=0

        return subselect(rays_o, inds, B=B), subselect(rays_d, inds, B=B), subselect(gt, inds, B=B)
    
    else:
        print('non implementation')
        return None

def volume_rendering(nerf, rays_o, rays_d, near, far, points_per_ray, use_viewdirs, rand, need_depth=False):
    """
        Args:
            rays_o, rays_d: shape (b ... 3)
        Returns:
            pred: (b ... 3)
    """
    # Reference: https://github.com/tancik/learnit/blob/main/Experiments/shapenet.ipynb
    B = rays_o.shape[0]
    query_shape = rays_o.shape[1: -1]
    rays_o = rays_o.view(B, -1, 3)
    rays_d = rays_d.view(B, -1, 3)
    n_rays = rays_o.shape[1]
    device = rays_o.device

    # Compute 3D query points
    z_vals = torch.linspace(near, far, points_per_ray, device=device)
    z_vals = einops.repeat(z_vals, 'p -> n p', n=n_rays)
    if rand:
        d = (far - near) / (points_per_ray - 1) # modified as points_per_ray - 1
        z_vals = z_vals + torch.rand(n_rays, points_per_ray, device=device) * d

    pts = rays_o.view(B, n_rays, 1, 3) + rays_d.view(B, n_rays, 1, 3) * z_vals.view(1, n_rays, points_per_ray, 1)

    # Run network
    pts_flat = einops.rearrange(pts, 'b n p d -> b (n p) d')
    if not use_viewdirs:
        raw = nerf(pts_flat)
    else:
        viewdirs = einops.repeat(rays_d, 'b n d -> b n p d', p=points_per_ray)
        raw = nerf(pts_flat, viewdirs=viewdirs)
    raw = einops.rearrange(raw, 'b (n p) c -> b n p c', n=n_rays)

    # Compute opacities and colors
    rgb, sigma_a = raw[..., :3], raw[..., 3]
    rgb = torch.sigmoid(rgb) # b n p 3
    sigma_a = F.relu(sigma_a) # b n p

    # Do volume rendering
    dists = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], torch.ones_like(z_vals[:, -1:]) * 1e-3], dim=-1) # n p
    alpha = 1. - torch.exp(-sigma_a * dists) # b n p, dist越大alpha越大,alpha应该是遮光系数
    trans = torch.clamp(1. - alpha + 1e-10, max=1.) # b n p， trans是当前小块的透光比例
    trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], dim=-1) # 当前dist的光强（连乘之前），所有的初始光强都是1
    weights = alpha * torch.cumprod(trans, dim=-1) # b n p

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2) # rgb的加权平均
    acc_map = torch.sum(weights, dim=-1) # 如果全部weight加起来不为一，说明透光了，能看到背景
    rgb_map = rgb_map + (1. - acc_map).unsqueeze(-1) # white background

    rgb_map = rgb_map.view(B, *query_shape, 3)
    
    if not need_depth:
        return rgb_map
    else:
        depth_map = torch.sum(weights * z_vals, dim=-1)
        depth_map = depth_map.view(B, *query_shape)
        return rgb_map, depth_map

def batched_volume_rendering(nerf, rays_o, rays_d, *args, batch_size=None, **kwargs):
    """
        Args:
            rays_o, rays_d: (b ... 3)
        Returns:
            pred: (b ... 3)
    """
    B = rays_o.shape[0]
    query_shape = rays_o.shape[1: -1]
    rays_o = rays_o.view(B, -1, 3)
    rays_d = rays_d.view(B, -1, 3)

    ret = []
    ll = 0
    while ll < rays_o.shape[1]:
        rr = min(ll + batch_size, rays_o.shape[1])
        rays_o_ = rays_o[:, ll: rr, :]
        rays_d_ = rays_d[:, ll: rr, :]
        ret.append(volume_rendering(nerf, rays_o_, rays_d_, *args, **kwargs))
        ll = rr
    ret = torch.cat(ret, dim=1)

    ret = ret.view(B, *query_shape, 3)
    return ret
