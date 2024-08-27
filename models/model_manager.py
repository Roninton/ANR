import torch
from torch import nn
import torch.nn.functional as F

from einops import reduce

import sys
sys.path.extend(["../../"])
from utils import *
from models import register
from loss.perceptual_loss import VGGLoss
from .hyper_network import HyperNetwork

@register('ModelMangager')
class ModelMangager(nn.Module):
    def __init__(
            self,
            hypernet,
            device,
            image_shape,
            target = "image", # image or nerf
            loss_type='l1',
            is_norm_01=True,
    ):
        super().__init__()
        assert isinstance(hypernet,HyperNetwork)
        self.hypernet = hypernet
        self.device = device
        self.hypo_model = hypernet.hyponet
        self.is_norm_01 = is_norm_01
        self.image_shape = [image_shape,image_shape] if isinstance(image_shape,int) else image_shape
        self.loss_type = loss_type
        self.target = "image" if target != "nerf" else "nerf"

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'perceptual':
            return VGGLoss().cuda()
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')    

    def hypo_pred(self, weights, b_h_w, coord_noise = False, **kw_args):
        # generate input coord
        b, h, w = b_h_w
        coord = make_coord_grid((h, w), (-1, 1)).to(self.device)
        coord = coord.unsqueeze(0)
        coord = coord.repeat(b, 1, 1, 1)

        if coord_noise:
            noise_max = 1 / max(h,w) # half of a grid
            magnitude, threshold = 0.3*noise_max, 0.5*noise_max # hyper-parameters

            noise = torch.randn(coord.shape,device=coord.device)
            noise = (magnitude * noise).clip(-threshold,threshold)
            coord = coord + noise # TODO: hyper para noise ratio

        # pred
        self.hypo_model.set_params(weights)
        img_pred = self.hypo_model(coord)
        img_pred = img_pred.view(b,h,w,-1).permute(0,3,1,2) # b, c, h, w
        return img_pred

    def forward_nerf(self, data, **kw_args):
        data = {k: v.cuda() for k, v in data.items()}
        query_imgs = data.pop('query_imgs')

        # forwarding
        model_out = self.hypernet(data, time=None)

        # sample n rays for loss computing
        B, n, c, H, W = query_imgs.shape
        rays_o, rays_d = poses_to_rays(data['query_poses'], H, W, data['query_focals'])
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')
        gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')

        ray_ids = np.random.choice(rays_o.shape[1], kw_args['train_n_rays'], replace=False)
        rays_o, rays_d, gt = map((lambda _: _[:, ray_ids, :]), [rays_o, rays_d, gt])
        pred = self.hypernet.get_pixels_from_NeRF(
            model_out, rays_o, rays_d,
            near=data['near'][0],
            far=data['far'][0],
            points_per_ray=kw_args["points_per_ray"],
            mini_batch_size=kw_args["mini_batch_size"],
            is_train=True)
        
        # evaluate
        more_info = dict()
        mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
        psnr = (-10 * torch.log10(mses)).mean()
        more_info["psnr"] = psnr
        loss = mses.mean()

        img_pred, gt = None, None # not return a full image
        return img_pred, gt, loss, more_info

    def forward_image(self, img, **kw_args):
        ## image case
        b, c, h, w, device = *img.shape, img.device,
        assert h == self.image_shape[0] and w == self.image_shape[1], \
            f'height and width of image must be {self.image_shape}, but get{(h,w)}'

        # input
        img_in = img
        peak_noise_square = 1
        if self.is_norm_01:
            peak_noise_square = 4 # pixel in (-1,1) -> 2**2 = 4
            img = normalize_to_neg_one_to_one(img)

        # predict
        model_out = self.hypernet(img, time = None)
        img_pred = self.hypo_pred(model_out, [b,h,w], **kw_args)

        # loss
        mses = self.loss_fn(img_pred, img, reduction='none')
        loss = reduce(mses, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        # build ret
        more_info = dict()
        mses = mses.reshape(b, -1).mean(dim=-1)
        psnr = (10 * torch.log10(peak_noise_square / mses)).mean()
        more_info["psnr"] = psnr
        if self.is_norm_01:
            img_pred = unnormalize_to_zero_to_one(img_pred)
        img_pred = img_pred.clamp(min=0, max=1)
        return img_pred, img_in, loss, more_info

    def forward(self, img, **kw_args):
        if self.target == "image":
            return self.forward_image(img, **kw_args)
        else:
            return self.forward_nerf(img, **kw_args)

    def infer_image(self, img, target_resolution = None, coord_noise=False):
        b, c, h, w, device = *img.shape, img.device
        assert h == self.image_shape[0] and w == self.image_shape[1], \
            f'height and width of image must be {self.image_shape}, but get{(h,w)}'
        target_h, target_w = target_resolution if target_resolution is not None else [h,w]

        # input
        B, gt = img.shape[0], img
        if self.is_norm_01:
            img = normalize_to_neg_one_to_one(img)
        hypo_weight = self.hypernet(img, time = None)
        img_pred = self.hypo_pred(hypo_weight, (b,target_h,target_w), coord_noise)
        if self.is_norm_01:
            img_pred = unnormalize_to_zero_to_one(img_pred)

        # evaluate
        more_info = dict()
        if  target_h == h and target_w == w:
            mses = ((img_pred - gt)**2).reshape(B, -1).mean(dim=-1)
            psnr = (-10 * torch.log10(mses)).mean()
            more_info["psnr"] = psnr
            loss = mses.mean()
        else:
            more_info["psnr"] = -1
            loss = 0
        return img_pred, gt, loss, more_info

    def infer_nerf(self, data, **kw_args):
        data = {k: v.cuda() for k, v in data.items()}
        query_imgs = data.pop('query_imgs')

        # forwarding
        model_out = self.hypernet(data, time=None)

        # sample rays
        B, n, c, H, W = query_imgs.shape
        rays_o, rays_d = poses_to_rays(data['query_poses'], H, W, data['query_focals'])
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')
        gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')

        # for visualize
        resolution, downsample = query_imgs.shape[-1], kw_args["downsample"]
        rays_o_sqr, rays_d_sqr, gt_sqr = square_sample_rays(rays_o, rays_d, gt, resolution=resolution, downsample=downsample, H=H,W=W)
        with torch.no_grad():
            pred = self.hypernet.get_pixels_from_NeRF(
                model_out, rays_o_sqr, rays_d_sqr,
                near=data['near'][0],
                far=data['far'][0],
                points_per_ray=kw_args["points_per_ray"],
                mini_batch_size=kw_args["mini_batch_size"],
                is_train=False)
        img_pred = pred.view(B, n, resolution // downsample, resolution // downsample, c)[:, 0, :, :, :]
        img_pred = img_pred.clamp(min=0, max=1)
        img_pred = img_pred.permute(0, 3, 1, 2)
        gt_sqr = gt_sqr.view(B, n, resolution // downsample, resolution // downsample, c)[:, 0, :, :, :]
        gt_sqr = gt_sqr.clamp(min=0, max=1)
        gt = gt_sqr.permute(0, 3, 1, 2) # swith gt to gt_sqr
        
        # evaluate
        more_info = dict()
        mses = ((img_pred - gt)**2).reshape(B, -1).mean(dim=-1)
        psnr = (-10 * torch.log10(mses)).mean()
        more_info["psnr"] = psnr
        loss = mses.mean()

        return img_pred, gt, loss, more_info
    
    def infer(self, img, **kw_args):
        if self.target == "image":
            return self.infer_image(img, **kw_args)
        else:
            return self.infer_nerf(img, **kw_args)
