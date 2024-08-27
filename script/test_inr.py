import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import math
import yaml
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from torchvision import transforms as utils

from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator

from utils import *
from models import *
from datasets import *
from get_model import *


class Tester(object):
    def __init__(
            self,
            model,
            dataset_kwargs,
            results_folder,
            coord_noise,
            train_batch_size=16,
            train_num_steps=100000,
            log_every=100,
            save_every=1000,
            gradient_accumulate_every=1,
            ema_decay=0.995,
            ema_update_every=10,
            amp=False,  # turn on mixed precision
            fp16=False,
            split_batches=True,
            train_nerf=False,
            # for test and infer
            sr_mini_batch=1,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        self.model = model

        self.coord_noise = coord_noise
        self.log_every = log_every
        self.save_every = save_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.train_nerf = train_nerf

        # dataset and dataloader
        if "imageset_kwargs" in dataset_kwargs[1]:
            dataset_kwargs[1]["imageset_kwargs"]["split"] = "test_val" # or val
        else:
            dataset_kwargs[1]["split"] = "test" # or val

        self.ds = make_dataset(dataset_kwargs[0])(**dataset_kwargs[1])
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # for test and infer
        self.sr_mini_batch = sr_mini_batch

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)

            self.results_folder = Path(results_folder)
            try:
                self.results_folder.mkdir(exist_ok=False)
            except FileExistsError as e:
                print("File already Exists, continue?(y)")
                c = input()[0]
                if c == 'y':
                    self.results_folder.mkdir(exist_ok=True)
                else:
                    exit(0)
            
            self.test_folder = Path(results_folder+"/testing")
            self.test_folder.mkdir(exist_ok=True)
            config_copy(config_file,results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model = self.accelerator.prepare(self.model)

        # constant forward kw_args
        if train_nerf:
            self.vis_kw_args = {
                "downsample":2,
                "points_per_ray":128,
                "mini_batch_size":32,
            }
        else:
            self.vis_kw_args = {}
        
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        # torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        torch.save(data, str(self.results_folder / f'model_inner_test{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        # data_path = str(self.results_folder / f'model-{milestone}.pt') # original use
        data_path = milestone
        data = torch.load(data_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def test(self,sr_ratio=3.0):
        accelerator = self.accelerator
        device = accelerator.device

        data = next(self.dl)
        with torch.no_grad():     
            with self.accelerator.autocast():
                img, gt, loss, more_info = self.model.infer(data, 
                    coord_noise=False,
                    **self.vis_kw_args,
                )
                psnr = more_info["psnr"]
                # super-resolution for image reconstruction
                if not self.train_nerf:
                    if self.sr_mini_batch == self.batch_size:
                        sr_img, *_ = self.super_resolution(data,ratio=sr_ratio)
                    else:
                        i = 0
                        sr_imgs = []
                        while i < self.batch_size:
                            n_i = min(i+self.sr_mini_batch,self.batch_size)
                            di = data[i:n_i]
                            i = n_i
                            sr_img_i, _ = self.super_resolution(di,ratio=sr_ratio)
                            sr_imgs.append(sr_img_i)
                        sr_img = torch.cat(sr_imgs,dim=0)
        
        print('trained step: ', self.step, ' # psnr: ', psnr)

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()
            
            self.ema.ema_model.eval()
            with torch.no_grad():                        
                utils.save_image(gt, str(self.test_folder / 'target.png'),
                    nrow=int(math.sqrt(self.batch_size)))
                utils.save_image(img, str(self.test_folder / 'img_pred.png'),
                    nrow=int(math.sqrt(self.batch_size)))
                if not self.train_nerf:
                    utils.save_image(sr_img, str(self.test_folder / 'sr_img_pred.png'),
                        nrow=int(math.sqrt(self.batch_size)))

    def super_resolution(self, img, ratio=2.0):
        b, c, h, w, device = *img.shape, img.device,
        sr_h, sr_w = int(ratio*h), int(ratio*w)
        
        img_pred, gt, loss, more_info = self.model.infer(img, target_resolution = [sr_h, sr_w])
        return img_pred, gt, loss, more_info

if __name__ == "__main__":
    global config_file
    config_file = "./configs/test/Celeba32_anr_d1_relu_for_testing.yaml"
    # config_file = "./configs/test/ShapenetCar128_anr_d1_for_testing.yaml"

    with open(file=config_file, mode='r', encoding='utf-8') as f:
        input_f = f.read()
        params = yaml.load(input_f, Loader=yaml.FullLoader)

    # get and init models, get dataset args also
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ret = get_model_from_params(params,device)
    model, dataset_kwargs ,milestone, model_comment = ret

    # optional default params
    coord_noise = params["coord_noise"] if "coord_noise" in params else True
    train_nerf = (params["train_target"] == "nerf") if "train_target" in params else False

    # set result folder and copy config
    Path("./output/").mkdir(exist_ok=True)
    results_folder = os.path.join("./output/",params['model_comment'])
    tester = Tester(
        model,
        dataset_kwargs,
        results_folder=results_folder,
        coord_noise=coord_noise,
        # training para
        train_batch_size=params['batch_size'],
        train_num_steps=params['train_num_steps'], 
        log_every=100,
        save_every=params['save_every'],
        gradient_accumulate_every=params['gradient_accumulate_every'],
        ema_decay=0.995,  # exponential moving average decay
        ema_update_every=10,
        fp16=False,
        split_batches=True,
        # for nerf
        train_nerf=train_nerf,
        # for test and infer
        sr_mini_batch = 1,
    )
    
    if not params['milestone'] is None:
        tester.load(params['milestone'])
    tester.test()    
