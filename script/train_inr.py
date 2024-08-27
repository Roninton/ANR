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


class Trainer(object):
    def __init__(
            self,
            model,
            dataset_kwargs,
            results_folder,
            coord_noise,
            train_batch_size=16,
            train_lr=5e-5,
            train_num_steps=100000,
            log_every=100,
            save_every=1000,
            gradient_accumulate_every=1,
            ema_decay=0.995,
            ema_update_every=10,
            amp=False,  # turn on mixed precision
            opt_type="adam", # rmsprop, adam
            adam_betas=(0.9, 0.99),
            fp16=False,
            split_batches=True,
            # for nerf
            train_nerf=False,
            train_n_rays=128,
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
        self.train_n_rays = train_n_rays

        # dataset and dataloader

        self.ds = make_dataset(dataset_kwargs[0])(**dataset_kwargs[1])
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        if opt_type == "rmsprop":
            self.opt = RMSprop(model.parameters(), lr=train_lr)
        elif opt_type == "adam":
            self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)
        else:
            raise ValueError("opt not supported.")

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
            
            self.train_folder = Path(results_folder+"/training")
            self.train_folder.mkdir(exist_ok=True)
            config_copy(config_file,results_folder)
            self.info_rec = open(self.train_folder/"running_log.csv","a")

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # constant forward kw_args
        if train_nerf:
            self.forward_kw_args = {
                "train_n_rays":train_n_rays,
                "points_per_ray":128,
                "mini_batch_size":32,
            }
            self.vis_kw_args = {
                "downsample":2,
                "points_per_ray":128,
                "mini_batch_size":32,
            }
        else:
            self.forward_kw_args = {}
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
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        running_info = None
        # with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

        while self.step < self.train_num_steps:

            total_loss = 0.
            # total_reg_loss = 0.

            for _ in range(self.gradient_accumulate_every):
                data = next(self.dl)
                try:
                    data = data.to(device)
                except:
                    pass # not tensor type, do nothing

                with self.accelerator.autocast():
                    img, gt, loss, more_info = self.model(data,
                        coord_noise=self.coord_noise,
                        **self.forward_kw_args
                        )
                    # print(loss)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                self.accelerator.backward(loss)

            accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            info_str = f"step: {self.step}, loss: {total_loss:.4f}"
            more_str = ""
            for key in more_info:
                more_str += f"# {key}: {more_info[key]:.4f}"
            info_str = info_str+more_str
            print(info_str)
            
            # save running info
            if running_info is None:
                running_info = more_info
                for key in more_info:
                    if torch.isinf(running_info[key]):
                        running_info[key] = 0
                running_info["loss"] = total_loss
            else:
                for key in more_info:
                    if torch.isinf(more_info[key]):
                        running_info[key] += 0
                    else:
                        running_info[key] += more_info[key]
                running_info["loss"] += total_loss
                
            accelerator.wait_for_everyone()

            self.opt.step()
            self.opt.zero_grad()

            accelerator.wait_for_everyone()

            self.step += 1
            if accelerator.is_main_process:
                self.ema.to(device)
                self.ema.update()
                
                if self.step != 0 and self.step % self.log_every == 0:
                    if self.step == self.log_every:
                        info_str = f"step"
                        for key in running_info:
                            info_str += f",{key}"
                        info_str += "\n"
                        self.info_rec.write(info_str)
                        self.info_rec.flush()

                    info_str = f"{self.step}"
                    for key in running_info:
                        info_str += f",{running_info[key]/self.log_every:.4f}"
                    info_str += "\n"
                    self.info_rec.write(info_str)
                    self.info_rec.flush()
                    running_info = None

                if self.step != 0 and self.step % self.save_every == 0:
                    self.ema.ema_model.eval()
                    img, gt, loss, more_info = self.model.infer(data, coord_noise=False,
                        **self.vis_kw_args,
                    )

                    with torch.no_grad():                        
                        utils.save_image(gt, str(self.train_folder / '_temp_gt.png'),
                                            nrow=int(math.sqrt(self.batch_size)))
                        utils.save_image(img, str(self.train_folder / f'sample-{self.step}.png'),
                                            nrow=int(math.sqrt(self.batch_size)))
                    self.save("")

                    # save ground truth
                    if self.step % (10*self.save_every) == 0:
                        utils.save_image(gt, str(self.train_folder / f'gt_{self.step}.png'),
                                        nrow=int(math.sqrt(self.batch_size)))
                    
                    # save model with "step" post name
                    if self.step % (100*self.save_every) == 0:
                        self.save(f"_{self.step}")
                        
        accelerator.print('training complete')
        self.info_rec.close()


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
    train_lr = params["train_lr"] if "train_lr" in params else 5e-5
    opt_type = 'adam' if (not "opt_type" in params) else params["opt_type"]
    train_nerf = (params["train_target"] == "nerf") if "train_target" in params else False
    train_n_rays = params["train_n_rays"] if "train_n_rays" in params else 128

    # set result folder and copy config
    Path("./output/").mkdir(exist_ok=True)
    results_folder = os.path.join("./output/",params['model_comment'])
    trainer = Trainer(
        model,
        dataset_kwargs,
        results_folder=results_folder,
        coord_noise=coord_noise,
        # training para
        train_batch_size=params['batch_size'],
        train_lr= train_lr, # 5e-5
        train_num_steps=params['train_num_steps'], 
        log_every=100,
        save_every=params['save_every'],
        gradient_accumulate_every=params['gradient_accumulate_every'],
        ema_decay=0.995,  # exponential moving average decay
        ema_update_every=10,
        opt_type=opt_type,
        adam_betas=(0.9, 0.99),
        fp16=False,
        split_batches=True,
        # for nerf
        train_nerf=train_nerf,
        train_n_rays=train_n_rays,
    )
    
    if not params['milestone'] is None:
        trainer.load(params['milestone'])
    trainer.train()    
