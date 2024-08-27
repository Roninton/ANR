import torch
from torchvision import transforms as T, utils
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import os
import sys
sys.path.extend(['../','./'])

from utils import *
from datasets.datasets import register
    
@register("ERA5_npy")
class ERA5_npy(Dataset):
    def __init__(self, root_path, resize, split,
        convert_image_to=None,
        augment_horizontal_flip=False,
        shift= -260,
        scale= 0.01,
    ):
        if split == 'train':
            s, t = 0, 0.85
        elif split == 'val':
            s, t = 0.85, 0.95
        elif split == 'test':
            s, t = 0.95, 1
        elif split == 'test_val':
            s, t = 0.85, 1
        else:
            s, t = 0, 1
        self.data = [os.path.join(root_path,p) for p in os.listdir(root_path) 
                    if p.lower().endswith(".npy")]
        # split
        datalen = len(self.data)
        s, t = int(datalen*s), int(datalen*t)
        self.data = self.data[s:t]
        self.datalen = len(self.data)
        # transform
        self.resize = resize
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(resize),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(resize),
            # T.ToTensor(),
        ])
        self.shift = shift
        self.scale = scale


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(self.data[idx])[None,:,:]
        x = torch.tensor(x,dtype=torch.float32)
        # print(x.shape) # torch.Size([1, 721, 1440])
        x = self.transform(x)
        x = self.scale*(x+self.shift)
        # if x.shape[0] != 3:
        #     x = x.repeat(3,1,1)
        return x
    