import torch
from torchvision import transforms as T, utils
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import os
import sys
sys.path.extend(['../','./'])
#from diff_util import *
from datasets.datasets import register
from utils import *
    
@register("ImageSet")
class ImageSet(Dataset):
    def __init__(self, root_path, split):
        if split == 'train':
            s, t = 0, 0.75
        elif split == 'val':
            s, t = 0.75, 0.95
        elif split == 'test':
            s, t = 0.95, 1
        elif split == 'test_val':
            s, t = 0.75, 1
        else:
            s, t = 0, 1
        self.data = [os.path.join(root_path,p) for p in os.listdir(root_path) 
                    if p.lower().endswith(".png") or 
                    p.lower().endswith(".jpg") or
                    p.lower().endswith(".jpeg")]
        # split
        datalen = len(self.data)
        s, t = int(datalen*s), int(datalen*t)
        self.data = self.data[s:t]
        # loop data
        self.datalen = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        # print(img.size)
        return img

@register("ImageSetDataset")
class ImageSetDataset(Dataset):
    def __init__(self,
                 imageset_kwargs,
                 resize,
                 convert_image_to=None,
                 augment_horizontal_flip=False,
                 ):
        self.resize = resize
        self.imageset = ImageSet(**imageset_kwargs)
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(resize),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(resize),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, idx):
        x = self.transform(self.imageset[idx])
        if x.shape[0] != 3:
            x = x.repeat(3,1,1)
        return x
