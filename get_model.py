from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as utils

from utils import *
from models import *
from datasets import *

# dict prepare
init_method_dict = dict()

def init_method_register(name):
    def decorator(cls):
        init_method_dict[name] = cls
        return cls
    return decorator

def get_init_method(name):
    return init_method_dict[name]

# get method
def get_model_from_params(params,device):
    # constants
    milestone = params['milestone']
    model_comment = params['model_comment']
    dataset_kwargs = params['dataset_kwargs']
    # model init
    model = get_loaded_model(params,device).to(device)
    print("inr management model init.")
    return model, dataset_kwargs ,milestone, model_comment

def get_loaded_model(params,device):
    model_framework = params['model_framework']['frame_work']
    model = get_init_method(model_framework)(params,device)
    milestone = params['milestone']
    if not milestone is None:
        data = torch.load(milestone)
        print(data['step'])
        model.load_state_dict(data['model'],strict=False)
    return model

@init_method_register('NormalINR')
def get_inr_model(params,device):
    framework = params['model_framework']
    hyper_network = make_model(framework['inner_model'][0])(device = device,**framework['inner_model'][1]).to(device)
    manager_params = framework['outer_model']
    model_manager = make_model(manager_params[0])(
        hypernet = hyper_network,
        device = device,
        **manager_params[1],
    ).to(device)
    return model_manager
