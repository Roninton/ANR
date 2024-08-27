import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import sys
sys.path.extend(["../../","../"])
from models import register
from utils import batched_linear_mm, init_wb

@register("HypoMlpIPC")
class HypoMlpIPC(nn.Module):
    def __init__(self, 
        depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim,
        modulated_layer_id = 1,
        expected_r_width=32, # expected_r_width not used for default
        activation = None, largest_dim_padding = 8, out_bias=0, pe_sigma=1024):
        super().__init__()
        self.use_pe = use_pe
        if pe_dim % 2 != 0:
            print(f"pe_dim == {pe_dim} must be even number, automatic minus 1 ->{pe_dim-1}")
            pe_dim -= 1
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.expected_r_width = expected_r_width
        self.activation = activation
        self.largest_dim_padding = largest_dim_padding
        self.param_shapes = dict()
        self.global_param_shapes = dict()

        self.modulated_layer_id = modulated_layer_id
        self.global_params = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim

        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            if i != modulated_layer_id:
                self.global_params[f"wb{i}"] = nn.Parameter(init_wb((last_dim + 1, cur_dim)))
                self.global_param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            else:
                self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.global_params = nn.ParameterDict(self.global_params)
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias
        self.in_dim = in_dim

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        # w = [0,a1,...,self.pe_sigma], length = self.pe_dim // 2, ai+1/ai  is const
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        # unsqueeze @ shape = [a,b,c,..., channel, 1]
        # matmul @ [a,b,c,..., channel, len(w)], len(w) = elf.pe_dim // 2
        # view @ [a,b,c,..., channel * len(w)], len(w) = elf.pe_dim // 2
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        # cat[cos,sin] @ [a,b,c,..., channel * len(w) *2], len(w) = elf.pe_dim // 2
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x, **kw_args):
        assert x.shape[-1] == self.in_dim, f"invalid input, x's shape must be [...,{self.in_dim}]"
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x) # channel = x.shape[-1] * 2 * self.pe_dim // 2
        for i in range(self.depth):
            if i != self.modulated_layer_id:
                para = self.global_params[f'wb{i}']
            else:
                para = self.params[f'wb{i}']
            x = batched_linear_mm(x, para)
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + self.out_bias
        x = x.view(B, *query_shape, -1)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        return x

    def count_parameters(self,g=1):
        mi = self.modulated_layer_id
        shape = self.param_shapes[f'wb{mi}']
        s0,s1 = shape
        s1 = min(s1,g)
        assert s1 % g ==0
        return s0*s1

    def para_info(self, **kw_args):
        count = 0
        for i in range(self.depth):
            if i != self.modulated_layer_id:
                shape = self.global_param_shapes[f'wb{i}']
            else:
                shape = self.param_shapes[f'wb{i}']
            count += shape[0] * shape[1]
        return count