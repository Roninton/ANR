import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import sys
sys.path.extend(["../../","../"])
from models import register
from models.HelperModels import Sine
from utils import batched_linear_mm

@register("HypoMlp")
class HypoMlp(nn.Module):
    def __init__(self, 
        depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim,
        activation = "relu", out_bias=0, pe_sigma=1024,
        ):
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
        self.param_shapes = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim

        # grouping op note: post func will proj(dim) -> shape[0] , and grouping on shape[1]
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.activation = activation
        if activation is None:
            self.act = nn.Identity()
        elif activation=="relu":
            self.act = nn.ReLU()
        elif activation == "sine":
            self.act = Sine()
        elif activation=="sigmoid":
            self.act = nn.Sigmoid()
        elif activation=="tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError()
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
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = x + self.out_bias
        x = x.view(B, *query_shape, -1)
        return x

    def __str__(self) -> str:
        _str = f"<HypoMlp> input dim # {self.in_dim}\n"
        _str += f"parameter number: {self.count_parameters()}\n"
        _str += "- - - - - -\n"
        for i in range(self.depth):
            name = f'wb{i}'
            shape = self.param_shapes[name]
            _str_i = f'{name} : {shape}\n'
            _str += _str_i
        _str += '- - - - - -'
        return _str

    def count_parameters(self,g=1):
        count = 0
        for key in self.param_shapes:
            shape = self.param_shapes[key]
            s0,s1 = shape
            s1 = min(s1,g)
            assert s1 % g ==0 or s1 < g
            count += s0*s1
        return count

    def para_info(self):
        count = 0
        for key in self.param_shapes:
            shape = self.param_shapes[key]
            count += shape[0] * shape[1]
        return count
