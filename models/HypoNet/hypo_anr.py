import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import einops
import sys
sys.path.extend(["../../","../"])
from models import register
from models.HelperModels import Sine

### Hypo Net of Attention-based Localized Implicit Representation ###

class Mish(nn.Module):
    """
    Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    implemented for PyTorch / FastAI by lessw2020 
    github: https://github.com/lessw2020/mish
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class SignalAttn(nn.Module):
    def __init__(self, dim, n_head, head_dim, att_threshold=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        inner_dim = n_head * head_dim
        self.inner_dim = inner_dim
        self.dim = dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = head_dim ** -0.5
        self.att_threshold = att_threshold

    def forward(self, fr, dt):
        q = self.to_q(fr)
        k = self.to_k(dt)
        v = self.to_v(dt)
        b, n, h_d = q.shape
        dn = k.shape[1]
        # print(q.shape,k.shape)
        q = q.reshape(b, n, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # b n (h d) -> b h n d
        k = k.reshape(b, dn, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # b dn (h d) -> b h dn d
        v = v.reshape(b, dn, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # b dn (h d) -> b h dn d
        # print(q.shape,k.shape)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # almost vary between -1.5~1.5
        attn = F.softmax(attn, dim=-1)  # b h n dn # almost vary between 0.003~0.065
        if self.att_threshold is not None:
            # print(attn.max(),attn.min(),attn.sum(dim=-1).mean())
            attn = attn.clip(self.att_threshold) - self.att_threshold
            attn = attn / torch.sum(attn, dim=-1, keepdim=True)
            # print(attn.max(),attn.min(),attn.sum(dim=-1).mean())
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, h_d)  # b h n d -> b n (h d)
        return self.to_out(out)

    def count_parameters(self):
        return 4 * self.inner_dim * self.dim + self.dim


class SignalMLP(nn.Module):
    def __init__(self, dim, ff_dim,
                 activation="mish",
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        if activation == "mish":
            act_func = Mish
        elif activation == "relu":
            act_func = nn.ReLU
        else:
            act_func = Mish

        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            act_func(),
            nn.Linear(ff_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@register('ANR')
class ANR(nn.Module):
    def __init__(self, depth, in_dim, out_dim,
                 hidden_dim,
                 token_len,
                 n_head=6,
                 head_dim=64,
                 use_pe=True, pe_dim=128, pe_sigma=1024,  # pe_related
                 out_scale=1,
                 out_bias=0,
                 # tunning para
                 adaptive_idim=False,
                 att_threshold=None,
                 activation="relu",  # mish, relu
                 rescale=False,
                 ):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        if use_pe:
            pos_dim = in_dim * pe_dim
        else:
            pos_dim = in_dim

        if pos_dim != hidden_dim:
            assert adaptive_idim, f"pos_dim:({pos_dim})!=hidden_dim:({hidden_dim}) while init_linear is False."
            print(f"[I] pos_dim:({pos_dim})!=hidden_dim:({hidden_dim}), use adaptive_idim.")
            self.pe_dim = hidden_dim // in_dim
            pos_dim = in_dim * self.pe_dim
            self.res_dim = hidden_dim - pos_dim
        else:
            self.res_dim = 0

        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        # build attn layer
        self.attn = SignalAttn(hidden_dim, n_head, head_dim, att_threshold=att_threshold)
        # global mlps
        if activation == "mish":
            act_func = Mish
        elif activation == "relu":
            act_func = nn.ReLU
        else:
            act_func = nn.ReLU
        mlp = []
        for _ in range(self.depth - 1):
            mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act_func()
            ))
        mlp.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*mlp)

        self.out_dim = out_dim
        self.out_bias = out_bias
        self.out_scale = out_scale
        # params shapes
        self.param_shapes = dict()
        # self.param_shapes["dtoken"] = (token_len, hidden_dim)
        self.param_shapes["dtoken"] = (token_len, hidden_dim)
        self.params = None
        self.rescale = rescale

    def init_randn_token(self):
        params = dict()
        for key, val in self.param_shapes.items():
            print(key, val)
            params[key] = torch.randn(val)
        return params

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)  # xs||ys at last dim
        # query_shape = x.shape[:-1]
        # x = x.reshape(*query_shape,2,-1).transpose(-1, -2).reshape(*query_shape,-1) #  -> xys
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)  # cos(xys)||sin(xys) at last dim
        # query_shape = x.shape[:-1]
        # x = x.reshape(*query_shape,2,-1).transpose(-1, -2).reshape(*query_shape,-1) #  -> cos and sin (xys)
        return x

    @staticmethod
    def _convert_posenc(x, pe_sigma=1024, pe_dim=128):
        w = torch.exp(torch.linspace(0, np.log(pe_sigma), pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)  # xs||ys at last dim
        # query_shape = x.shape[:-1]
        # x = x.reshape(*query_shape,2,-1).transpose(-1, -2).reshape(*query_shape,-1) #  -> xys
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)  # cos(xys)||sin(xys) at last dim
        # query_shape = x.shape[:-1]
        # x = x.reshape(*query_shape,2,-1).transpose(-1, -2).reshape(*query_shape,-1) #  -> cos and sin (xys)
        return x

    def count_parameters(self):
        shape = self.param_shapes[f'dtoken']
        count = shape[0] * shape[1]
        return count

    def forward(self, x, **kwargs):
        B, query_shape = x.shape[0], x.shape[1: -1]
        # print(x.shape)
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)

        if self.res_dim > 0:
            zeros = torch.zeros([B, x.shape[1], self.res_dim], device=x.device)
            x = torch.cat([x, zeros], dim=-1)

        # declaration
        dt = self.params["dtoken"]
        x = self.attn(x, dt)
        x = self.mlp(x)
        x = x.view(B, *query_shape, -1)
        if self.out_scale != 1:
            x = self.out_scale * x
        x = x + self.out_bias

        if self.rescale:
            x = (x * 2) - 1
        return x

    def para_info(self, **kw_args):
        repr_para = self.count_parameters()
        all_para = (self.attn.count_parameters() +  # attn_para
                    (self.depth - 1) * (self.hidden_dim * self.hidden_dim + self.hidden_dim) +  # mlp_para
                    self.hidden_dim * self.out_dim + self.out_dim +
                    repr_para
                    )
        return all_para
