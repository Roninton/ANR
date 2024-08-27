import torch
from torch import nn
import einops
import torch.nn.functional as F
import math

from models import register, make_model

from models.HelperModels import LearnedSinusoidalPosEmb,SinusoidalPosEmb
from models.HelperModels import Attention,FeedForward,PreNorm
from utils import make_coord_grid, batched_volume_rendering, init_w, init_wb

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

class Permute(nn.Module):
    def __init__(self,permute_query) -> None:
        super().__init__()
        self.permute_query = permute_query
    
    def forward(self,x):
        return x.permute(*self.permute_query)

@register('TInrEncoder')
class TInrEncoder(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, 
                 shortcut_every=2, # work when > 0
                 dropout=0.,):
        super().__init__()
        self.layers = nn.ModuleList()
        self.shortcut_every = shortcut_every
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))
            
    @staticmethod
    def update_shortcut(x, last_x, sc_x, sc_count, sce):
        """
        return x = x + last_x if not skip\n
        else return x = x + sc_x and update sc_x
        """
        if sce <= 0:
            return x+last_x,sc_x,sc_count
        else:
            sc_count += 1
            if sc_count % sce == 0:
                x = x + sc_x
                sc_x = x
                sc_count = 0
            else:
                x = x + last_x
        return x,sc_x,sc_count
        
    def forward(self, x):
        sce = self.shortcut_every
        sc_x, sc_count = None, 0
        if sce > 0:
            sc_x = x
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            last_x = x
            x = norm_ff(x)
            x, sc_x, sc_count = self.update_shortcut(x,last_x,sc_x,sc_count,sce)
        return x

@register('TInrDecoder')
class TInrDecoder(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim, 
        dropout=0.):
        super().__init__()
        self.hypobuilder = nn.ModuleList()
        for _ in range(depth):
            self.hypobuilder.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x, to):
        for i, sublayer in enumerate(self.hypobuilder):
            norm_self_attn, norm_cross_attn, norm_ff = sublayer
            x = x + norm_self_attn(x,to=None)
            x = x + norm_cross_attn(x,to=to)
            x = x + norm_ff(x)
        return x

@register('HyperNetwork') # use full transformer as inr builder
class HyperNetwork(nn.Module):
    def __init__(self, device, tokenizer, hyponet, n_groups, 
                 tinr_enc,tinr_dec,
                 grouping_mode= "mlp",# "mlp", "mlp_mod"
                 direct_wtoken=False,
                 # for time as a token
                 use_timetoken=False,
                 learned_sinusoidal_cond=False,
                 learned_sinusoidal_dim=16):
        super().__init__()
        
        self.device = device
        dim = tokenizer['args']['dim']

        self.hyponet = make_model(hyponet["name"])(**hyponet["args"])
        self.tokenizer = make_model(tokenizer["name"])(**tokenizer["args"])
        
        # enc-dec
        self.tinr_enc = make_model(tinr_enc["name"])(**(tinr_enc["args"]))
        self.tinr_dec = make_model(tinr_dec["name"])(**(tinr_dec["args"]))
        
        # time module
        self.use_timetoken = use_timetoken
        if use_timetoken:
            if learned_sinusoidal_cond:
                sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )

        # hypo para
        self.direct_wtoken = direct_wtoken
        self.grouping_mode = grouping_mode
        if not direct_wtoken:
            self.base_params = nn.ParameterDict() # parameters
            n_wtokens = 0 # numbers of weight tokens
            self.wtoken_postfc = nn.ModuleDict() # post function
            self.wtoken_rng = dict() # range
            for name, shape in self.hyponet.param_shapes.items():
                if grouping_mode == "mlp":
                    self.base_params[name] = nn.Parameter(init_wb(shape))# .contiguous()
                    g = min(n_groups, shape[1])
                    # print(n_groups,shape)
                    assert shape[1] % g == 0
                    self.wtoken_postfc[name] = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, shape[0] - 1),
                    )
                elif grouping_mode == "mlp_mod":
                    g = min(n_groups, shape[1])
                    self.base_params[name] = nn.Parameter(init_w((g+1,shape[1])))# .contiguous()
                    assert shape[1] % g == 0
                    self.wtoken_postfc[name] = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, shape[0] - 1),
                    )
                else:
                    raise NotImplementedError()
                self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
                n_wtokens += g
            self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        else:
            self.base_params = nn.ParameterDict() # parameters
            n_wtokens = 0 # numbers of weight tokens
            self.wtoken_postfc = nn.ModuleDict() # post function
            self.wtoken_rng = dict() # range
            for name, shape in self.hyponet.param_shapes.items():
                t_len, t_dim = shape
                self.wtoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, t_dim),
                )
                self.base_params[name+"_w"] = nn.Parameter(torch.ones(shape))
                self.base_params[name+"_b"] = nn.Parameter(torch.randn(shape)) # is positional emb
                self.wtoken_rng[name] = (n_wtokens, n_wtokens + t_len)
                n_wtokens += t_len
            self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        print(f"n_wtokens: {n_wtokens}")
        # exit(0)
        
    def get_encoded(self, data, time=None):
        # ttoken and dtoken building
        if self.use_timetoken and time is not None:
            ttokens = self.time_mlp(time)[:,None,:]
            dtokens = self.tokenizer(data)
            dt_tokens = torch.cat([dtokens, ttokens], dim=1)
        else:
            dt_tokens = self.tokenizer(data)
        # forward attn
        encoded = self.tinr_enc(dt_tokens)
        return encoded

    def get_wtoken(self,encoded):
        # build wtoken
        B = encoded.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        # get wtoken with decoder
        wtoken = self.tinr_dec(wtokens,to=encoded)
        return wtoken

    def forward(self, data, time=None):
        encoded = self.get_encoded(data, time)
        wtoken = self.get_wtoken(encoded)
        hypo_param = self.get_hypo_by_wtoken(wtoken)
        return hypo_param
    
    def forward_with_encoded(self,encoded):
        wtoken = self.get_wtoken(encoded)
        hypo_param = self.get_hypo_by_wtoken(wtoken)
        return hypo_param
    
    def get_hypo_by_wtoken(self, wtoken):
        # print("trans_out",trans_out.shape)
        B = wtoken.shape[0]
        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            
            if not self.direct_wtoken:
                # transformer output
                l, r = self.wtoken_rng[name]
                trans_out_i = wtoken[:, l: r, :]
                if self.grouping_mode == 'mlp':
                    ## fc ##
                    # base parameters
                    wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B) # (shape[0], shape[1])
                    w, b = wb[:, :-1, :], wb[:, -1:, :] # w @ (shape[0]-1, shape[1])
                    # bind two part
                    x = self.wtoken_postfc[name](trans_out_i) # (B, g, shape[0] - 1)
                    x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
                    w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

                    # build weight-bias matrix
                    wb = torch.cat([w, b], dim=1)
                    params[name] = wb
                elif self.grouping_mode == 'mlp_mod':
                    ## fc ##
                    # base parameters
                    wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B) # (g+1, shape[1])
                    w, b = wb[:, :-1, :], wb[:, -1:, :] # w @ (g, shape[1])
                    # bind two part
                    x = self.wtoken_postfc[name](trans_out_i) # (B, g, shape[0] - 1)
                    x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
                    w = torch.matmul(x,w)
                    # build weight-bias matrix
                    wb = torch.cat([w, b], dim=1)
                    params[name] = wb
            else:
                l, r = self.wtoken_rng[name]
                trans_out_i = wtoken[:, l: r, :]
                w = einops.repeat(self.base_params[name+"_w"], 'n m -> b n m', b=B) # (shape[0], shape[1])
                b = einops.repeat(self.base_params[name+"_b"], 'n m -> b n m', b=B) # (shape[0], shape[1])
                x = self.wtoken_postfc[name](trans_out_i) # (B, g, shape[0])
                # x = F.normalize(w * x, dim=-1) + b
                x = w * x + b
                params[name] = x
                
        return params

    def get_image(self, hypo_wb, image_shape):
        hypo_model = self.hyponet
        hypo_model.set_params(hypo_wb)
        coord = make_coord_grid((image_shape[1], image_shape[2]), (-1, 1)).cuda()
        coord = coord.unsqueeze(0)
        coord = coord.repeat(image_shape[0], 1, 1, 1)

        return hypo_model(coord)

    def get_pixels_from_NeRF(self,hypo_wb,rays_o,rays_d,near,far,points_per_ray=128,mini_batch_size=32, is_train=True):
        hypo_model = self.hyponet
        hypo_model.set_params(hypo_wb)
        use_viewdirs = hypo_model.use_viewdirs if hasattr(hypo_model,"use_viewdirs") else False
        model_output = batched_volume_rendering(
            hypo_model, rays_o, rays_d,
            near=near,
            far=far,
            points_per_ray=points_per_ray,
            use_viewdirs=use_viewdirs,
            rand=is_train,

            batch_size=mini_batch_size,
        )

        return model_output