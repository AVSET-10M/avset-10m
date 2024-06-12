import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import copy

# from deepspeed.moe.sharded_moe import gumbel_rsample
# from transformers import CLIPModel

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)
def get_clones(module: nn.Module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    def __init__(
        self, channel=512, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(channel, int(channel * res_expansion), bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(channel * res_expansion), channel, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return self.net2(self.net1(x))
    
class MLP_dim(nn.Module):
    def __init__(
        self, in_dim=512, out_dim=1024, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, int(out_dim), bias=bias),
            nn.BatchNorm1d(int(out_dim)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(out_dim), out_dim, bias=bias),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.net2(self.net1(x))

class Proj_Head(torch.nn.Module):
    def __init__(self, init_mode):
        super(Proj_Head, self).__init__()
 
        self.mlp1 = MLP(res_expansion=2)
        self.mlp2 = MLP(res_expansion=2)

        self.init_weights(init_mode)

    def forward(self, embs):
        embs = self.mlp1(embs)
        embs = self.mlp2(embs)
        return F.normalize(embs, dim=-1)

    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
    
    def get_device(self):
        return next(self.parameters()).device

class Proj_Head_dim(torch.nn.Module):
    def __init__(self, in_dim, out_dim, init_mode, dim_act='relu'):
        super(Proj_Head_dim, self).__init__()
 
        self.mlp1 = MLP_dim(in_dim=in_dim, out_dim=out_dim, activation=dim_act)
        self.mlp2 = MLP(channel=out_dim, res_expansion=2, activation=dim_act)

        self.init_weights(init_mode)

    def forward(self, embs):
        embs = self.mlp1(embs)
        embs = self.mlp2(embs)
        return F.normalize(embs, dim=-1)

    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
    
    def get_device(self):
        return next(self.parameters()).device

class Proj_Head_mini(torch.nn.Module):
    def __init__(self, in_dim, out_dim, init_mode, dim_act='relu'):
        super(Proj_Head_mini, self).__init__()
 
        self.mlps = nn.Sequential(
            nn.Linear(in_dim, 2048, bias=True),
            nn.BatchNorm1d(2048),
            get_activation(dim_act),
            nn.Linear(2048, out_dim, bias=True),
            nn.BatchNorm1d(out_dim),
        )

        self.init_weights(init_mode)

    def forward(self, embs):
        embs = self.mlps(embs)
        return F.normalize(embs, dim=-1)

    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
    
    def get_device(self):
        return next(self.parameters()).device

class ib_with_proj(torch.nn.Module):
    def __init__(self, ib):
        super(ib_with_proj, self).__init__()
        self.ib       = ib
        # self.ib_head  = nn.Linear(1024, 768, bias=True)
        self.ib_head  = Proj_Head_mini(1024, 768, 'xav', 'relu')
        for m in self.ib_head.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

class ib_with_proj_1024(torch.nn.Module):
    def __init__(self, ib):
        super(ib_with_proj_1024, self).__init__()
        self.ib       = ib
        # self.ib_head  = nn.Linear(1024, 768, bias=True)
        self.ib_head  = Proj_Head_mini(1024, 1024, 'xav', 'relu')
        for m in self.ib_head.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        
