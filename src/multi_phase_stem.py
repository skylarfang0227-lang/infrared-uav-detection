"""MP-YOLO MPM padding geometry, only as a single-stem candidate.

Author source: zhangjun2636808827/MP-YOLO e53657f2, MP.py MPM,
AGPL-3.0. Independently expressed via partitioning the common scratch stem.
Use reference BN settings and exactly its initial parameter values; no weights
from a trained model are required by the training adapter.
"""
import copy
import torch
from torch import nn


class MultiPhaseStem(nn.Module):
    pads=((1,1,1,1),(0,2,1,1),(1,1,0,2),(0,2,0,2))

    def __init__(self,source):
        super().__init__()
        c=source.conv
        if c.kernel_size!=(3,3) or c.stride!=(2,2) or c.padding!=(1,1) or c.groups!=1 or c.bias is not None or c.out_channels%4:
            raise ValueError('Expected divisible-by4 standard3x3 stride2 stem')
        channels=c.out_channels//4
        blocks=[]
        for i in range(4):
            block=copy.deepcopy(source);sl=slice(i*channels,(i+1)*channels)
            block.conv.out_channels=channels;block.conv.padding=(0,0)
            block.conv.weight=nn.Parameter(c.weight.detach()[sl].clone())
            block.bn.num_features=channels
            for name in ('weight','bias'):
                setattr(block.bn,name,nn.Parameter(getattr(source.bn,name).detach()[sl].clone()))
            for name in ('running_mean','running_var'):
                setattr(block.bn,name,getattr(source.bn,name)[sl].clone())
            blocks.append(block)
        self.branches=nn.ModuleList(blocks)

    def forward(self,x):
        return torch.cat([m(nn.functional.pad(x,p)) for m,p in zip(self.branches,self.pads)],dim=1)


def install_mpm_stem(model):
    old=model.model[0]
    if old.node_id!=model.spec['nodes'][0]['id'] or model.spec['nodes'][0]['op']!='Conv':
        raise ValueError('Unexpected stem')
    new=MultiPhaseStem(old)
    for key in ('i','f','node_id'):setattr(new,key,getattr(old,key))
    new.type='MultiPhaseStem';new.np=sum(p.numel() for p in new.parameters())
    model.model[0]=new
