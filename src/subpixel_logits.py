"""ESPCN-style logit shuffling adapted to the first YOLO11 DFL head.

Sources: Shi et al., CVPR2016 (subpixel projection); Li et al., Sensors2021,
doi:10.3390/s21165656 (joint upsampling/detection precedent). This is an
independent DFL/TAL adaptation, not either full algorithm or an originality claim.
"""
import copy
import math
import torch
from torch import nn


class SubpixelLogits(nn.Module):
    def __init__(self,source,classification=False):
        super().__init__()
        if not isinstance(source,nn.Conv2d) or source.kernel_size!=(1,1) or source.groups!=1 or source.bias is None:
            raise ValueError('Expected pointwise prediction convolution with bias')
        self.projection=copy.deepcopy(source)
        self.projection.out_channels=source.out_channels*4
        self.projection.weight=nn.Parameter(source.weight.detach().repeat_interleave(4,dim=0))
        self.projection.bias=nn.Parameter(source.bias.detach().repeat_interleave(4,dim=0))
        if classification:
            # Same upstream expected-object-density initialization at stride2.
            with torch.no_grad():self.projection.bias.sub_(math.log(4))

    def forward(self,x):return nn.functional.pixel_shuffle(self.projection(x),2)


def install_subpixel_head(model):
    head=model.model[-1]
    if head.end2end or head.stride.tolist()!=[4.,8.,16.,32.]:
        raise ValueError('Expected ordinary complete MSDA four-scale head')
    head.cv2[0][-1]=SubpixelLogits(head.cv2[0][-1])
    head.cv3[0][-1]=SubpixelLogits(head.cv3[0][-1],classification=True)
    head.stride=head.stride.clone();head.stride[0]=2.;model.stride=head.stride
    head.shape=None
    head.np=sum(p.numel() for p in head.parameters())
