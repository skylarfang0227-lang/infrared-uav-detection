"""Frozen MSDA plus factorial switches and a same-count phase-collapse control.

The historical implementation is imported read-only. No baseline core is disabled.
"""
from pathlib import Path
import sys
E=Path(__file__).resolve().parents[1];R=E
sys.path.insert(0,str(R))
import torch
from torch import nn
from src.models.msda_irstd import MSDAIRSTD
from src.subpixel_logits import SubpixelLogits,install_subpixel_head
from src.multi_phase_stem import install_mpm_stem

VARIANTS=('reference','head','stem','combined','phase_collapsed')
class PhaseCollapsedLogits(SubpixelLogits):
    """Keep all expanded parameters, but discard their spatial phase distinction.

    Matches output count, DFL support, decoding stride, initialization and raw
    parameter count of the head. Redundant phase parameters remain trainable,
    but the average ties their output values. This is a mechanistic ablation,
    not a claim of equal effective function capacity or an optimized competitor.
    """
    def forward(self,x):
        z=self.projection(x);b,c,h,w=z.shape
        mean=z.reshape(b,c//4,4,h,w).mean(2)
        return mean.repeat_interleave(2,-2).repeat_interleave(2,-1)

class DetectorModel(MSDAIRSTD):
    def __init__(self,variant,spec=None,enabled=None):
        if variant not in VARIANTS:raise ValueError(variant)
        super().__init__(spec,enabled)
        if not all(self.enabled.values()):raise ValueError('Complete reference core required')
        self.extension_variant=variant
        if variant in ('head','combined'):
            install_subpixel_head(self)
        elif variant=='phase_collapsed':
            head=self.model[-1]
            head.cv2[0][-1]=PhaseCollapsedLogits(head.cv2[0][-1])
            head.cv3[0][-1]=PhaseCollapsedLogits(head.cv3[0][-1],classification=True)
            head.stride=head.stride.clone();head.stride[0]=2.;self.stride=head.stride;head.shape=None
            head.np=sum(p.numel() for p in head.parameters())
        if variant in ('stem','combined'):install_mpm_stem(self)

def load_detector(path,device='cpu',ema=True):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=DetectorModel(state['extension_variant'],state['spec'],state['enabled'])
    model.load_state_dict(state['ema' if ema else 'model'],strict=True)
    return model.to(device).float().eval(),state
