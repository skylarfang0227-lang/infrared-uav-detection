"""MSDA-IRSTD independent reconstruction, fixed stage-1 specification.

Paper equations 1-12; unreported settings are disclosed in the specification.
Ultralytics components retain their upstream AGPL-3.0 license.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules import Conv, C3k2, C2PSA, SPPF, Concat, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import initialize_weights

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/models/msda_irstd_spec.json"


class DropPath(nn.Module):
    def __init__(self, probability=.1):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
        return x * mask / keep


class ASFEBlock(nn.Module):
    def __init__(self, channels, options):
        super().__init__()
        k1, k2 = options["dw_kernels"]
        hidden = channels * options["f1_f2_expansion"]
        self.dw1 = nn.Conv2d(channels, channels, k1, padding=k1//2, groups=channels)
        self.f1 = nn.Conv2d(channels, hidden, 1)
        self.f2 = nn.Conv2d(channels, hidden, 1)
        self.g = nn.Conv2d(hidden, channels, 1)
        self.dw2 = nn.Conv2d(channels, channels, k2, padding=k2//2, groups=channels)
        self.drop = DropPath(options["drop_path_probability"])

    def forward(self, z):
        zs = self.dw1(z)
        return z + self.drop(self.dw2(self.g(F.relu6(self.f1(zs)) * self.f2(zs))))


class ASFE(nn.Module):
    def __init__(self, c1, c2, n, options):
        super().__init__()
        h = c2//2
        self.project = Conv(c1, c2, 1)
        self.eds = nn.Sequential(Conv(c2, c2, 3, g=c2), Conv(c2, h, 1))
        self.channel = Conv(c2, h, 1)
        self.blocks = nn.ModuleList(ASFEBlock(h, options["block"]) for _ in range(n))
        self.fuse = Conv((4+n)*h, c2, 1)

    def forward(self, x):
        y0 = self.project(x)
        y3, y4 = y0.chunk(2, dim=1)
        ys = [self.eds(y0), self.channel(y0), y3, y4]
        for block in self.blocks:
            ys.append(block(ys[-1]))
        return self.fuse(torch.cat(ys, dim=1))


class DynamicTanh(nn.Module):
    def __init__(self, channels, alpha=.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.tanh(self.alpha*x)*self.weight[None, :, None, None] + self.bias[None, :, None, None]


class DynamicPSA(nn.Module):
    def __init__(self, original, channels, alpha):
        super().__init__()
        self.attn, self.ffn, self.add = original.attn, original.ffn, original.add
        self.dyt_attn = DynamicTanh(channels, alpha)
        self.dyt_ffn = DynamicTanh(channels, alpha)

    def forward(self, x):
        a = self.attn(self.dyt_attn(x))
        x = x+a if self.add else a
        f = self.ffn(self.dyt_ffn(x))
        return x+f if self.add else f


class C2DYT(C2PSA):
    def __init__(self, c1, c2, n, e, alpha):
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.Sequential(*(DynamicPSA(block, self.c, alpha) for block in self.m))


class MFA(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def forward(self, xs):
        high, middle, low = xs
        size = middle.shape[-2:]
        if self.enabled:
            # Fixed pyramid inputs have integral spatial ratios. Disjoint fixed
            # windows equal adaptive pooling here and have deterministic CUDA backward.
            ratio = tuple(a//b for a, b in zip(high.shape[-2:], size))
            if tuple(a*b for a, b in zip(ratio, size)) != tuple(high.shape[-2:]):
                raise ValueError("Frozen MFA expects integral pyramid ratios")
            high = F.max_pool2d(high, ratio, ratio) + F.avg_pool2d(high, ratio, ratio)
        else:
            high = F.interpolate(high, size=size, mode="nearest")
        return torch.cat((high, middle, F.interpolate(low, size=size, mode="nearest")), dim=1)


class ScaleMaxPool(nn.Module):
    def forward(self, volume):
        # Equivalent to non-overlapping MaxPool3d((3,1,1)); first-index tie rule.
        # Avoids the nondeterministic generic CUDA MaxPool3d backward implementation.
        return volume.max(dim=2, keepdim=True).values


class DSA(nn.Module):
    def __init__(self, channels, cout, options, enabled=True):
        super().__init__()
        self.enabled, self.cout = enabled, cout
        if enabled:
            self.projections = nn.ModuleList(Conv(c, cout, 1) for c in channels)
            self.conv3d = nn.Conv3d(cout, cout, 1, bias=False)
            self.bn = nn.BatchNorm3d(cout, eps=options["bn3d_eps"], momentum=options["bn3d_momentum"])
            self.act = nn.LeakyReLU(options["leaky_relu_slope"])
            self.pool = ScaleMaxPool()

    def forward(self, xs):
        size = xs[0].shape[-2:]
        if not self.enabled:
            return xs[0].new_zeros((xs[0].shape[0], self.cout, *size))
        ys = [F.interpolate(proj(x), size=size, mode="nearest") for proj, x in zip(self.projections, xs)]
        volume = torch.stack(ys, dim=2)
        return self.pool(self.act(self.bn(self.conv3d(volume)))).squeeze(2)


class Add(nn.Module):
    def forward(self, xs):
        return xs[0] + xs[1]


class MSDAIRSTD(DetectionModel):
    def __init__(self, spec=None, enabled=None):
        nn.Module.__init__(self)
        self.spec = copy.deepcopy(spec) if isinstance(spec, dict) else json.loads(Path(spec or SPEC).read_text(encoding="utf-8"))
        self.enabled = {"ASFE": True, "C2DYT": True, "MFA": True, "DSA": True}
        if enabled:
            if set(enabled) - set(self.enabled):
                raise ValueError("Unknown ablation switch")
            self.enabled.update(enabled)
        self.yaml = {"nc": 2, "channels": 3, "backbone": [], "head": [], "msda_spec": self.spec["spec_id"]}
        self.names, self.nc = {0: "UAV", 1: "Bird"}, 2
        self.args = copy.deepcopy(DEFAULT_CFG)
        self.inplace, self.end2end = True, False
        self.task = "detect"
        layers, channels, indices = [], {"image": 3}, {"image": -1}
        options = self.spec["modules"]
        for i, node in enumerate(self.spec["nodes"]):
            inputs, c2, op = node["inputs"], node["channels"], node["op"]
            cs = [channels[k] for k in inputs]
            c1 = cs[0]
            if op == "Conv":
                m = Conv(c1, c2, node["k"], node["s"])
            elif op == "C3k2":
                m = C3k2(c1, c2, n=node["n"], c3k=node["c3k"], e=node["e"])
            elif op == "ASFE":
                m = ASFE(c1, c2, node["n"], options[op]) if self.enabled[op] else C3k2(c1, c2, n=node["n"], c3k=True)
            elif op == "C2DYT":
                m = C2DYT(c1, c2, node["n"], node["e"], options[op]["alpha_initial"]) if self.enabled[op] else C2PSA(c1, c2, node["n"], node["e"])
            elif op == "SPPF":
                m = SPPF(c1, c2, node["k"])
            elif op == "MFA":
                m = MFA(self.enabled[op])
            elif op == "DSA":
                m = DSA(cs, c2, options[op], self.enabled[op])
            elif op == "Concat":
                m = Concat(1)
            elif op == "Add":
                m = Add()
            elif op == "Upsample":
                m = nn.Upsample(scale_factor=node["factor"], mode=node["mode"])
            else:
                raise ValueError(op)
            src = [indices[k] for k in inputs]
            m.i, m.f, m.type, m.node_id = i, src[0] if len(src) == 1 else src, op, node["id"]
            m.np = sum(p.numel() for p in m.parameters())
            layers.append(m)
            channels[node["id"]], indices[node["id"]] = c2, i
        # Keep the upstream global class flag unchanged for other models.
        legacy = Detect.legacy
        try:
            Detect.legacy = False
            head = Detect(nc=2, ch=self.spec["detect"]["channels"])
        finally:
            Detect.legacy = legacy
        head.legacy = False
        head.i, head.f = len(layers), [indices[k] for k in self.spec["detect"]["inputs"]]
        head.type, head.node_id = "Detect", "Detect"
        head.np = sum(p.numel() for p in head.parameters())
        head.stride = torch.tensor(self.spec["detect"]["strides"], dtype=torch.float32)
        layers.append(head)
        self.model = nn.Sequential(*layers)
        self.save = sorted({j for m in layers for j in ([m.f] if isinstance(m.f, int) else m.f) if j >= 0})
        self.stride = head.stride
        head.bias_init()
        initialize_weights(self)

    def _predict_augment(self, x):
        raise RuntimeError("TTA is disabled by the fixed protocol")
