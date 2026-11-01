"""CPU algebra check of what the phase-collapse control actually constrains."""
from pathlib import Path
import os, sys, json
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics')
sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
import torch
from detector_variants import PhaseCollapsedLogits


def main():
    torch.set_num_threads(1);torch.manual_seed(20261029)
    source=torch.nn.Conv2d(3,2,1,bias=True).double()
    collapsed=PhaseCollapsedLogits(source)
    x=torch.randn(2,3,4,5,dtype=torch.float64)
    target=torch.randn(2,2,8,10,dtype=torch.float64)
    direct=source(x).repeat_interleave(2,-2).repeat_interleave(2,-1)
    output=collapsed(x)
    output_error=float((direct-output).abs().max().detach())
    ((direct-target)**2).mean().backward()
    ((output-target)**2).mean().backward()
    grad=collapsed.projection.weight.grad.reshape(2,4,3,1,1)
    gradient_error=float((grad-source.weight.grad[:,None]/4).abs().max())
    assert output_error<1e-12 and gradient_error<1e-12
    result={'output_error':output_error,'phase_gradient_vs_quarter_direct_error':gradient_error,
            'interpretation':'Averaging four trainable phase projections equals a lower-capacity pointwise projection followed by nearest-neighbour replication. Each phase receives one quarter of the effective projection gradient. With the same SGD learning rate and decay, this is not an optimization-matched conventional upsampling control.',
            'scope':'CPU operator algebra check; no detection score, trained model, or causal AP attribution.'}
    (E/'analysis/phase_control_algebra.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
