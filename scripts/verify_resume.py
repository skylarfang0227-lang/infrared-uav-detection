"""Compare two trusted local probe checkpoints without using CUDA or training."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch

E=Path(__file__).resolve().parents[1]


def equal(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--full',type=Path,default=E/'experiments/smoke_v2_full/train/weights/last.pt')
    ap.add_argument('--resumed',type=Path,default=E/'experiments/smoke_v2_resume/train/weights/last.pt')
    ap.add_argument('--output',type=Path,default=E/'analysis/resume_check_recomputed.json')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve the previous check; use a new output name.')
    torch.set_num_threads(1)
    a=torch.load(args.full,map_location='cpu',weights_only=False)
    b=torch.load(args.resumed,map_location='cpu',weights_only=False)
    fields=['model','ema','optimizer','scaler','scheduler','rng','gradients','next_epoch','last_opt_step','ema_updates']
    result={k:equal(a[k],b[k]) for k in fields}
    histories=[[{k:v for k,v in row.items() if k!='seconds'} for row in state['history']] for state in [a,b]]
    result['history']=equal(*histories)
    result['all_exact']=all(result.values())
    result['scope']='Exact restorable training state and loss/input history; elapsed seconds and output-directory metadata are intentionally not compared.'
    result['checkpoint_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('full',args.full),('resumed',args.resumed)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    assert result['all_exact'], 'Resume equivalence failed; do not launch the campaign.'


if __name__=='__main__':main()
