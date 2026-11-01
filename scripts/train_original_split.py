"""Train the original-split reference or combined model used for frozen transfer."""
from pathlib import Path
import argparse, json, os, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'logs/ultralytics')
from src.msda_training import MSDATrainer,protocol_overrides,sha
from src.msda_subpixel_mpm import SubpixelMPMTrainer
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['reference','candidate'],required=True);p.add_argument('--seed',type=int,choices=[17,29,43],required=True);p.add_argument('--resume',type=Path);p.add_argument('--run-name');a=p.parse_args()
    run=ROOT/'experiments/original_split'/(a.run_name or f'{a.method}_s{a.seed}');run.mkdir(parents=True,exist_ok=False)
    args=protocol_overrides('formal',run,seed=a.seed,batch=8,workers=4)
    candidate=json.loads((ROOT/'configs/candidates/msda_subpixel_mpm_v1.json').read_text())
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    trainer=MSDATrainer('formal',args) if a.method=='reference' else SubpixelMPMTrainer('formal',args,candidate)
    trainer.setup()
    if a.resume:trainer.resume_local(a.resume)
    assert len(trainer.train_loader.dataset)==2400
    (run/'config.json').write_text(json.dumps(args,indent=2))
    trainer.run_epochs(stop_after=50)
    (run/'result.json').write_text(json.dumps({'method':a.method,'seed':a.seed,'next_epoch':trainer.next_epoch,'checkpoint':trainer.last.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(trainer.last),'elapsed_seconds':time.monotonic()-start,'loader_audit':trainer.loader_audit,'training_time_evaluations':0},indent=2))
if __name__=='__main__':main()
