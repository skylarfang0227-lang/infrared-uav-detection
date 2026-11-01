"""Train one preregistered extension run; never constructs evaluation data."""
from pathlib import Path
import os,sys,json,argparse,time,subprocess
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
import torch
from experiment_trainer import ExperimentTrainer,overrides,source_hashes,sha
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--members',type=Path,default=E/'data/train.txt');p.add_argument('--stop-after',type=int);p.add_argument('--resume',type=Path);a=p.parse_args()
    run=E/'experiments'/a.run;run.mkdir(exist_ok=False)
    protocol=json.loads(a.protocol.read_text());args=overrides(protocol,run,a.seed,a.members.resolve())
    save(run/'config.json',{'protocol':protocol,'protocol_sha256':sha(a.protocol),'variant':a.variant,'seed':a.seed,'args':args,'sources':source_hashes(),'command':sys.argv,'git_commit':None})
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();started=time.time()
    trainer=ExperimentTrainer(a.variant,a.protocol,args);trainer.setup()
    if a.resume:trainer.resume_local(a.resume)
    trainer.run_epochs(stop_after=a.stop_after)
    save(run/'result.json',{'variant':a.variant,'seed':a.seed,'next_epoch':trainer.next_epoch,'checkpoint':str(trainer.last.relative_to(R)),'checkpoint_sha256':sha(trainer.last),'elapsed_seconds':time.time()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'parameters':sum(p.numel() for p in trainer.model.parameters()),'loader_audit':trainer.loader_audit,'training_time_evaluations':0})
if __name__=='__main__':main()
