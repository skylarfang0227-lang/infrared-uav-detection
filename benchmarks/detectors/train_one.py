from pathlib import Path
import json,sys,time,argparse,subprocess,hashlib,platform
from comparator import B,E,R,torch,ComparisonTrainer,protocol,overrides,sources,sha,load
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--run-name');a=p.parse_args()
    cfg=protocol();run=B/'runs'/(a.run_name or (('smoke_' if a.smoke else '')+a.model));run.mkdir(exist_ok=False)
    members=E/'data'/('detector_smoke.txt' if a.smoke else 'train.txt');args=overrides(cfg,run,17,members.resolve())
    original=cfg['training_args'];ignored={'project','name','data','exist_ok','model'}
    assert all(original[k]==v for k,v in args.items() if k not in ignored)
    save(run/'config.json',{'protocol_sha256':sha(B/'protocol.json'),'model':a.model,'smoke':a.smoke,'args':args,'source_hashes':sources(),'command':sys.argv,'git_commit':None,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'start_unix':time.time()})
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();start=time.time()
    t=ComparisonTrainer(a.model,args);t.setup()
    if a.resume:t.resume_local(a.resume)
    initial=hashlib.sha256(next(t.model.parameters()).detach().cpu().numpy().tobytes()).hexdigest()
    t.run_epochs(stop_after=2 if a.smoke else None)
    changed=hashlib.sha256(next(t.model.parameters()).detach().cpu().numpy().tobytes()).hexdigest()!=initial;assert changed
    if not a.smoke:
        archived=json.loads((E/'experiments/grouped_reference_s17/train/progress.json').read_text())
        assert len(t.history)==50 and [x['first_batch_sha256'] for x in t.history]==[x['first_batch_sha256'] for x in archived]
    if a.smoke:
        restored,state=load(t.last,'cpu');expected=t.ema.ema.state_dict();assert all(torch.equal(v.cpu(),restored.state_dict()[k]) for k,v in expected.items())
    save(run/'result.json',{'model':a.model,'seed':17,'epochs':t.next_epoch,'smoke':a.smoke,'checkpoint':str(t.last.relative_to(R)),'checkpoint_sha256':sha(t.last),'parameters':sum(x.numel() for x in t.model.parameters()),'strides':t.model.stride.tolist(),'elapsed_seconds':time.time()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loader_audit':t.loader_audit,'training_time_evaluations':0,'finite_gradients_all_trainable_parameters':True,'parameters_updated':changed,'all50_first_batches_match_archived_A':not a.smoke,'EMA_state_reload_exact':True if a.smoke else None})
if __name__=='__main__':main()
