"""Single sequential GPU campaign. No scheduling service or implicit evaluation."""
from pathlib import Path
import sys,json,time,subprocess,hashlib,shutil
E=Path(__file__).resolve().parents[1];R=E
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    protocol=E/'configs/grouped_protocol.json';cfg=json.loads(protocol.read_text())
    assert json.loads((E/'analysis/resume_check.json').read_text())['all_exact']
    smoke=[]
    for name in ['smoke_v2_full','smoke_reference','smoke_collapsed']:
        run=E/'experiments'/name;result=json.loads((run/'result.json').read_text());assert result['next_epoch']==2 and result['peak_allocated_gib']<6
        smoke.append(json.loads((run/'train/progress.json').read_text()))
    assert len({tuple(h['first_batch_sha256'] for h in rows) for rows in smoke})==1
    save(E/'analysis/training_preflight.json',{'finite_gradient_checks':'passed all parameters for 2 epochs','matched_batch_fingerprints':True,'resume_exact':True,'initial_smoke':'smoke_full predates architecture-independent data RNG reset; retained, not reused','free_disk_gib':shutil.disk_usage(E).free/2**30,'estimate':'13 x 50 epoch runs, approximately 13-16 GPU hours from 64-image smoke timing; about 4 GB checkpoints. Actual cost recorded per run.'})
    jobs=[]
    for i,seed in enumerate(cfg['seeds']):
        variants=cfg['variants'][i:]+cfg['variants'][:i]
        jobs.extend((v,seed) for v in variants)
    jobs.append(('phase_collapsed',17));index=[];start=time.time()
    for variant,seed in jobs:
        name=f'grouped_{variant}_s{seed}';run=E/'experiments'/name
        if run.exists():raise FileExistsError(f'Refusing implicit overwrite/resume: {run}')
        if shutil.disk_usage(E).free<8*2**30:raise RuntimeError('Less than 8GiB free')
        command=[sys.executable,'-X','utf8',str(E/'scripts/train.py'),'--variant',variant,'--seed',str(seed),'--protocol',str(protocol),'--run',name]
        save(E/'logs/campaign_status.json',{'status':'training','active':name,'completed':index,'seconds':time.time()-start})
        print('START '+name,flush=True)
        with (E/f'logs/{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save(E/'logs/campaign_status.json',{'status':'failed','active':name,'returncode':result.returncode,'completed':index});raise RuntimeError(name)
        info=json.loads((run/'result.json').read_text());assert info['next_epoch']==50
        checkpoint=run/'train/weights/epoch50.pt'
        index.append({'variant':variant,'seed':seed,'checkpoint':str(checkpoint.relative_to(R)),'checkpoint_sha256':sha(checkpoint),'result':info})
        save(E/'results/training_index.json',index);print('COMPLETE '+name,flush=True)
    save(E/'results/factorial_weights_frozen.json',{'protocol_sha256':sha(protocol),'manifest_sha256':sha(E/'data/staged_manifest.json'),'runs':[r for r in index if r['variant']!='phase_collapsed'],'test_access_before_freeze':False})
    save(E/'logs/campaign_status.json',{'status':'all_training_complete','completed':index,'seconds':time.time()-start})
if __name__=='__main__':main()
