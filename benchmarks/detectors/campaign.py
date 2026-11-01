"""One sequential GPU campaign: preflight all models, train all, freeze, evaluate."""
from pathlib import Path
import sys,json,time,subprocess,gzip,gc
from comparator import B,E,R,protocol,torch,sha,load,sources
from scripts.evaluate_sirst_protocol import evaluate_model,score
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
    cfg=protocol();began=time.time();completed=[]
    assert sha(E/'data/train.txt')==cfg['train_members_sha256'] and sha(E/'data/staged_manifest.json')==cfg['manifest_sha256']
    save(B/'audit/frozen_implementation.json',{'protocol_sha256':sha(B/'protocol.json'),'source_hashes':sources(),'time_unix':began})
    for smoke in [True,False]:
        for m in cfg['models']:
            name=('smoke_' if smoke else '')+m['id'];run=B/'runs'/name
            if run.exists():raise FileExistsError('Refuse implicit overwrite: '+str(run))
            command=[sys.executable,'-X','utf8',str(B/'train_one.py'),'--model',m['id']]+(['--smoke'] if smoke else [])
            save(B/'logs/status.json',{'status':'preflight' if smoke else 'training','active':name,'completed':completed,'start_unix':began,'elapsed_seconds':time.time()-began})
            print('START',name,flush=True)
            with (B/'logs'/f'{name}.log').open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=R,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:
                save(B/'logs/status.json',{'status':'failed','active':name,'completed':completed,'exit_code':r.returncode});raise RuntimeError(name)
            result=json.loads((run/'result.json').read_text());completed.append(name);print('COMPLETE',name,result['elapsed_seconds'],flush=True)
        if smoke:
            histories=[json.loads((B/'runs'/('smoke_'+m['id'])/'train/progress.json').read_text()) for m in cfg['models']]
            assert len({tuple(h['first_batch_sha256'] for h in x) for x in histories})==1
            save(B/'audit/preflight.json',{'all_four_complete':True,'matched_smoke_batches':True,'loss_and_gradients_finite':True,'EMA_reload_exact':True,'no_evaluation':True})
    jobs=[]
    for m in cfg['models']:
        result=json.loads((B/'runs'/m['id']/'result.json').read_text());assert result['epochs']==50 and result['all50_first_batches_match_archived_A']
        jobs.append({'model':m['id'],'label':m['label'],'checkpoint':result['checkpoint'],'checkpoint_sha256':result['checkpoint_sha256']})
    save(B/'audit/weights_frozen.json',{'protocol_sha256':sha(B/'protocol.json'),'manifest_sha256':sha(E/'data/staged_manifest.json'),'runs':jobs,'time_unix':time.time(),'new_final_inference_before_freeze':False})
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    records=json.loads((E/'data/staged_manifest.json').read_text())['test'];summary=[]
    for j in jobs:
        save(B/'logs/status.json',{'status':'evaluation','active':j['model'],'completed':completed,'elapsed_seconds':time.time()-began})
        path=R/j['checkpoint'];assert sha(path)==j['checkpoint_sha256'];model,state=load(path,'cuda');del state;gc.collect()
        output=B/'results'/j['model'];result=evaluate_model(model,output,records=records,compress=True)
        result.update(split='grouped_test389',model=j['model'],seed=17,epoch=50,checkpoint_sha256=j['checkpoint_sha256'],protocol_sha256=sha(B/'protocol.json'),manifest_sha256=sha(E/'data/staged_manifest.json'))
        with gzip.open(output/'predictions.json.gz','rt',encoding='utf-8') as f:pred=json.load(f)
        rescored=score(records,pred);assert rescored['metrics']==result['metrics'] and rescored['primary']==result['primary'];result['exact_rescore']=True
        save(output/'metrics.json',result);summary.append({'model':j['model'],'label':j['label'],'primary':result['primary'],'metrics':str((output/'metrics.json').relative_to(R))});print('EVALUATED',j['model'],result['primary'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    for item in cfg['reuse']:
        p=R/item['metrics'];assert sha(p)==item['metrics_sha256'] and sha(p.parent/'predictions.json.gz')==item['predictions_sha256']
        result=json.loads(p.read_text());summary.append({'model':item['variant'],'label':item['label'],'primary':result['primary'],'metrics':item['metrics']})
    save(B/'results/comparison.json',{'protocol_sha256':sha(B/'protocol.json'),'metric':'UAV/all/AP50:95','seed':17,'epochs':50,'models':summary,'elapsed_seconds':time.time()-began})
    save(B/'logs/status.json',{'status':'complete','completed':completed,'elapsed_seconds':time.time()-began})
    print('ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
