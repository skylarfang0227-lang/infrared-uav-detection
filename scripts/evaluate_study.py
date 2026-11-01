"""Explicit frozen assessment using the original matching and inference rules."""
from pathlib import Path
import os,sys,json,argparse,gzip,time,hashlib,gc
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
import torch
from detector_variants import load_detector
from src.msda_training import load_model
from src.msda_subpixel_mpm import load_subpixel_mpm_model
from scripts.evaluate_sirst_protocol import evaluate_model,score
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['dev','test','sidd'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if a.study=='sidd':
        audit=json.loads((E/'analysis/sidd_audit.json').read_text());assert audit['clean_unique']>0
        records=json.loads((E/'data/sidd_extended_first_records.json').read_text());alternate=json.loads((E/'data/sidd_extended_last_records.json').read_text());clean=json.loads((E/'data/sidd_clean_records.json').read_text())
        jobs=json.loads((R/'experiments/original_split/frozen/weights_frozen.json').read_text())['runs']
        jobs=[dict(j,variant=j['method'],epoch=50) for j in jobs]
    else:
        records=json.loads((E/'data/staged_manifest.json').read_text())[a.study]
        frozen=json.loads((E/'results/factorial_weights_frozen.json').read_text());cfg=json.loads((E/'configs/grouped_protocol.json').read_text())
        assert frozen['protocol_sha256']==sha(E/'configs/grouped_protocol.json') and frozen['manifest_sha256']==sha(E/'data/staged_manifest.json')
        assert {(j['variant'],j['seed']) for j in frozen['runs']}=={(v,s) for v in cfg['variants'] for s in cfg['seeds']}
        for j in frozen['runs']:assert sha(R/j['checkpoint'])==j['checkpoint_sha256']
        jobs=[dict(j,epoch=50) for j in frozen['runs']]
        if a.study=='dev':
            jobs=[]
            for j in json.loads((E/'results/training_index.json').read_text()):
                if j['seed']!=17 or j['variant'] not in ['reference','head','combined','phase_collapsed']:continue
                for epoch in [50]:
                    checkpoint=(R/j['checkpoint']).with_name(f'epoch{epoch}.pt');jobs.append(dict(j,epoch=epoch,checkpoint=str(checkpoint.relative_to(R)),checkpoint_sha256=sha(checkpoint)))
    out=E/'results'/a.study;out.mkdir(exist_ok=False);summary=[]
    for j in jobs:
        path=R/j['checkpoint'];assert sha(path)==j['checkpoint_sha256'];name=f'{j["variant"]}_s{j["seed"]}_e{j["epoch"]}'
        loader=(load_model if j['variant']=='reference' else load_subpixel_mpm_model) if a.study=='sidd' else load_detector
        model,state=loader(path,'cuda');del state;gc.collect()
        result=evaluate_model(model,out/name,records=records,compress=True)
        # The historical helper's dev-only entry point receives explicit records;
        # overwrite its label with the actual frozen study, never its calculations.
        result.update(split='study_'+a.study,variant=j['variant'],seed=j['seed'],epoch=j['epoch'],checkpoint_sha256=j['checkpoint_sha256'],records_sha256=sha(E/'data'/('sidd_extended_first_records.json' if a.study=='sidd' else 'staged_manifest.json')),evaluation_code_sha256=sha(R/'scripts/evaluate_sirst_protocol.py'))
        if a.study=='sidd':
            with gzip.open(out/name/'predictions.json.gz','rt') as f:pred=json.load(f)
            ids={int(r['id']) for r in clean};clean_pred=[p for p in pred if p['image_id'] in ids];full=dict(result);screened=score(clean,clean_pred)
            result.update(screened);result.update(image_count=len(clean),inference_image_count=len(records),screened=screened,all_unique_first_annotations=full,all_unique_last_annotations=score(alternate,pred),primary_records_sha256=sha(E/'data/sidd_clean_records.json'),alternate_records_sha256=sha(E/'data/sidd_extended_last_records.json'),scene_categories={})
            for scene in ['city','moutain','sea','sky']:
                rr=[r for r in clean if r['scene_category']==scene];ii={int(r['id']) for r in rr}
                result['scene_categories'][scene]={'images':len(rr),'metrics':score(rr,[p for p in clean_pred if p['image_id'] in ii]) if rr else None}
        save(out/name/'metrics.json',result);summary.append(result);save(out/'summary.json',summary);print(json.dumps({'study':a.study,'run':name,'primary':result['primary'],'screened':result.get('screened',{}).get('primary')}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
