from pathlib import Path
import sys,os,json,gzip,copy
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
from cluster_ap import cache_matches,ap
from scripts.evaluate_sirst_protocol import score
rows=json.loads((E/'data/staged_manifest.json').read_text())['test'][:12];ids={int(r['id']) for r in rows}
with gzip.open(E/'results/test/reference_s17_e50/predictions.json.gz','rt') as f:pred=[p for p in json.load(f) if p['image_id'] in ids]
cache,expected=cache_matches(rows,pred);report=[]
for indices in [list(range(12)),[0,1,1,1,5,9,10],[11,2,3,3,3,4,6,7]]:
    rr=[];pp=[]
    for new,i in enumerate(indices):
        row=dict(rows[i],id=str(new+1));rr.append(row)
        pp.extend(dict(p,image_id=new+1) for p in pred if p['image_id']==int(rows[i]['id']))
    a=ap(cache,[int(rows[i]['id']) for i in indices]);b=score(rr,pp)['primary'];assert abs(a-b)<1e-12
    report.append({'resampled_indices':indices,'cached':a,'original':b,'absolute_error':abs(a-b)})
(E/'analysis/cluster_accumulator_validation.json').write_text(json.dumps({'all_pass':True,'cases':report},indent=2));print(report)
