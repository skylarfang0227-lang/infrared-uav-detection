"""Fixed post-hoc analysis of existing COCO matches, without new inference."""
from pathlib import Path
import os,sys,json,gzip,hashlib,collections,time,csv
import numpy as np
N=Path(__file__).resolve().parents[1];E=N;R=E
O=N/'analysis'
os.environ['YOLO_CONFIG_DIR']=str(O/'ultralytics');sys.path.insert(0,str(R))
from scripts import evaluate_sirst_protocol as evaluator
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Capture(evaluator.COCOeval):
    latest=None
    def accumulate(self,*args,**kwargs):
        result=super().accumulate(*args,**kwargs);Capture.latest=self;return result
def iou(a,b):
    if not b:return np.empty(0)
    a=np.asarray(a,float);b=np.asarray(b,float)
    wh=np.maximum(0,np.minimum(a[:2]+a[2:],b[:,:2]+b[:,2:])-np.maximum(a[:2],b[:,:2]))
    inter=wh.prod(1);return inter/(a[2]*a[3]+b[:,2:].prod(1)-inter)
def main():
    dest=O/'prediction_diagnostics.json';assert not dest.exists()
    started=time.monotonic();evaluator.COCOeval=Capture
    records=read(E/'data/staged_manifest.json')['test'];runs=[];matched={};targetrows=[]
    bins=np.array([.001,.01,.1,.25,.5,1.000001])
    for variant in ['reference','head']:
        for seed in [17,29,43]:
            folder=E/f'results/test/{variant}_s{seed}_e50';p=folder/'predictions.json.gz';old=read(folder/'metrics.json')
            assert sha(p)==old['predictions_sha256']
            with gzip.open(p,'rt',encoding='utf-8') as f:pred=json.load(f)
            new=evaluator.score(records,pred);assert new['metrics']==old['metrics']
            ev=Capture.latest;run={'variant':variant,'seed':seed,'sha256':sha(p),'exact_rescore':True,'thresholds':{}}
            targets={a['id']:a for a in ev.cocoGt.anns.values() if a['category_id']==1}
            for ti,tau in [(0,.5),(5,.75)]:
                counts=collections.Counter();scores=[collections.Counter() for _ in range(len(bins)-1)];hit=set()
                for entry in ev.evalImgs:
                    if entry is None or entry['category_id']!=1 or entry['aRng']!=[0.,1e10]:continue
                    allgt=ev.cocoGt.imgToAnns[entry['image_id']]
                    same=[g['bbox'] for g in allgt if g['category_id']==1];other=[g['bbox'] for g in allgt if g['category_id']==2]
                    hit.update(g for g,m,ign in zip(entry['gtIds'],entry['gtMatches'][ti],entry['gtIgnore']) if m and not ign)
                    for did,m,ign in zip(entry['dtIds'],entry['dtMatches'][ti],entry['dtIgnore'][ti]):
                        if ign:continue
                        d=ev.cocoDt.anns[did];ix=int(np.searchsorted(bins,d['score'],side='right')-1);assert 0<=ix<len(scores)
                        if m:kind='TP'
                        else:
                            sameiou=iou(d['bbox'],same);otheriou=iou(d['bbox'],other)
                            if len(sameiou) and sameiou.max()>=tau:kind='duplicate'
                            elif len(sameiou) and sameiou.max()>=.1:kind='localization'
                            elif len(otheriou) and otheriou.max()>=tau:kind='class_confusion'
                            else:kind='background'
                        counts[kind]+=1;scores[ix][kind]+=1
                assert len(hit)==counts['TP']
                matched[(variant,seed,tau)]=hit
                run['thresholds'][str(tau)]={'detections':dict(counts),'score_bins':[dict(x) for x in scores],
                    'matched_all':len(hit),'matched_tiny':sum(targets[g]['area']<=16 for g in hit)}
                assert abs(len(hit)/len(targets)-ev.eval['recall'][ti,0,0,-1])<1e-12
            runs.append(run);print(variant,seed,'verified',flush=True)
    paired=[]
    for seed in [17,29,43]:
        for tau in [.5,.75]:
            a=matched[('reference',seed,tau)];h=matched[('head',seed,tau)]
            for group in ['all','tiny']:
                subset={g for g,t in targets.items() if group=='all' or t['area']<=16}
                counts={'both':len(a&h&subset),'A_only':len((a-h)&subset),'H_only':len((h-a)&subset),'neither':len(subset-(a|h))}
                assert sum(counts.values())==len(subset)
                paired.append({'seed':seed,'iou':tau,'group':group,'targets':len(subset),**counts})
            for g,t in targets.items():targetrows.append({'seed':seed,'iou':tau,'annotation_id':g,'image_id':t['image_id'],'area':t['area'],'A_match':int(g in a),'H_match':int(g in h)})
    aggregate=[]
    for group in ['all','tiny']:
        for tau in [.5,.75]:
            rs=[p for p in paired if p['group']==group and p['iou']==tau]
            aggregate.append({'group':group,'iou':tau,**{k:sum(r[k] for r in rs) for k in ['both','A_only','H_only','neither']},'unit':'target-seed occurrences, not independent targets'})
    result={'protocol_sha256':sha(E/'configs/diagnostics.json'),'manifest_sha256':sha(E/'data/staged_manifest.json'),'evaluator_sha256':sha(R/'scripts/evaluate_sirst_protocol.py'),
            'runs':runs,'paired':paired,'aggregate':aggregate,'score_bin_edges':bins.tolist(),'seconds':time.monotonic()-started,
            'scope':'All-area COCO GT matches stratified by GT native area. Diagnostic unmatched categories are ordered and post-NMS; cannot identify NMS suppression or causal geometric error.'}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (O/'target_matches.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(targetrows[0]));w.writeheader();w.writerows(targetrows)
    print(json.dumps({'aggregate':aggregate,'paired':paired},indent=2))
if __name__=='__main__':main()
