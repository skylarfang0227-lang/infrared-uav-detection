"""Post-hoc CPU analysis of frozen predictions; no inference, training or raw edits."""
from pathlib import Path
import os,sys,json,gzip,hashlib,collections,statistics
import numpy as np
from scipy.stats import t
N=Path(__file__).resolve().parents[1];E=N;R=E
os.environ['YOLO_CONFIG_DIR']=str(N/'analysis/ultralytics')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(R))
from scripts import evaluate_sirst_protocol as evaluator

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(a):
    a=np.asarray(a,dtype=float);m=float(a.mean());s=float(a.std(ddof=1));h=float(t.ppf(.975,len(a)-1)*s/np.sqrt(len(a)))
    return {'values':a.tolist(),'mean':m,'sample_sd':s,'descriptive_t95':[m-h,m+h]}

class Capture(evaluator.COCOeval):
    latest=None
    def accumulate(self,*a,**kw):
        result=super().accumulate(*a,**kw);Capture.latest=self;return result

def main():
    evaluator.COCOeval=Capture
    records=read(E/'data/staged_manifest.json')['test']
    result={'status':'post-hoc descriptive analysis; primary endpoint unchanged','source_evaluator_sha256':sha(R/'scripts/evaluate_sirst_protocol.py'),
            'manifest_sha256':sha(E/'data/staged_manifest.json'),'iou_thresholds':np.linspace(.5,.95,10).tolist(),'runs':[]}
    for variant in ['reference','head','stem','combined']:
        for seed in [17,29,43]:
            folder=E/f'results/test/{variant}_s{seed}_e50';old=read(folder/'metrics.json');p=folder/'predictions.json.gz'
            assert sha(p)==old['predictions_sha256']
            with gzip.open(p,'rt') as f:pred=json.load(f)
            now=evaluator.score(records,pred)
            assert now['metrics']==old['metrics'] and now['primary']==old['primary'],folder
            ev=Capture.latest;curves={}
            for ai,group in enumerate(ev.params.areaRngLbl):
                pr=ev.eval['precision'][:,:,0,ai,-1];rec=ev.eval['recall'][:,0,ai,-1]
                curves[group]={'AP':[float(x[x>=0].mean()) for x in pr],'recall_at_retained_detection_limit':rec.tolist()}
            result['runs'].append({'variant':variant,'seed':seed,'source':str(folder.relative_to(E)).replace('\\','/'),
                                   'prediction_sha256':sha(p),'exact_rescore_match':True,'curves':curves})
            print(variant,seed,'exact match',flush=True)
    contrasts={}
    for group in ['all','tiny']:
        for metric,idx in [('AP50',0),('AP75',5)]:
            for measure in ['AP','recall_at_retained_detection_limit']:
                a=[r['curves'][group][measure][idx] for r in result['runs'] if r['variant']=='reference']
                h=[r['curves'][group][measure][idx] for r in result['runs'] if r['variant']=='head']
                contrasts[f'{group}/{metric}/{measure}/H-A']=stats(np.array(h)-a)
    result['contrasts']=contrasts
    tiny=collections.Counter()
    for r in records:
        for cls,x,y,w,h in r['boxes']:
            if cls==0 and 0<w*r['width']*h*r['height']<=16:tiny[r['background_group']]+=1
    result['tiny_group_counts']=dict(tiny);result['tiny_targets']=sum(tiny.values())
    (N/'analysis/threshold_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'contrasts':contrasts,'tiny_groups':dict(tiny)},indent=2))

if __name__=='__main__':main()
