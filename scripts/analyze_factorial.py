"""All-seed factorial contrasts and paired background-proxy sensitivity."""
from pathlib import Path
import os,sys,json,gzip,collections,time
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');sys.path.insert(0,str(E/'src'))
import numpy as np
from scipy.stats import t
from cluster_ap import cache_matches,ap
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def stat(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return {'values':x.tolist(),'mean':mean,'sample_sd':sd,'descriptive_t95':[mean-half,mean+half]}
def main():
    assert json.loads((E/'analysis/cluster_accumulator_validation.json').read_text())['all_pass']
    cfg=json.loads((E/'configs/group_uncertainty.json').read_text());rows=json.loads((E/'data/staged_manifest.json').read_text())['test'];results=json.loads((E/'results/test/summary.json').read_text());seeds=[17,29,43];variants=['reference','head','stem','combined'];metrics={}
    for v in variants:
        metrics[v]={}
        for name,get in [('primary',lambda r:r['primary']),('UAV_AP50',lambda r:r['metrics']['UAV/all']['ap50']),('tiny_AP50_95',lambda r:r['metrics']['UAV/tiny']['ap50_95']),('Bird_AP50_95',lambda r:r['metrics']['Bird/all']['ap50_95']),('two_class_AP50_95',lambda r:r['two_class_ap50_95'])]:
            metrics[v][name]=stat([get(next(r for r in results if r['variant']==v and r['seed']==s)) for s in seeds])
    A,H,S,HS=(np.array(metrics[v]['primary']['values']) for v in variants)
    contrasts={k:stat(x) for k,x in {'head_without_stem':H-A,'stem_without_head':S-A,'head_with_stem':HS-S,'stem_with_head':HS-H,'average_head_effect':(H-A+HS-S)/2,'average_stem_effect':(S-A+HS-H)/2,'interaction':HS-H-S+A,'combined_vs_reference':HS-A}.items()}
    groups=collections.defaultdict(list)
    for r in sorted(rows,key=lambda r:int(r['id'])):groups[r['background_group']].append(int(r['id']))
    keys=sorted(groups,key=int);cache={};allids=sorted(int(r['id']) for r in rows)
    for v in variants:
        for s in seeds:
            with gzip.open(E/f'results/test/{v}_s{s}_e50/predictions.json.gz','rt') as f:pred=json.load(f)
            c,expected=cache_matches(rows,pred);actual=ap(c,allids);saved=next(r['primary'] for r in results if r['variant']==v and r['seed']==s)
            assert abs(actual-expected)<1e-12 and abs(actual-saved)<1e-12;cache[(v,s)]=c
    def paired(ids):return [ap(cache[('combined',s)],ids)-ap(cache[('reference',s)],ids) for s in seeds]
    leave=[]
    for g in keys:
        ids=[i for i in allids if i not in groups[g]];leave.append({'omitted_group':g,'images_removed':len(groups[g]),'paired':paired(ids)})
    pergroup=[]
    for g in keys:
        ids=groups[g];targets=sum(b[0]==0 for r in rows if r['background_group']==g for b in r['boxes'])
        pergroup.append({'group':g,'images':len(ids),'UAV_targets':targets,'supported':targets>=10,'reference':[ap(cache[('reference',s)],ids) for s in seeds],'combined':[ap(cache[('combined',s)],ids) for s in seeds]})
    extension=json.loads((E/'configs/component_group_uncertainty.json').read_text())
    assert extension['registered_before_new_model_scores']
    rng=np.random.default_rng(cfg['seed']);replicates=[];component_replicates={k:[] for k in extension['additional_contrasts']};started=time.monotonic()
    for i in range(cfg['bootstrap_replicates']):
        selected=rng.integers(0,len(keys),len(keys));ids=sorted(x for j in selected for x in groups[keys[j]])
        a,h,s,hs=[np.array([ap(cache[(v,seed)],ids) for seed in seeds]) for v in variants]
        difference=(hs-a).tolist();replicates.append(difference)
        effects={'head_without_stem':h-a,'stem_without_head':s-a,'head_with_stem':hs-s,'stem_with_head':hs-h,'average_head_effect':(h-a+hs-s)/2,'average_stem_effect':(s-a+hs-h)/2,'interaction':hs-h-s+a}
        for key,values in effects.items():component_replicates[key].append(values.tolist())
        if i%100==0:print(json.dumps({'bootstrap':i,'seconds':time.monotonic()-started}),flush=True)
    component_groups={'protocol':extension,'contrasts':{key:{'paired_differences':values,'mean_gain_percentile95':np.quantile(np.mean(values,axis=1),[.025,.975]).tolist()} for key,values in component_replicates.items()}}
    means=np.mean(replicates,axis=1);out={'protocol':cfg,'metrics':metrics,'contrasts':contrasts,'cluster_bootstrap':{'paired_differences':replicates,'mean_gain_percentile95':np.quantile(means,[.025,.975]).tolist(),'groups':len(groups)},'component_group_sensitivity':component_groups,'leave_one_group_out':leave,'per_group':pergroup,'interpretation':'All intervals descriptive; no multiplicity-corrected or independent-scene significance claim.'};save(E/'analysis/factorial_results.json',out);print(json.dumps(contrasts,indent=2),flush=True)
if __name__=='__main__':main()
