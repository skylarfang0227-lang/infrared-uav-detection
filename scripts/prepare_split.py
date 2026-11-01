"""Data-only, pre-score split selection and read-only source staging."""
from pathlib import Path
import json,hashlib,os,shutil
from collections import defaultdict
import numpy as np
E=Path(__file__).resolve().parents[1];R=E
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    cfg=json.loads((E/'configs/grouped_protocol.json').read_text())['split']
    rows=json.loads((E/'data/background_catalog.json').read_text());groups=defaultdict(list)
    for r in rows:groups[r['background_group']].append(r)
    keys=sorted(groups,key=int);features=[]
    for k in keys:
        rs=groups[k];boxes=[(b,r) for r in rs for b in r['boxes']]
        features.append([len(rs),sum(b[0]==0 for b,r in boxes),sum(b[0]==1 for b,r in boxes),sum(b[0]==0 and b[3]*r['width']*b[4]*r['height']<=16 for b,r in boxes)])
    f=np.array(features);total=f.sum(0);large=f[:,0]>cfg['large_group_train_threshold'];rng=np.random.default_rng(cfg['seed']);best=None
    for trial in range(cfg['trials']):
        a=rng.choice(3,len(keys),p=[.5,.25,.25]);a[large]=0
        if min(sum(a==1),sum(a==2))<cfg['minimum_heldout_groups']:continue
        counts=np.array([f[a==s].sum(0) for s in range(3)])
        objective=float(np.square(counts/total-np.array(cfg['fractions'])[:,None]).sum())
        if best is None or objective<best[0]:best=(objective,trial,a.copy(),counts)
    if best is None:raise RuntimeError('No feasible split')
    objective,trial,a,counts=best;mapping={k:['train','dev','test'][int(s)] for k,s in zip(keys,a)};staged={s:[] for s in ['train','dev','test']}
    if (E/'data/staged_manifest.json').exists():raise FileExistsError('Frozen split already exists')
    for r in rows:
        split=mapping[r['background_group']];source=R/r['image'];label=Path(str(source).replace('/images/','/labels/').replace('\\images\\','\\labels\\')).with_suffix('.txt')
        if sha(source)!=r['image_sha256'] or sha(label)!=r['label_sha256']:raise ValueError('Source fingerprint changed')
        image=E/f'data/dataset/images/{split}/{r["id"]}.jpg';lab=E/f'data/dataset/labels/{split}/{r["id"]}.txt'
        image.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,image)
        except PermissionError:shutil.copyfile(source,image)
        shutil.copyfile(label,lab)
        staged[split].append(dict(r,source_image=r['image'],image=str(image),label=str(lab),split=split))
    for s,rs in staged.items():
        rs.sort(key=lambda r:int(r['id']));(E/f'data/{s}.txt').write_text(''.join(r['image']+'\n' for r in rs),encoding='utf-8')
    (E/'data/smoke_train.txt').write_text(''.join(r['image']+'\n' for r in staged['train'][:64]),encoding='utf-8')
    save(E/'data/staged_manifest.json',staged)
    report={'configuration':cfg,'objective':objective,'selected_trial':trial,'feature_order':['images','UAV boxes','Bird boxes','tiny UAV boxes'],'counts':dict(zip(staged,counts.tolist())),'groups':{s:sum(mapping[k]==s for k in keys) for s in staged},'group_assignment':mapping,'source_catalog_sha256':sha(E/'data/background_catalog.json'),'protocol_sha256':sha(E/'configs/grouped_protocol.json'),'manifest_sha256':sha(E/'data/staged_manifest.json'),'cross_boundary_group_overlap':0,'true_scene_independence':'unknown'}
    save(E/'analysis/split_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='group_assignment'},indent=2))
if __name__=='__main__':main()
