"""Reconstruct frozen similarity grouping and inspect fixed retained-edge samples."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import cv2
from PIL import Image,ImageDraw
N=Path(__file__).resolve().parents[1];E=N;R=E;O=N/'analysis'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class DSU:
    def __init__(self,keys):self.p={k:k for k in keys}
    def root(self,k):
        while self.p[k]!=k:self.p[k]=self.p[self.p[k]];k=self.p[k]
        return k
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.p[max(a,b,key=int)]=min(a,b,key=int)
def main():
    dest=O/'group_verification.json';assert not dest.exists();cv2.setNumThreads(1)
    catalog=read(E/'data/pixel_unique_catalog.json');pe=read(E/'data/similarity_edges.json');bg=read(E/'analysis/background_links.json')
    rows=read(E/'data/staged_manifest.json');split={str(r['id']):s for s,rs in rows.items() for r in rs}
    result={'protocol_sha256':sha(E/'configs/diagnostics.json'),'inputs':{p:sha(E/p) for p in ['data/pixel_unique_catalog.json','data/similarity_edges.json','analysis/background_links.json','data/staged_manifest.json']},'perceptual':{}}
    for mode in ['main','broad']:
        ds=DSU(range(len(catalog)));cross=0
        for a,b,h,mae in pe[mode]:
            ds.join(a,b);cross+=split[str(catalog[a]['id'])]!=split[str(catalog[b]['id'])]
        mapping={str(catalog[i]['id']):str(catalog[ds.root(i)]['id']) for i in range(len(catalog))}
        assert all(mapping[str(r['id'])]==r[f'{mode}_group'] for r in catalog)
        assert cross==0
        result['perceptual'][mode]={'components':len(set(mapping.values())),'edges':len(pe[mode]),'cross_split_edges':cross}
    broad=collections.defaultdict(list)
    for r in catalog:broad[r['broad_group']].append(r)
    reps={g:min(rs,key=lambda r:int(r['id'])) for g,rs in broad.items()}
    for mode in ['original','stricter_retained_edges']:
        ds=DSU(broad);count=0;cross=0
        for e in bg['edges']:
            if mode!='original' and not (e['inliers']>=24 and e['ratio']>=.7 and min(e['span'])>=.15 and .75<=e['scale']<=1.33 and e['angle']<=20):continue
            a,b=e['groups'];ds.join(a,b);count+=1;cross+=split[str(reps[a]['id'])]!=split[str(reps[b]['id'])]
        mapping={g:ds.root(g) for g in broad};sizes=collections.Counter(mapping[r['broad_group']] for r in catalog)
        if mode=='original':assert mapping==bg['mapping']
        result[mode]={'edges':count,'groups':len(sizes),'cross_split_edges':cross,'largest_sizes':sorted(sizes.values(),reverse=True)[:8]}
    large={k:v for k,v in bg['group_sizes'].items() if v>187}
    assert all(split[str(r['id'])]=='train' for r in catalog if bg['mapping'][r['broad_group']] in large)
    result['group_cap']={'threshold':187,'groups_above_cap':large,'images_forced_train':sum(large.values()),'target_heldout_images':len(catalog)*.15,
                         'rationale_status':'Original log states largest groups remain in training by count rule; exact numerical187 optimality/calibration not documented. Cap is approximately half the nominal heldout size, a numerical relation rather than proof of original intent.'}
    edges=bg['edges'];chosen=sorted(range(len(edges)),key=lambda i:(edges[i]['inliers'],i))[:4]
    chosen+=sorted((i for i in range(len(edges)) if i not in chosen),key=lambda i:(min(edges[i]['span']),i))[:4]
    remaining=[i for i in range(len(edges)) if i not in chosen];chosen+=[remaining[k] for k in np.linspace(0,len(remaining)-1,4,dtype=int)]
    result['visual_sample']=[]
    for page in range(3):
        sheet=Image.new('RGB',(1200,920),'white');draw=ImageDraw.Draw(sheet)
        for k,idx in enumerate(chosen[page*4:page*4+4]):
            e=edges[idx];a,b=[reps[g] for g in e['groups']]
            ia=cv2.imread(str(R/a['image']),0);ib=cv2.imread(str(R/b['image']),0)
            warp=cv2.warpAffine(ia,np.array(e['affine']),(640,512));blend=np.stack([ib,warp,ib],axis=-1)
            draw.text((10,k*230+3),f"edge {idx}, ids {a['id']}/{b['id']}; inliers {e['inliers']}; ratio {e['ratio']:.3f}; span {min(e['span']):.3f}",fill='black')
            for j,arr in enumerate([ia,ib,blend]):
                im=Image.fromarray(arr).convert('RGB');im.thumbnail((390,200));sheet.paste(im,(j*400+5,k*230+25))
            result['visual_sample'].append({'edge_index':idx,'ids':[a['id'],b['id']],**e})
        sheet.save(O/f'background_pairs_{page+1}.png')
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['visual_sample','inputs']},indent=2))
if __name__=='__main__':main()
