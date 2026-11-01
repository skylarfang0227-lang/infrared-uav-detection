"""Select the original real-image examples from retained predictions, without image synthesis."""
from pathlib import Path
import json,gzip,hashlib,collections
N=Path(__file__).resolve().parents[1];E=N
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);w=max(0,min(a[0]+a[2],b[0]+b[2])-x);h=max(0,min(a[1]+a[3],b[1]+b[3])-y)
    q=w*h;return q/(a[2]*a[3]+b[2]*b[3]-q)
def main():
    # Post-hoc illustration rule fixed before selecting any individual case.
    records=sorted(read(E/'data/staged_manifest.json')['test'],key=lambda r:int(r['id']))
    selections={};counts=collections.Counter();sources={}
    for seed in [17,29,43]:
        tables=[]
        for v in ['reference','head']:
            p=E/f'results/test/{v}_s{seed}_e50/predictions.json.gz';sources[str(p.relative_to(E))]=hashlib.sha256(p.read_bytes()).hexdigest()
            with gzip.open(p,'rt') as f:ps=json.load(f)
            table=collections.defaultdict(list)
            for pred in ps:
                if pred['category_id']==1 and pred['score']>=.25:table[pred['image_id']].append(pred)
            tables.append(table)
        for r in records:
            for index,(cls,x,y,w,h) in enumerate(r['boxes']):
                area=w*r['width']*h*r['height']
                if cls!=0 or not 0<area<=16:continue
                bb=[(x-w/2)*r['width'],(y-h/2)*r['height'],w*r['width'],h*r['height']];best=[]
                for table in tables:
                    ps=table[int(r['id'])];p=max(ps,key=lambda p:iou(bb,p['bbox'])) if ps else None
                    best.append({'prediction':p,'iou':iou(bb,p['bbox']) if p else 0})
                a,b=[x['iou'] for x in best]
                conditions={'cross75':a<.75<=b,'loss50':b<.5<=a,'shared_miss':a<.5 and b<.5}
                for case,yes in conditions.items():
                    if yes:
                        counts[case]+=1
                        selections.setdefault(case,{'seed':seed,'image_id':r['id'],'image':r['image'],'annotation_index':index,'box':bb,'area':area,'A':best[0],'H':best[1]})
    out={'selection_rule':'Post-hoc illustration: first seed in 17/29/43 then numeric image ID then annotation index for each declared outcome; native area<=16; score>=.25; per-GT best IoU, not one-to-one COCO matching.','counts_across_seed_target_pairs':dict(counts),'selected':selections,'sources_sha256':sources}
    (N/'analysis/qualitative_selection.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:v[a] for a in ['seed','image_id','area']} for k,v in selections.items()},indent=2))

if __name__=='__main__':main()
