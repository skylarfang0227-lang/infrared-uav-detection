from pathlib import Path
import json,numpy as np
E=Path(__file__).resolve().parents[1]
rows=json.loads((E/'data/staged_manifest.json').read_text())['train'];records=[]
for r in rows:
    scale=min(640/r['width'],640/r['height']);padx=(640-round(r['width']*scale))/2;pady=(640-round(r['height']*scale))/2
    for k,(cls,xc,yc,w,h) in enumerate(r['boxes']):
        if cls!=0:continue
        x1=(xc-w/2)*r['width']*scale+padx;x2=(xc+w/2)*r['width']*scale+padx;y1=(yc-h/2)*r['height']*scale+pady;y2=(yc+h/2)*r['height']*scale+pady;area=w*r['width']*h*r['height'];out={'id':r['id'],'annotation':k,'area':area,'group':'tiny' if area<=16 else '16to49' if area<=49 else 'gt49'}
        for s in [4,2]:
            grid=(np.arange(640//s)+.5)*s;x=grid[(grid>x1)&(grid<x2)];y=grid[(grid>y1)&(grid<y2)]
            nx=sum((x-x1<15*s)&(x2-x<15*s));ny=sum((y-y1<15*s)&(y2-y<15*s));out[str(s)]={'inside':len(x)*len(y),'inside_and_range':int(nx*ny)}
        records.append(out)
summary={}
for name in ['all','tiny','16to49','gt49']:
    rs=[r for r in records if name=='all' or r['group']==name];summary[name]={'targets':len(rs)}
    for s in ['4','2']:
        summary[name][s]={'mean_inside_sites':float(np.mean([r[s]['inside'] for r in rs])),'zero_inside':sum(r[s]['inside']==0 for r in rs),'no_inside_and_range':sum(r[s]['inside_and_range']==0 for r in rs)}
result={'protocol':json.loads((E/'configs/geometry.json').read_text()),'summary':summary,'targets':records};(E/'analysis/geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
