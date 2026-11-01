"""Read only the recorded SIRST images and labels; audit all similarity pairs."""
from pathlib import Path
import collections,hashlib,json,time
import cv2
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parents[1];R=E
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class DSU:
 def __init__(self,n):self.p=list(range(n))
 def find(self,a):
  while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
  return a
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)
def main():
 t=time.monotonic();cv2.setNumThreads(1)
 rows=json.loads((R/'data/splits/sirst_uavb/manifest.json').read_text())['records']
 policy=json.loads((E/'configs/data_audit.json').read_text()); assert len(rows)==3000
 pixel_groups=collections.defaultdict(list); thumbs=[]; ph=[]; signatures=[]
 for i,row in enumerate(rows):
  p=(R/row['image']).resolve();assert p.is_relative_to((R/'data/processed/sirst_uavb/images').resolve())
  label=R/'data/processed/sirst_uavb/labels'/row['split']/(p.stem+'.txt')
  assert sha(p)==row['image_sha256'] and sha(label)==row['label_sha256']
  im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);assert im.shape==(row['height'],row['width'])
  px=hashlib.sha256(im.tobytes()).hexdigest();pixel_groups[px].append(i)
  rows[i]['decoded_sha256']=px
  signatures.append(tuple(sorted(tuple(round(float(x),6) for x in line.split()) for line in label.read_text(encoding='utf-8-sig').splitlines() if line.strip())))
  thumbs.append(cv2.resize(im,(80,64),interpolation=cv2.INTER_AREA).astype(np.int16))
  coeff=cv2.dct(cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()[1:];ph.append(coeff>np.median(coeff))
 conflicts=[ids for ids in pixel_groups.values() if len({signatures[i] for i in ids})>1]
 excluded={i for group in conflicts for i in group}
 representatives=sorted(min(g,key=lambda i:int(rows[i]['id'])) for g in pixel_groups.values() if not set(g)&excluded)
 th=np.asarray(thumbs)[representatives];ph=np.asarray(ph)[representatives];n=len(representatives)
 groups={name:DSU(n) for name in ['main','broad']};edges={name:[] for name in groups}
 for a in range(n-1):
  distances=np.count_nonzero(ph[a+1:]!=ph[a],axis=1)
  idx=np.flatnonzero(distances<=8)+a+1
  for start in range(0,len(idx),128):
   js=idx[start:start+128];mae=np.abs(th[js]-th[a]).mean(axis=(1,2))
   for b,dist in zip(js[mae<=4],mae[mae<=4]):
    h=int(distances[b-a-1]);edge=[a,int(b),h,float(dist)]
    groups['broad'].union(a,int(b));edges['broad'].append(edge)
    if h<=4 and dist<=2:groups['main'].union(a,int(b));edges['main'].append(edge)
  if a%500==0:print(json.dumps({'pair_rows':a,'total':n,'seconds':time.monotonic()-t}),flush=True)
 diagnostics={};catalog=[]
 for mode,ds in groups.items():
  members=collections.defaultdict(list)
  for a in range(n):members[ds.find(a)].append(a)
  sizes=sorted([len(g) for g in members.values()],reverse=True)
  diagnostics[mode]={'components':len(members),'edge_count':len(edges[mode]),'largest_sizes':sizes[:30],'singleton_groups':sizes.count(1),'groups_at_least5':sum(x>=5 for x in sizes)}
  for a,i in enumerate(representatives):
   if mode=='main':catalog.append({**rows[i],'main_group':str(rows[representatives[ds.find(a)]]['id']),'original_duplicate_ids':[rows[j]['id'] for j in pixel_groups[rows[i]['decoded_sha256']]]})
   else:catalog[a]['broad_group']=str(rows[representatives[ds.find(a)]]['id'])
 train_pixels={r['decoded_sha256'] for r in rows if r['split']=='train'}
 original_overlap=sum(r['split']=='val' and r['decoded_sha256'] in train_pixels for r in rows)
 result={'policy':policy,'manifest_sha256':sha(R/'data/splits/sirst_uavb/manifest.json'),'images':len(rows),'unique_decoded_images':len(pixel_groups),'exact_duplicate_excess':len(rows)-len(pixel_groups),'official_eval_exact_train_overlap':original_overlap,'label_conflict_groups':[[rows[i]['id'] for i in g] for g in conflicts],'label_conflict_images':len(excluded),'eligible_pixel_unique_images':n,'grouping':diagnostics,'seconds':time.monotonic()-t,'actual_scene_metadata':'unavailable; groups are visual-similarity proxies, not verified scenes or flights'}
 save(E/'data/pixel_unique_catalog.json',catalog);save(E/'analysis/data_integrity.json',result);save(E/'data/similarity_edges.json',edges)
 # A fixed range of group sizes, not model results, chooses the diagnostic contact sheet.
 member=collections.defaultdict(list)
 for r in catalog:member[r['broad_group']].append(r)
 ordered=sorted(member.values(),key=lambda g:(-len(g),int(g[0]['id'])))
 panels=[]
 for group in ordered[:16]:
  panel=Image.new('RGB',(960,180),'white');d=ImageDraw.Draw(panel)
  for k,row in enumerate([group[i] for i in np.linspace(0,len(group)-1,min(4,len(group)),dtype=int)]):
   im=Image.open(R/row['image']).convert('RGB');im.thumbnail((230,145));panel.paste(im,(k*240,25));d.text((k*240+3,3),f"group {group[0]['broad_group']}; n={len(group)}; id={row['id']}",fill='black')
  panels.append(panel)
 sheet=Image.new('RGB',(960,180*len(panels)),'white')
 for j,p in enumerate(panels):sheet.paste(p,(0,180*j))
 sheet.save(E/'analysis/group_audit.png');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
