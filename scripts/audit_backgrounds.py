from pathlib import Path
import json,collections,time,math
import cv2,numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parents[1];R=E
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 cv2.setNumThreads(2);cv2.setRNGSeed(20261029);t=time.monotonic()
 cfg=json.loads((E/'configs/background_link_audit.json').read_text());rows=json.loads((E/'data/pixel_unique_catalog.json').read_text());groups=collections.defaultdict(list)
 for r in rows:groups[r['broad_group']].append(r)
 ids=sorted(groups,key=int);orb=cv2.ORB_create(nfeatures=600);features=[]
 for g in ids:
  r=min(groups[g],key=lambda x:int(x['id']));gray=cv2.imread(str(R/r['image']),0);k,d=orb.detectAndCompute(gray,None);features.append((np.float32([x.pt for x in k]),d))
 parent=list(range(len(ids)))
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 edges=[];matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
 for i in range(len(ids)-1):
  a,da=features[i]
  if da is None or len(da)<18:continue
  for j in range(i+1,len(ids)):
   b,db=features[j]
   if db is None or len(db)<18:continue
   matches=matcher.knnMatch(da,db,k=2);good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.70*n.distance]
   if len(good)<18:continue
   x=np.float32([a[m.queryIdx] for m in good]);y=np.float32([b[m.trainIdx] for m in good])
   M,mask=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1000,confidence=.995,refineIters=10)
   if M is None:continue
   mask=mask.ravel().astype(bool);n=int(mask.sum());ratio=n/len(good)
   if n<18 or ratio<.6:continue
   spanx=float(np.prod(np.ptp(x[mask],axis=0))/(640*512));spany=float(np.prod(np.ptp(y[mask],axis=0))/(640*512));scale=math.hypot(M[0,0],M[1,0]);angle=abs(math.degrees(math.atan2(M[1,0],M[0,0])))
   if min(spanx,spany)<.1 or not .6<=scale<=1.7 or angle>35:continue
   aa,bb=root(i),root(j);parent[max(aa,bb)]=min(aa,bb)
   edges.append({'groups':[ids[i],ids[j]],'inliers':n,'matches':len(good),'ratio':ratio,'span':[spanx,spany],'scale':scale,'angle':angle,'affine':M.tolist()})
  if i%100==0:print(json.dumps({'representative':i,'of':len(ids),'edges':len(edges),'seconds':time.monotonic()-t}),flush=True)
 mapping={g:ids[root(i)] for i,g in enumerate(ids)};sizes=collections.Counter(mapping[r['broad_group']] for r in rows)
 for r in rows:r['background_group']=mapping[r['broad_group']]
 save(E/'data/background_catalog.json',rows);save(E/'analysis/background_links.json',{'config':cfg,'edges':edges,'mapping':mapping,'group_sizes':dict(sizes),'seconds':time.monotonic()-t})
 merged=collections.defaultdict(list)
 for g,target in mapping.items():merged[target].append(g)
 panels=[]
 for target,gs in sorted(merged.items(),key=lambda p:(-sizes[p[0]],int(p[0])))[:16]:
  panel=Image.new('RGB',(960,180),'white');d=ImageDraw.Draw(panel)
  selected=[gs[k] for k in np.linspace(0,len(gs)-1,min(4,len(gs)),dtype=int)]
  for k,g in enumerate(selected):
   row=groups[g][0];im=Image.open(R/row['image']).convert('RGB');im.thumbnail((230,145));panel.paste(im,(k*240,25));d.text((k*240+2,3),f"bg{target}; n={sizes[target]}; perceptual={g}",fill='black')
  panels.append(panel)
 image=Image.new('RGB',(960,len(panels)*180),'white')
 for i,p in enumerate(panels):image.paste(p,(0,i*180))
 image.save(E/'analysis/background_groups.png');print(json.dumps({'groups':len(sizes),'largest':sizes.most_common(20),'edges':len(edges),'seconds':time.monotonic()-t}),flush=True)
if __name__=='__main__':main()
