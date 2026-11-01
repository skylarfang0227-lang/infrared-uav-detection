"""Audit the specifically authorized existing SIDD archive; never scan other data."""
from pathlib import Path,PurePosixPath
import zipfile,posixpath,json,hashlib,collections,time,math
import cv2,numpy as np
E=Path(__file__).resolve().parents[1];R=E
def sha(x):return hashlib.sha256(x).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def fingerprint(im):
    coeff=cv2.dct(cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()[1:]
    return coeff>np.median(coeff),cv2.resize(im,(80,64),interpolation=cv2.INTER_AREA).astype(np.int16)
def main():
    started=time.monotonic();cv2.setNumThreads(1);cv2.setRNGSeed(20261029)
    cfg=json.loads((E/'configs/sidd_external.json').read_text());archive=R/'SIDD-main.zip';assert sha(archive.read_bytes())==cfg['archive_sha256']
    output=E/'data/sidd';output.mkdir(exist_ok=False);rows=[];grouped=collections.defaultdict(list);pixels={}
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if '/annotations/instances_' in n and n.endswith('.json'))
        for n in names:
            j=json.loads(z.read(n));assert j['categories']==[{'supercategory':None,'id':0,'name':'UAV'}]
            annotations=collections.defaultdict(list)
            for a in j['annotations']:annotations[a['image_id']].append(a)
            for item in j['images']:
                member=posixpath.normpath(posixpath.join(posixpath.dirname(n),item['file_name']));encoded=z.read(member);im=cv2.imdecode(np.frombuffer(encoded,np.uint8),0)
                assert im.shape==(item['height'],item['width']);boxes=[];signature=[]
                for a in annotations[item['id']]:
                    assert a['category_id']==0 and not a.get('iscrowd',0);x,y,w,h=map(float,a['bbox']);assert w>0 and h>0 and x>=0 and y>=0 and x+w<=item['width'] and y+h<=item['height']
                    signature.append([x,y,w,h]);boxes.append([0,(x+w/2)/item['width'],(y+h/2)/item['height'],w/item['width'],h/item['height']])
                digest=sha(im.tobytes());rid=str(100000+len(rows));row={'id':rid,'image':str(output/f'{rid}.jpg'),'source_member':member,'source_annotation':n,'original_id':item['id'],'source_split':'train' if 'instances_train' in n else 'val','scene_category':n.split('/')[1],'width':item['width'],'height':item['height'],'boxes':boxes,'native_boxes':sorted(signature),'decoded_sha256':digest,'image_sha256':sha(encoded)}
                rows.append(row);grouped[digest].append(row)
                if digest not in pixels:pixels[digest]=(im,encoded)
    conflicts=[g for g in grouped.values() if len({json.dumps(r['native_boxes']) for r in g})>1];bad={r['decoded_sha256'] for g in conflicts for r in g};eligible=[]
    for digest,g in grouped.items():
        if digest in bad:continue
        row=g[0];row['duplicate_ids']=[r['id'] for r in g];Path(row['image']).write_bytes(pixels[digest][1]);eligible.append(row)
    sirst=json.loads((R/'data/splits/sirst_uavb/manifest.json').read_text())['records'];sunique={};features=[];thumbs=[]
    for r in sirst:
        im=cv2.imread(str(R/r['image']),0);digest=sha(im.tobytes())
        if digest not in sunique:
            ph,th=fingerprint(im);features.append(ph);thumbs.append(th);sunique[digest]=r
    sr=list(sunique.values());features=np.array(features);thumbs=np.array(thumbs);all_exact=set(sunique)
    bg=json.loads((E/'data/background_catalog.json').read_text());bgs=collections.defaultdict(list)
    for r in bg:bgs[r['background_group']].append(r)
    reps=[]
    for g in bgs.values():
        g.sort(key=lambda r:int(r['id']));reps.extend(g[i] for i in np.linspace(0,len(g)-1,min(4,len(g)),dtype=int))
    represented={i for r in bg for i in r['original_duplicate_ids']};reps.extend(r for r in sirst if r['id'] not in represented)
    orb=cv2.ORB_create(nfeatures=600);rf=[]
    for r in reps:
        im=cv2.imread(str(R/r['image']),0);k,d=orb.detectAndCompute(im,None);rf.append((r,np.float32([p.pt for p in k]),d))
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING);cross=[];clean=[]
    for i,r in enumerate(eligible):
        im=pixels[r['decoded_sha256']][0];ph,th=fingerprint(im);matches=[]
        if r['decoded_sha256'] in all_exact:matches.append({'type':'exact','sirst_id':sunique[r['decoded_sha256']]['id']})
        ds=np.count_nonzero(features!=ph,axis=1);js=np.flatnonzero(ds<=8)
        for j in js:
            mae=float(np.abs(thumbs[j]-th).mean())
            if mae<=4:matches.append({'type':'perceptual','sirst_id':sr[j]['id'],'hamming':int(ds[j]),'mae':mae})
        k,d=orb.detectAndCompute(im,None);a=np.float32([p.pt for p in k])
        if d is not None and len(d)>=18:
            for other,b,db in rf:
                if db is None or len(db)<18:continue
                good=[m for pair in matcher.knnMatch(d,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.70*n.distance]
                if len(good)<18:continue
                x=np.float32([a[m.queryIdx] for m in good]);y=np.float32([b[m.trainIdx] for m in good]);M,mask=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1000,confidence=.995,refineIters=10)
                if M is None:continue
                mask=mask.ravel().astype(bool);nin=int(mask.sum());ratio=nin/len(good)
                if nin<18 or ratio<.6:continue
                spans=[float(np.prod(np.ptp(v[mask],axis=0))/(640*512)) for v in [x,y]];scale=math.hypot(M[0,0],M[1,0]);angle=abs(math.degrees(math.atan2(M[1,0],M[0,0])))
                if min(spans)>=.1 and .6<=scale<=1.7 and angle<=35:matches.append({'type':'geometry','sirst_id':other['id'],'inliers':nin,'ratio':ratio,'span':spans,'scale':scale,'angle':angle})
        r['cross_matches']=matches
        if matches:cross.append({'id':r['id'],'matches':matches})
        else:clean.append(r)
        if i%200==0:print(json.dumps({'processed':i,'eligible':len(eligible),'cross_linked':len(cross),'seconds':time.monotonic()-started}),flush=True)
    save(E/'data/sidd_all_records.json',rows);save(E/'data/sidd_unique_records.json',eligible);save(E/'data/sidd_clean_records.json',clean)
    result={'protocol':cfg,'original_images':len(rows),'unique_pixels':len(grouped),'duplicate_excess':len(rows)-len(grouped),'conflicting_groups':[[r['id'] for r in g] for g in conflicts],'eligible_unique':len(eligible),'cross_linked_unique':len(cross),'clean_unique':len(clean),'cross_links':cross,'geometric_representatives':[r['id'] for r in reps],'scene_counts':dict(collections.Counter(r['scene_category'] for r in clean)),'tiny_uav_boxes':sum(b[3]*r['width']*b[4]*r['height']<=16 for r in clean for b in r['boxes']),'seconds':time.monotonic()-started}
    save(E/'analysis/sidd_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['cross_links','conflicting_groups','geometric_representatives','protocol']}),flush=True)
if __name__=='__main__':main()
