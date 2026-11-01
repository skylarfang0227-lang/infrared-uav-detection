"""COCO matching cache for paired cluster resampling, without repeated IoU work."""
import contextlib,io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def cache_matches(records,predictions):
    images=[];ann=[]
    for r in records:
        images.append({'id':int(r['id']),'width':r['width'],'height':r['height']})
        for cls,x,y,w,h in r['boxes']:
            if cls!=0:continue
            x,y,w,h=x*r['width'],y*r['height'],w*r['width'],h*r['height']
            ann.append({'id':len(ann)+1,'image_id':int(r['id']),'category_id':1,'bbox':[x-w/2,y-h/2,w,h],'area':w*h,'iscrowd':0})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset={'info':{},'images':images,'annotations':ann,'categories':[{'id':1,'name':'UAV'}]};gt.createIndex()
        pred=[p for p in predictions if p['category_id']==1]
        if pred:dt=gt.loadRes(pred)
        else:dt=COCO();dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.maxDets=[300];ev.params.areaRng=[[0.,1e10]];ev.params.areaRngLbl=['all'];ev.evaluate();ev.accumulate()
    expected=ev.eval['precision'];expected=float(expected[expected>=0].mean())
    return {int(e['image_id']):e for e in ev.evalImgs if e is not None},expected
def ap(cache,image_ids):
    entries=[cache[int(i)] for i in image_ids if int(i) in cache]
    if not entries:return None
    n=sum(np.count_nonzero(e['gtIgnore']==0) for e in entries)
    if not n:return None
    scores=np.concatenate([e['dtScores'][:300] for e in entries]);order=np.argsort(-scores,kind='mergesort')
    matches=np.concatenate([e['dtMatches'][:,:300] for e in entries],axis=1)[:,order]
    ignored=np.concatenate([e['dtIgnore'][:,:300] for e in entries],axis=1)[:,order]
    tp=np.cumsum((matches>0)&~ignored,axis=1,dtype=float);fp=np.cumsum((matches==0)&~ignored,axis=1,dtype=float)
    if not scores.size:return 0.
    recall=tp/n;precision=tp/(tp+fp+np.spacing(1));precision=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1];out=np.zeros((10,101))
    for t in range(10):
        inds=np.searchsorted(recall[t],np.linspace(0,1,101),side='left');good=inds<len(scores);out[t,good]=precision[t,inds[good]]
    return float(out.mean())
