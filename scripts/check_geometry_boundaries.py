"""Verify geometric counts with the actual float32 assigner and rounding margin."""
from pathlib import Path
import os,json,sys
E=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics')
import torch
from ultralytics.utils.tal import TaskAlignedAssigner
torch.set_num_threads(2)
rows=json.loads((E/'data/staged_manifest.json').read_text())['train'];boxes=[];areas=[]
for r in rows:
    for cls,x,y,w,h in r['boxes']:
        if cls==0:boxes.append([(x-w/2)*640,(y-h/2)*512+64,(x+w/2)*640,(y+h/2)*512+64]);areas.append(w*640*h*512)
b=torch.tensor(boxes,dtype=torch.float32);out={};available={};safe_available={}
for s in [4,2,8,16,32]:
    xy=(torch.arange(640//s,dtype=torch.float32)+.5)*s;yy,xx=torch.meshgrid(xy,xy,indexing='ij');anc=torch.stack([xx.flatten(),yy.flatten()],-1);counts=[];safe=[]
    for chunk in b.split(32):
        counts.append(TaskAlignedAssigner.select_candidates_in_gts(anc,chunk[None]).sum(-1)[0]);safe.append(TaskAlignedAssigner.select_candidates_in_gts(anc,chunk[None],eps=.001).sum(-1)[0])
    counts=torch.cat(counts);safe=torch.cat(safe);tiny=torch.tensor(areas)<=16;available[s]=counts>0;safe_available[s]=safe>0
    out[str(s)]={'actual_assigner_float32_zero':int((counts==0).sum()),'boundary_margin_0.001px_zero':int((safe==0).sum()),'mean_sites':float(counts.mean()),'safe_mean_sites':float(safe.mean()),'tiny_zero':int((counts[tiny]==0).sum()),'tiny_safe_zero':int((safe[tiny]==0).sum())}
joint={}
for name,strides in [('reference',[4,8,16,32]),('head',[2,8,16,32])]:
    ok=torch.stack([available[s] for s in strides]).any(0);safe=torch.stack([safe_available[s] for s in strides]).any(0)
    joint[name]={'zero_all_levels':int((~ok).sum()),'tiny_zero_all_levels':int((~ok[tiny]).sum()),'safe_zero_all_levels':int((~safe).sum()),'safe_tiny_zero_all_levels':int((~safe[tiny]).sum())}
result={'reason':'Six-decimal normalized labels can place edges within0.001 native pixels of nominal integer boundaries. Compare native float32 strict inclusion with a conservative rounding margin. Unaugmented training labels, not actual augmented TAL positive assignments. No AP or model selection.','counts':out,'all_levels':joint}
(E/'analysis/geometry_boundary_sensitivity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
