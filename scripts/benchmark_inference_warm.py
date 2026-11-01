"""Synchronized matched inference timing; no training and no score selection."""
from pathlib import Path
import os,sys,json,time,hashlib,gc,math
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.path.insert(0,str(R))
import torch,numpy as np
from src.msda_training import load_model
from src.msda_subpixel_mpm import load_subpixel_mpm_model
from scripts.evaluate_sirst_protocol import image_tensor
from ultralytics.utils import ops
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def summarize(a):return {'median_ms':float(np.median(a)),'mean_ms':float(np.mean(a)),'p90_ms':float(np.percentile(a,90)),'n':len(a)}
def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize();return (time.perf_counter()-t)*1000,value
@torch.inference_mode()
def main():
 cfg=json.loads((E/'configs/efficiency_warm.json').read_text());out=E/'results/efficiency_warm';out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 rows=json.loads((R/'data/splits/sirst_uavb/manifest.json').read_text())['records'];rows=sorted([r for r in rows if r['split']=='train'],key=lambda r:int(r['id']))[:32]
 inputs=[image_tensor(r,'cuda') for r in rows]
 campaign=json.loads((R/'experiments/original_split/frozen/weights_frozen.json').read_text())['runs'];results=[]
 for si,seed in enumerate([17,29,43]):
  for method in (['reference','candidate'] if si%2==0 else ['candidate','reference']):
   item=next(r for r in campaign if r['seed']==seed and r['method']==method);p=R/item['checkpoint'];assert sha(p)==item['checkpoint_sha256']
   torch.cuda.empty_cache();model,state=(load_model if method=='reference' else load_subpixel_mpm_model)(p,'cuda');del state;gc.collect();head=model.model[-1]
   counter={'conv_linear_macs':0};handles=[]
   def hook(m,args,y):
    if isinstance(m,torch.nn.Linear):counter['conv_linear_macs']+=y.numel()*m.in_features
    else:counter['conv_linear_macs']+=y.numel()*(m.in_channels//m.groups)*math.prod(m.kernel_size)
   for m in model.modules():
    if isinstance(m,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):handles.append(m.register_forward_hook(hook))
   model(inputs[0]);[h.remove() for h in handles]
   for i in range(cfg['warmup_iterations']):
    z=image_tensor(rows[i%32],'cuda');p0=model(z);b0=ops.non_max_suppression(p0,conf_thres=.001,iou_thres=.7,multi_label=False,agnostic=False,max_det=300,nc=2,max_time_img=10.)[0]
    ops.scale_boxes(z.shape[2:],b0[:,:4],(rows[i%32]['height'],rows[i%32]['width']));b0.cpu();del z,p0,b0
   torch.cuda.synchronize()
   durations={k:[] for k in ['raw_forward','decode','nms','end_to_end']};blocks=[]
   torch.cuda.reset_peak_memory_stats()
   for b in range(cfg['blocks']):
    current={k:[] for k in durations}
    for j in range(cfg['iterations_per_block']):
     i=j%32;x=inputs[i]
     head.training=True;dt,raw=timed(lambda:model(x));current['raw_forward'].append(dt)
     head.training=False;dt,decoded=timed(lambda:head._inference(raw));current['decode'].append(dt)
     dt,boxes=timed(lambda:ops.non_max_suppression(decoded.clone(),conf_thres=.001,iou_thres=.7,multi_label=False,agnostic=False,max_det=300,nc=2,max_time_img=10.));current['nms'].append(dt)
     def pipeline():
      z=image_tensor(rows[i],'cuda');pred=model(z);bb=ops.non_max_suppression(pred,conf_thres=.001,iou_thres=.7,multi_label=False,agnostic=False,max_det=300,nc=2,max_time_img=10.)[0]
      ops.scale_boxes(z.shape[2:],bb[:,:4],(rows[i]['height'],rows[i]['width']));return bb.cpu()
     dt,_=timed(pipeline);current['end_to_end'].append(dt)
     del raw,decoded,boxes
    for k in durations:durations[k].extend(current[k])
    blocks.append({k:summarize(v) for k,v in current.items()})
    print(json.dumps({'method':method,'seed':seed,'block':b+1,'median_e2e_ms':blocks[-1]['end_to_end']['median_ms']}),flush=True)
   result={'method':method,'seed':seed,'checkpoint':str(p.relative_to(R)),'checkpoint_sha256':sha(p),'parameters':sum(p.numel() for p in model.parameters()),**counter,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'timings':{k:summarize(v) for k,v in durations.items()},'blocks':blocks,'raw_ms':durations}
   save(out/f'{method}_s{seed}.json',result);results.append(result);del model;gc.collect();torch.cuda.empty_cache()
 summary={'protocol':cfg,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'image_ids':[r['id'] for r in rows],'runs':results,'memory_caveat':'Peak includes32 preloaded input tensors and staged raw/decoded buffers, identically for both methods; process allocator reserved memory is not solely model memory.'}
 save(out/'summary.json',summary);print('Benchmark complete',flush=True)
if __name__=='__main__':main()
