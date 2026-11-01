"""Matched frozen A/H inference benchmark; no optimizer or training operations."""
from pathlib import Path
import os,sys,json,time,hashlib,gc,math,subprocess,platform
N=Path(__file__).resolve().parents[1];E=N;R=E;O=N/'analysis'
os.environ['YOLO_CONFIG_DIR']=str(O/'ultralytics');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
import torch,numpy as np,psutil
from detector_variants import load_detector
from scripts.evaluate_sirst_protocol import image_tensor
from ultralytics.utils import ops
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def summary(x):return {'median_ms':float(np.median(x)),'mean_ms':float(np.mean(x)),'p90_ms':float(np.percentile(x,90)),'n':len(x)}
def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();y=fn();torch.cuda.synchronize();return (time.perf_counter()-start)*1000,y
def nms(pred):return ops.non_max_suppression(pred,conf_thres=.001,iou_thres=.7,multi_label=False,agnostic=False,max_det=300,nc=2,max_time_img=10.)
def gpu():return subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,pstate,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
@torch.inference_mode()
def main():
    out=E/'results/cost_grouped';out.mkdir(exist_ok=True);assert not list(out.iterdir());start=time.monotonic()
    ancestors={p.pid for p in psutil.Process().parents()}
    python_processes=[]
    for p in psutil.process_iter(['pid','name']):
        if 'python' in (p.info['name'] or '').lower() and p.pid!=os.getpid() and p.pid not in ancestors:python_processes.append(p.info)
    assert not python_processes,('Other Python processes: do not benchmark concurrently',python_processes)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    rows=sorted(read(E/'data/staged_manifest.json')['train'],key=lambda r:int(r['id']))[:32]
    frozen=read(E/'results/factorial_weights_frozen.json');runs=[]
    meta={'protocol_sha256':sha(E/'configs/diagnostics.json'),'code_sha256':sha(Path(__file__)),'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_before':gpu(),'other_python_processes':python_processes,'image_ids':[r['id'] for r in rows],'manifest_sha256':sha(E/'data/staged_manifest.json')}
    save(out/'environment.json',meta)
    for si,seed in enumerate([17,29,43]):
        for variant in (['reference','head'] if si%2==0 else ['head','reference']):
            j=next(j for j in frozen['runs'] if j['seed']==seed and j['variant']==variant);p=R/j['checkpoint'];assert sha(p)==j['checkpoint_sha256']
            model,state=load_detector(p,'cuda');del state;gc.collect();head=model.model[-1]
            inputs=[image_tensor(r,'cuda') for r in rows];counter={'conv_linear_macs':0};handles=[]
            def hook(m,args,y):
                counter['conv_linear_macs']+=y.numel()*(m.in_features if isinstance(m,torch.nn.Linear) else (m.in_channels//m.groups)*math.prod(m.kernel_size))
            for m in model.modules():
                if isinstance(m,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):handles.append(m.register_forward_hook(hook))
            model(inputs[0])
            for handle in handles:handle.remove()
            del handles,handle,m
            def pipeline(row):
                z=image_tensor(row,'cuda');pred=model(z);b=nms(pred)[0];ops.scale_boxes(z.shape[2:],b[:,:4],(row['height'],row['width']));return b.cpu()
            for i in range(20):pipeline(rows[i%32])
            torch.cuda.synchronize();durations={k:[] for k in ['raw_forward','decode','nms','end_to_end']};blocks=[]
            for block in range(3):
                current={k:[] for k in durations}
                for i in range(64):
                    x=inputs[i%32]
                    # Only select the Detect module's raw-output branch; model/BN remain eval and autograd is disabled.
                    head.training=True
                    try:dt,raw=timed(lambda:model(x))
                    finally:head.training=False
                    current['raw_forward'].append(dt)
                    dt,decoded=timed(lambda:head._inference(raw));current['decode'].append(dt)
                    dt,boxes=timed(lambda:nms(decoded.clone()));current['nms'].append(dt)
                    dt,host=timed(lambda:pipeline(rows[i%32]));current['end_to_end'].append(dt)
                    del raw,decoded,boxes,host
                for k in durations:durations[k]+=current[k]
                blocks.append({k:summary(v) for k,v in current.items()})
                print(json.dumps({'variant':variant,'seed':seed,'block':block+1,'median_e2e_ms':blocks[-1]['end_to_end']['median_ms']}),flush=True)
            del inputs,x,pipeline;gc.collect();torch.cuda.empty_cache()
            x=image_tensor(rows[0],'cuda')
            for i in range(10):
                pred=model(x);boxes=nms(pred);del pred,boxes
            memory=[]
            for i in range(3):
                torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                pred=model(x);boxes=nms(pred);torch.cuda.synchronize()
                memory.append({'resident_mib':base/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'incremental_peak_mib':(torch.cuda.max_memory_allocated()-base)/2**20})
                del pred,boxes
            assert sha(p)==j['checkpoint_sha256']
            result={'variant':variant,'seed':seed,'checkpoint':j['checkpoint'],'checkpoint_sha256':sha(p),'parameters':sum(p.numel() for p in model.parameters()),**counter,
                    'timings':{k:summary(v) for k,v in durations.items()},'blocks':blocks,'raw_ms':durations,'memory':memory,'peak_allocated_mib':float(np.median([m['peak_allocated_mib'] for m in memory])),'gpu_after':gpu()}
            save(out/f'{variant}_s{seed}.json',result);runs.append(result)
            del x,head,model;gc.collect();torch.cuda.empty_cache()
    aggregates={v:{**{k:float(np.mean([r['timings'][k]['median_ms'] for r in runs if r['variant']==v])) for k in durations},
                   'peak_allocated_mib':float(np.mean([r['peak_allocated_mib'] for r in runs if r['variant']==v])),
                   'parameters':next(r['parameters'] for r in runs if r['variant']==v),'conv_linear_macs':next(r['conv_linear_macs'] for r in runs if r['variant']==v)} for v in ['reference','head']}
    paired=[{'seed':s,**{k:next(r['timings'][k]['median_ms'] for r in runs if r['variant']=='head' and r['seed']==s)-next(r['timings'][k]['median_ms'] for r in runs if r['variant']=='reference' and r['seed']==s) for k in durations}} for s in [17,29,43]]
    save(out/'summary.json',{'environment':meta,'runs':runs,'aggregates':aggregates,'paired_ms':paired,'seconds':time.monotonic()-start,'gpu_after':gpu()})
    print(json.dumps({'aggregates':aggregates,'paired_ms':paired},indent=2),flush=True)
if __name__=='__main__':main()
