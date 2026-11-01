"""Separate single-image FP32 inference-memory probe; no timing under training."""
from pathlib import Path
import os,sys,json,gc,hashlib
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.path.insert(0,str(R))
import torch
from src.msda_training import load_model
from src.msda_subpixel_mpm import load_subpixel_mpm_model
from scripts.evaluate_sirst_protocol import image_tensor
from ultralytics.utils import ops
@torch.inference_mode()
def main():
    destination=E/'results/efficiency/single_input_memory.json'
    if destination.exists():raise FileExistsError('Preserve the previous memory measurement.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    row=min((r for r in json.loads((R/'data/splits/sirst_uavb/manifest.json').read_text())['records'] if r['split']=='train'),key=lambda r:int(r['id']));runs=json.loads((R/'experiments/original_split/frozen/weights_frozen.json').read_text())['runs'];results=[]
    for j in runs:
        assert hashlib.sha256((R/j['checkpoint']).read_bytes()).hexdigest()==j['checkpoint_sha256']
        gc.collect();torch.cuda.empty_cache();model,state=(load_model if j['method']=='reference' else load_subpixel_mpm_model)(R/j['checkpoint'],'cuda');del state;x=image_tensor(row,'cuda')
        for _ in range(10):
            pred=model(x);boxes=ops.non_max_suppression(pred,conf_thres=.001,iou_thres=.7,multi_label=False,agnostic=False,max_det=300,nc=2,max_time_img=10.)[0];del pred,boxes
        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        pred=model(x);boxes=ops.non_max_suppression(pred,conf_thres=.001,iou_thres=.7,multi_label=False,agnostic=False,max_det=300,nc=2,max_time_img=10.)[0];torch.cuda.synchronize()
        results.append({'method':j['method'],'seed':j['seed'],'checkpoint_sha256':j['checkpoint_sha256'],'resident_before_forward_mib':baseline/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'incremental_peak_mib':(torch.cuda.max_memory_allocated()-baseline)/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20});del pred,boxes,x,model;gc.collect();torch.cuda.empty_cache()
    output={'protocol':'Batch1 FP32 unfused, one640 input;10warmups, cache emptied before measurement; forward+decode+NMS; includes live model, grid caches and input, excludes host/process-driver memory. Same image and all6weights.','runs':results}
    destination.write_text(json.dumps(output,indent=2));print(results)
if __name__=='__main__':main()
