"""Only development evaluation of the preregistered100-epoch budget check."""
from pathlib import Path
import os,sys,json,gc,hashlib
E=Path(__file__).resolve().parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(E/'logs/ultralytics');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
import torch
from detector_variants import load_detector
from scripts.evaluate_sirst_protocol import evaluate_model
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    index=json.loads((E/'results/long_schedule_weights_frozen.json').read_text());assert len(index['runs'])==2 and index['protocol_sha256']==sha(E/'configs/long_schedule.json');records=json.loads((E/'data/staged_manifest.json').read_text())['dev'];out=E/'results/long_dev';out.mkdir(exist_ok=False);results=[]
    for j in index['runs']:
        assert sha(R/j['checkpoint'])==j['checkpoint_sha256']
        for epoch in [100]:
            path=(R/j['checkpoint']).with_name(f'epoch{epoch}.pt');model,state=load_detector(path,'cuda');del state;gc.collect();name=f'{j["variant"]}_s17_e{epoch}'
            result=evaluate_model(model,out/name,records=records,compress=True);result.update(split='study_long_development',variant=j['variant'],seed=17,epoch=epoch,schedule_epochs=100,checkpoint_sha256=sha(path))
            (out/name/'metrics.json').write_text(json.dumps(result,indent=2));results.append(result);(out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps({'run':name,'AP':result['primary']}),flush=True);del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
