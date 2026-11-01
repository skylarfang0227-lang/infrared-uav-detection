"""Portable imports, detector geometry, finite gradients and exact reload checks."""
from pathlib import Path
import ast, gc, hashlib, json, os, sys, tempfile
ROOT=Path(__file__).resolve().parents[1];E=ROOT
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(E/'benchmarks/detectors'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'verification/ultralytics')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from detector_variants import DetectorModel,VARIANTS
from comparator import build,sources

def main():
    torch.set_num_threads(2);checks=[]
    for p in ROOT.rglob('*.py'):
        if '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    for name in ['src.msda_training','src.models.msda_irstd','detector_variants','experiment_trainer','comparator']:
        module=__import__(name,fromlist=['*']);assert Path(module.__file__).resolve().is_relative_to(ROOT),module.__file__
    assert sources()
    for key in list(VARIANTS)+['yolov5nu','yolov8n','yolo11n','yolo11n_p2']:
        torch.manual_seed(17)
        model=DetectorModel(key) if key in VARIANTS else build(key)
        expected=[2.,8.,16.,32.] if key in ['head','combined','phase_collapsed'] else ([4.,8.,16.,32.] if key in ['reference','stem','yolo11n_p2'] else [8.,16.,32.])
        assert model.stride.tolist()==expected
        model.train();x=torch.randn(2,3,64,64);raw=model(x)
        loss=sum(z.square().mean() for z in raw);loss.backward()
        params=[p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),key
        count=sum(p.numel() for p in params)
        model.eval()
        with torch.inference_mode():out=model(x)[0]
        assert torch.isfinite(out).all()
        fresh=DetectorModel(key) if key in VARIANTS else build(key)
        fresh.load_state_dict(model.state_dict(),strict=True);fresh.eval()
        with torch.inference_mode():again=fresh(x)[0]
        assert torch.equal(out,again),key
        checks.append({'model':key,'trainable_parameters':count,'total_parameters':sum(p.numel() for p in model.parameters()),'strides':expected,'finite_gradients_all_parameters':True,'exact_state_reload':True})
        print(key,'passed',count,flush=True)
        del model,fresh,params,raw,out,again,loss;gc.collect()
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    (out/'model_checks.json').write_text(json.dumps({'all_pass':True,'checks':checks,'training':'No full training was performed; these are CPU construction, geometry, gradient and reload checks.'},indent=2))
    print('All nine configurations passed.')
if __name__=='__main__':main()
