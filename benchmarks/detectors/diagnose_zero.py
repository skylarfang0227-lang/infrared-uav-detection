"""Post-assessment correctness check; no training, tuning or checkpoint selection."""
from pathlib import Path
import json,math,gc
from comparator import B,E,R,torch,load,sha
from ultralytics.nn.tasks import DetectionModel
from scripts.evaluate_sirst_protocol import image_tensor
def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    p=B/'runs/yolo11n/train/weights/last.pt';model,state=load(p,'cuda')
    native=DetectionModel(str(B/'configs/yolo11n.yaml'),ch=3,nc=2,verbose=False)
    native.load_state_dict(state['ema'],strict=True);native=native.to('cuda').float().eval()
    assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
    assert all(torch.isfinite(v).all() for v in state['ema'].values() if v.is_floating_point())
    records=json.loads((E/'data/staged_manifest.json').read_text());stats={}
    with torch.inference_mode():
        for label,rows in [('training_first16',sorted(records['train'],key=lambda r:int(r['id']))[:16]),('final_all389',records['test'])]:
            maximum=0.;above=0;finite=True;rawmin=math.inf;rawmax=-math.inf;matches=True
            for r in rows:
                x=image_tensor(r,'cuda');a=model(x);b=native(x)
                assert torch.equal(a[0],b[0]);matches=matches and torch.equal(a[0],b[0])
                confidence=a[0][:,4:,:];maximum=max(maximum,float(confidence.max()));above+=int((confidence.max(1).values>=.001).sum())
                finite=finite and bool(torch.isfinite(a[0]).all())
                for layer in a[1]:
                    logits=layer[:,64:];rawmin=min(rawmin,float(logits.min()));rawmax=max(rawmax,float(logits.max()))
            stats[label]={'images':len(rows),'native_loader_outputs_exact':matches,'all_outputs_finite':finite,'max_class_confidence':maximum,'candidates_at_frozen_confidence':above,'raw_class_logit_min':rawmin,'raw_class_logit_max':rawmax}
    assert stats['final_all389']['candidates_at_frozen_confidence']==0
    # Rule out an EMA-only loading issue on the same fixed training-image sample.
    native.load_state_dict(state['model'],strict=True);max_live=0.
    with torch.inference_mode():
        for r in sorted(records['train'],key=lambda r:int(r['id']))[:16]:max_live=max(max_live,float(native(image_tensor(r,'cuda'))[0][:,4:,:].max()))
    history=state['history'];selected=[r for r in history if r['epoch'] in [1,2,5,10,25,40,50]]
    result={'scope':'Post-hoc diagnosis of the frozen zero-score YOLO11n run; no retraining, alternate-score selection, confidence change or substitution.','checkpoint_sha256':sha(p),'both_live_and_EMA_weights_finite':True,'native_official_YAML_strict_reload':True,'image_checks':stats,'live_weights_training_first16_max_confidence':max_live,'loss_history_excerpt':selected,'interpretation':'The saved model itself assigns scores below the frozen cutoff before NMS. Native model loading reproduces these outputs exactly. Late training losses collapse to zero. This is a failed optimization outcome under this scratch fixed-budget recipe, not evidence of intrinsic YOLO11n capacity or an identified unique cause.'}
    (B/'audit/zero_score_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
