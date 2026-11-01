"""Native complete detector definitions on the unchanged matched training loop."""
from pathlib import Path
import os,sys,json,copy
B=Path(__file__).resolve().parent;E=B.parents[1];R=E
os.environ['YOLO_CONFIG_DIR']=str(B/'logs/ultralytics');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(E/'src'));sys.path.insert(0,str(R))
import torch,yaml,ultralytics
from ultralytics.nn.tasks import DetectionModel
from experiment_trainer import ExperimentTrainer,overrides
from src.msda_training import capture_state,restore_state,atomic_checkpoint,sha
def protocol():return json.loads((B/'protocol.json').read_text())
def sources():
    package=Path(ultralytics.__file__).parent
    paths=[B/'comparator.py',B/'train_one.py',E/'src/experiment_trainer.py',R/'src/msda_training.py',R/'scripts/evaluate_sirst_protocol.py',R/'configs/protocols/sirst_msda_v1.json']
    paths += [package/p for p in ['nn/tasks.py','nn/modules/head.py','utils/loss.py','utils/tal.py','engine/trainer.py','data/augment.py']]
    return {(p.relative_to(R).as_posix() if p.is_relative_to(R) else 'dependency/ultralytics/'+p.relative_to(package).as_posix()):sha(p) for p in paths}
def build(key):
    item=next(x for x in protocol()['models'] if x['id']==key);path=B/item['config'];assert sha(path)==item['sha256']
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'));cfg.update(scale='n',nc=2)
    model=DetectionModel(copy.deepcopy(cfg),ch=3,nc=2,verbose=False)
    model.spec={'id':key,'config':cfg,'source_sha256':sha(path)};model.enabled={'complete_native_detector':True}
    assert not model.end2end and model.model[-1].nc==2
    assert model.stride.tolist()==([4.,8.,16.,32.] if key=='yolo11n_p2' else [8.,16.,32.])
    return model
class ComparisonTrainer(ExperimentTrainer):
    def __init__(self,key,args):super().__init__(key,B/'protocol.json',args)
    def get_model(self,cfg=None,weights=None,verbose=True):
        if weights is not None or self.args.pretrained:raise ValueError('Scratch only')
        return build(self.variant)
    def save_model(self):
        state=capture_state(self.model,self.optimizer,self.scaler,self.ema,self.scheduler,next_epoch=self.next_epoch,last_opt_step=self.last_opt_step,train_args=vars(self.args),phase=self.phase,history=self.history,comparison_variant=self.variant,implementation_sha256=sources(),train_split_sha256=sha(self.data['train']),protocol_sha256=sha(B/'protocol.json'))
        state['format']='comparison_full_fp32_v1'
        atomic_checkpoint(state,self.last)
        if self.next_epoch==50:atomic_checkpoint(state,self.wdir/'epoch50.pt')
    def resume_local(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['format']=='comparison_full_fp32_v1' and state['comparison_variant']==self.variant
        assert state['implementation_sha256']==sources() and state['protocol_sha256']==sha(B/'protocol.json') and state['train_split_sha256']==sha(self.data['train'])
        for k,v in state['train_args'].items():
            if k not in {'project','name','save_dir','exist_ok','model','data'}:assert getattr(self.args,k)==v,k
        restore_state(state,self.model,self.optimizer,self.scaler,self.ema,self.scheduler)
        self.next_epoch,self.last_opt_step=state['next_epoch'],state['last_opt_step'];self.history=state['history']
def load(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['format']=='comparison_full_fp32_v1'
    current=state['implementation_sha256']==sources() and state['protocol_sha256']==sha(B/'protocol.json')
    # Relocation changes source-path fingerprints, never the stored tensor values.
    # Only explicitly indexed, byte-identical historical weights may bypass the
    # current-source identity comparison for inference. Resume remains strict.
    archived={x['sha256'] for x in json.loads((R/'configs/checkpoints.json').read_text())['files']}
    assert current or sha(path) in archived, 'Unrecognized checkpoint implementation'
    model=build(state['comparison_variant']);assert model.spec==state['spec'];model.load_state_dict(state['ema'],strict=True)
    return model.to(device).float().eval(),state
