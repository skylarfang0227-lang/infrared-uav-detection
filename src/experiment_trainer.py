"""Separate protocol-aware runner; existing trainer and experiment files untouched."""
from pathlib import Path
import sys,copy,json,hashlib,random
import numpy as np
E=Path(__file__).resolve().parents[1];R=E
sys.path.insert(0,str(R));sys.path.insert(0,str(E/'src'))
import torch,yaml
from src.msda_training import MSDATrainer,capture_state,restore_state,atomic_checkpoint,sha
from detector_variants import DetectorModel

DEPENDENCIES=[R/'src/models/msda_irstd.py',R/'src/msda_training.py',R/'src/subpixel_logits.py',R/'src/multi_phase_stem.py',R/'scripts/evaluate_sirst_protocol.py',R/'configs/models/msda_irstd_spec.json',E/'src/detector_variants.py',E/'src/experiment_trainer.py',E/'scripts/train.py']
def source_hashes():return {str(p.relative_to(R)).replace('\\','/'):sha(p) for p in DEPENDENCIES}

class ExperimentTrainer(MSDATrainer):
    def __init__(self,variant,protocol_file,overrides):
        self.variant=variant;self.protocol_file=Path(protocol_file).resolve()
        self.extension_protocol=json.loads(self.protocol_file.read_text(encoding='utf-8'))
        super().__init__('formal',overrides)
    def epoch_loader(self,dataset,batch,epoch):
        # With workers=0, augmentation uses the parent RNG. Reset independently
        # of architecture construction so every matched run sees the same data.
        seed=self.args.seed+1000003*epoch
        random.seed(seed);np.random.seed(seed % 2**32);torch.manual_seed(seed)
        return super().epoch_loader(dataset,batch,epoch)
    def get_dataset(self):
        data=yaml.safe_load(Path(self.args.data).read_text(encoding='utf-8'))
        if set(data)!={'train','val','names'} or data['names']!={0:'UAV',1:'Bird'} or data['train']!=data['val']:raise ValueError('Train-only data config required')
        members=Path(data['train']).resolve()
        if not members.is_relative_to(E/'data'):raise ValueError('Dataset split outside extension')
        permitted=json.loads((E/'data/staged_manifest.json').read_text(encoding='utf-8'))['train']
        by_path={Path(r['image']).resolve():r for r in permitted}
        for line in members.read_text().splitlines():
            p=Path(line).resolve();row=by_path[p]
            if sha(p)!=row['image_sha256'] or sha(row['label'])!=row['label_sha256']:raise ValueError('Data changed')
        data.update(nc=2,channels=3,yaml_file=str(self.args.data));return data
    def get_model(self,cfg=None,weights=None,verbose=True):
        if weights is not None or self.args.pretrained:raise ValueError('Scratch required')
        return DetectorModel(self.variant)
    def save_model(self):
        state=capture_state(self.model,self.optimizer,self.scaler,self.ema,self.scheduler,
          next_epoch=self.next_epoch,last_opt_step=self.last_opt_step,train_args=vars(self.args),phase=self.phase,history=self.history,
          extension_variant=self.variant,implementation_sha256=source_hashes(),train_split_sha256=sha(self.data['train']),
          protocol_sha256=sha(self.protocol_file),protocol_file=str(self.protocol_file.relative_to(R)))
        atomic_checkpoint(state,self.last)
        if self.next_epoch in self.extension_protocol.get('save_epochs',[self.args.epochs]):atomic_checkpoint(state,self.wdir/f'epoch{self.next_epoch}.pt')
    def resume_local(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['extension_variant']!=self.variant or state['protocol_sha256']!=sha(self.protocol_file) or state['train_split_sha256']!=sha(self.data['train']):raise ValueError('Resume identity mismatch')
        if state['implementation_sha256']!=source_hashes():raise ValueError('Implementation changed')
        ignored={'project','name','save_dir','exist_ok','model','data'}
        for k,v in state['train_args'].items():
            if k not in ignored and getattr(self.args,k)!=v:raise ValueError('Resume argument changed: '+k)
        restore_state(state,self.model,self.optimizer,self.scaler,self.ema,self.scheduler)
        self.next_epoch,self.last_opt_step=state['next_epoch'],state['last_opt_step'];self.history=state['history']

def overrides(protocol,run,seed,members):
    original=json.loads((R/'configs/protocols/sirst_msda_v1.json').read_text())['training']
    keys=('optimizer','lr0','lrf','cos_lr','momentum','weight_decay','nbs','warmup_epochs','warmup_momentum','warmup_bias_lr','amp','deterministic','cache','rect','multi_scale','mosaic','close_mosaic','scale','translate','fliplr','flipud','degrees','shear','perspective','mixup','copy_paste','hsv_h','hsv_s','hsv_v')
    args={k:original[k] for k in keys};args.update(protocol.get('training_overrides',{}))
    data=run/'data.yaml';data.write_text(yaml.safe_dump({'train':str(members),'val':str(members),'names':{0:'UAV',1:'Bird'}}),encoding='utf-8')
    args.update(model=str(R/'configs/models/yolo11n-p2.yaml'),data=str(data),device=0,batch=protocol['batch'],epochs=protocol['epochs'],workers=protocol['workers'],seed=seed,imgsz=640,pretrained=False,patience=0,val=False,plots=False,save=True,save_period=protocol['epochs'],box=7.5,cls=.5,dfl=1.5,project=str(run),name='train',exist_ok=False)
    return args
