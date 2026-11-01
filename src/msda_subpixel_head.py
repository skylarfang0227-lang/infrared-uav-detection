"""Subpixel first-head logits; complete-reference feature modules retained."""
import copy
import torch
from src.subpixel_logits import install_subpixel_head
from src.models.msda_irstd import MSDAIRSTD
from src.msda_training import ROOT,MSDATrainer,atomic_checkpoint,capture_state,sha


class MSDAWithSubpixelHead(MSDAIRSTD):
    def __init__(self,candidate,spec=None,enabled=None):
        super().__init__(spec,enabled)
        if candidate.get('enabled',False):
            if candidate['node']!='Detect.P2_logits' or candidate.get('scale')!=2:
                raise ValueError('Only first-head logit upscaling by2 is supported')
            install_subpixel_head(self)



class SubpixelHeadTrainer(MSDATrainer):
    def __init__(self,phase,overrides,candidate):
        self.candidate=copy.deepcopy(candidate)
        super().__init__(phase,overrides)

    def resume_local(self,path):
        state=torch.load(path,map_location="cpu",weights_only=False)
        if state.get("candidate")!=self.candidate:raise ValueError("Candidate configuration changed across resume")
        super().resume_local(path)

    def get_model(self,cfg=None,weights=None,verbose=True):
        if weights is not None or self.args.pretrained:raise ValueError('Scratch only')
        return MSDAWithSubpixelHead(self.candidate,enabled=self.enabled)

    def setup(self):MSDATrainer.setup(self)

    def save_model(self):
        sources=('src/models/msda_irstd.py','src/msda_training.py','scripts/evaluate_sirst_protocol.py',
                 'src/subpixel_logits.py','src/msda_subpixel_head.py',
                 'scripts/train_original_split.py')
        state=capture_state(self.model,self.optimizer,self.scaler,self.ema,self.scheduler,
                            next_epoch=self.next_epoch,last_opt_step=self.last_opt_step,
                            train_args=vars(self.args),phase=self.phase,history=self.history,
                            candidate=self.candidate,implementation_sha256={p:sha(ROOT/p) for p in sources},
                            train_split_sha256=sha(self.data['train']),
                            protocol_sha256=sha(ROOT/'configs/protocols/sirst_msda_v1.json'))
        atomic_checkpoint(state,self.last)
        if self.next_epoch%self.args.save_period==0:atomic_checkpoint(state,self.wdir/f'epoch{self.next_epoch}.pt')


def load_subpixel_head_model(path,device='cpu',ema=True):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='msda_full_fp32_v1' or 'candidate' not in state:
        raise ValueError('Expected local subpixel-head checkpoint')
    model=MSDAWithSubpixelHead(state['candidate'],state['spec'],state['enabled'])
    model.load_state_dict(state['ema' if ema else 'model'],strict=True)
    return model.to(device).float().eval(),state
