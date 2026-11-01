"""Conditional parent-increment candidate: subpixel head plus four-phase input stem."""
import torch
from src.msda_subpixel_head import MSDAWithSubpixelHead, SubpixelHeadTrainer
from src.multi_phase_stem import install_mpm_stem
from src.msda_training import ROOT, atomic_checkpoint, capture_state, sha


class MSDAWithSubpixelMPM(MSDAWithSubpixelHead):
    def __init__(self, candidate, spec=None, enabled=None):
        super().__init__(candidate, spec, enabled)
        stem=candidate.get('stem_phase',{})
        if stem.get('enabled',False):
            if stem.get('node')!='stem1':raise ValueError('Only the first stem is supported')
            install_mpm_stem(self)


class SubpixelMPMTrainer(SubpixelHeadTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None or self.args.pretrained:
            raise ValueError('Scratch only')
        return MSDAWithSubpixelMPM(self.candidate, enabled=self.enabled)

    def save_model(self):
        sources = ('src/models/msda_irstd.py', 'src/msda_training.py', 'scripts/evaluate_sirst_protocol.py',
                    'src/subpixel_logits.py', 'src/msda_subpixel_head.py',
                   'src/multi_phase_stem.py', 'src/msda_subpixel_mpm.py',
                   'scripts/train_original_split.py')
        state = capture_state(self.model, self.optimizer, self.scaler, self.ema, self.scheduler,
                              next_epoch=self.next_epoch, last_opt_step=self.last_opt_step,
                              train_args=vars(self.args), phase=self.phase, history=self.history,
                              candidate=self.candidate, implementation_sha256={p: sha(ROOT/p) for p in sources},
                              train_split_sha256=sha(self.data['train']),
                              protocol_sha256=sha(ROOT/'configs/protocols/sirst_msda_v1.json'))
        atomic_checkpoint(state, self.last)
        if self.next_epoch % self.args.save_period == 0:
            atomic_checkpoint(state, self.wdir/f'epoch{self.next_epoch}.pt')


def load_subpixel_mpm_model(path, device='cpu', ema=True):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'msda_full_fp32_v1' or 'stem_phase' not in state.get('candidate', {}):
        raise ValueError('Expected subpixel-MPM checkpoint')
    model = MSDAWithSubpixelMPM(state['candidate'], state['spec'], state['enabled'])
    model.load_state_dict(state['ema' if ema else 'model'], strict=True)
    return model.to(device).float().eval(), state
