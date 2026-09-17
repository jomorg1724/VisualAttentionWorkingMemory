"""Warm-started opponent sequence learner: all learned weights trainable.

Only fast/slow traces persist between frames. Checkpointing recomputes the
stateless encoder/projections exactly; it does not truncate temporal gradients.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from PreAttentiveVision.hybrid_models import build_hybrid
from PreAttentiveVision.TemporalIntegration.accumulators import StreamingPAVClassifier

VERSION = 'wm_opponent_all_learned_v1'


class SequenceOpponent(StreamingPAVClassifier):
    def __init__(self, task_classes, activation_checkpoint=True):
        super().__init__(build_hybrid('convnext_se_residual'), task_classes, 'opponent',
                         common_seed=30312, core_seed=30313)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.activation_checkpoint = activation_checkpoint
        self.config.update(implementation=VERSION, encoder_frozen=False,
                           encoder_eval=False, steps_per_pair=None,
                           activation_checkpoint=activation_checkpoint,
                           memory_routes='three fast/slow spatial trace pairs only')

    def train(self, mode=True):
        return nn.Module.train(self, mode)

    def _encode(self, frame):
        return tuple(p(h) for p,h in zip(self.projections, self.encoder((frame-.5)/.5)))

    def _trace_step(self, frame, state=None):
        if frame.ndim != 4 or tuple(frame.shape[1:]) != (3,100,100):
            raise ValueError('Expected current frame[B,3,100,100]')
        if self.training and self.activation_checkpoint and torch.is_grad_enabled():
            # Non-reentrant mode retains parameter gradients even when ordinary
            # image inputs have requires_grad=False (Torch1.13-compatible).
            current = checkpoint(self._encode, frame, use_reentrant=False, preserve_rng_state=True)
        else:
            current = self._encode(frame)
        if state is None:
            updated = tuple((u,u) for u in current)
        else:
            if len(state) != 3:
                raise ValueError('Expected three explicit state pairs')
            updated = tuple((.25*f+.75*u, .75*l+.25*u)
                            for u,(f,l) in zip(current,state))
        return current,updated

    def _emit(self,state):
        result=[]
        for core,(fast,slow) in zip(self.accumulators,state):
            energy,_=core.energy_channels(fast,slow)
            result.append(core.output(torch.cat((.5*(fast+slow),fast-slow,energy),1)))
        return result

    def step(self,frame,state=None):
        current,state=self._trace_step(frame,state)
        return self.readout.encode(current,self._emit(state)),state

    def forward(self,images,task):
        if images.ndim!=5 or tuple(images.shape[2:])!=(3,100,100) or images.shape[1]<1:
            raise ValueError('Expected homogeneous sequence[B,T,3,100,100]')
        state=None
        for t in range(images.shape[1]):
            current,state=self._trace_step(images[:,t],state)
        # Intermediate energy/readout values are unused and never feed state.
        # Compute them at the report only; every physical frame still advances.
        features=self.readout.encode(current,self._emit(state))
        return self.classify(features,task)


def load_parent(parent,task_classes,activation_checkpoint=True):
    if parent['version']!='pav_temporal_v1' or parent['step']!=4032 or parent['config']['model']!='opponent':
        raise ValueError('Expected trusted selected opponent4032 parent')
    model=SequenceOpponent(task_classes,activation_checkpoint)
    model.load_state_dict(parent['model'],strict=True)
    return model
