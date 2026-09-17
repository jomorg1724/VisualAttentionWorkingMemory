"""Scratch arm using pre-output attention context as the terminal comparator."""
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from WorkingMemory.PreUpdateAttention.model import JointAttention
from WorkingMemory.SpatialPriorityReadout.model import (
    SpatialPriorityMemory,
    groups,
    state_sha256,
)

VERSION = "attention_context_comparator_scratch_v1"
ARM = "attention_context_comparator_scratch"


class ContextJointAttention(JointAttention):
    """Original JointAttention with its merged pre-W_O context exposed."""

    def __init__(self):
        super().__init__()
        self.last_weights = None
        self.last_context = None

    def forward(self, field, old, diagnostic=False):
        batch = field.shape[0]
        visual = field.flatten(2).transpose(1, 2)
        memory = old.flatten(2).transpose(1, 2)
        query = self.query(self.query_norm(memory) + self.position + self.source[1])
        raw = torch.cat((visual, memory), 1)
        identities = torch.cat(
            (self.position + self.source[0], self.position + self.source[1]), 0
        )
        key = self.key(self.key_norm(raw) + identities)
        value = self.value(raw)
        query = query.reshape(batch, 169, 2, 32).transpose(1, 2)
        key = key.reshape(batch, 338, 2, 32).transpose(1, 2)
        value = value.reshape(batch, 338, 2, 32).transpose(1, 2)
        logits = query @ key.transpose(-1, -2) / math.sqrt(32)
        bias = (
            self.source_bias.repeat_interleave(169, dim=1)[:, None, :]
            - F.softplus(self.raw_locality)[:, None, None] * self.distance_squared
        )
        weights = (logits + bias[None]).softmax(-1)
        context = (weights @ value).transpose(1, 2).reshape(batch, 169, 64)
        context_field = context.transpose(1, 2).reshape(batch, 64, 13, 13)
        attended = self.output(context).transpose(1, 2).reshape(batch, 64, 13, 13)
        if diagnostic:
            if weights.requires_grad:
                weights.retain_grad()
            self.last_weights = weights
            self.last_context = context_field
            stats = dict(
                memory_attention_mass=float(
                    weights[:, :, :, 169:].detach().sum(-1).mean()
                ),
                locality=F.softplus(self.raw_locality).detach().tolist(),
            )
        else:
            self.last_weights = None
            self.last_context = None
            stats = {}
        return attended, context_field, stats


class AttentionContextComparatorMemory(SpatialPriorityMemory):
    """Control model with C_t defined as merged pre-W_O attention values."""

    def __init__(self, cfg):
        super().__init__(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["model_seed"])
            attention = ContextJointAttention()
        attention.load_state_dict(self.attention.state_dict(), strict=True)
        self.attention = attention
        del self.comparator

    def forward(self, images, task, diagnostic=False):
        traces = ()
        state = None
        records = []
        states = []
        first_field = None
        context = None
        for timestep in range(images.shape[1]):
            if self.training and self.activation_checkpoint and torch.is_grad_enabled():
                result = checkpoint(
                    self._sensory,
                    images[:, timestep],
                    *traces,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                result = self._sensory(images[:, timestep], *traces)
            field, traces = result[0], result[1:]
            old = torch.zeros_like(field) if state is None else state[0]
            attended, context, attention_stats = self.attention(
                field, old, diagnostic
            )
            rates, state, memory_stats = self.memory(
                self.memory_input(attended), state, diagnostic
            )
            if diagnostic:
                if timestep == 0:
                    field.retain_grad()
                    first_field = field
                for value in state:
                    if value.requires_grad:
                        value.retain_grad()
                states.append(state)
                records.append(
                    {
                        **{key: float(value) for key, value in memory_stats.items()},
                        **attention_stats,
                    }
                )
        logits, priority = self.priority_readout(field, rates, context, task)
        if diagnostic:
            return logits, dict(
                states=states,
                first_field=first_field,
                records=records,
                priority=priority,
                terminal_fields=(field, rates, context),
                attention_context=context,
                attention_weights=self.attention.last_weights,
            )
        return logits


def initialize_scratch(arm, cfg):
    if arm != ARM or cfg.get("battery") != "spatial":
        raise ValueError("Expected scratch attention-context comparator arm")
    model = AttentionContextComparatorMemory(cfg)
    if hasattr(model, "comparator"):
        raise RuntimeError("Old learned comparator must be absent")
    if "attention.gamma" in model.state_dict():
        raise RuntimeError("Prospective gamma must be absent")
    optimizer = torch.optim.Adam(groups(model, cfg), eps=cfg["adam_eps"])
    if optimizer.state:
        raise RuntimeError("Scratch optimizer must have no inherited state")
    return model, optimizer, dict(
        version=VERSION,
        initialization_kind="full_model_from_scratch",
        loaded_parent=False,
        loaded_model_tensors=0,
        loaded_optimizer_states=0,
        initial_model_sha256=state_sha256(model),
        seeds=dict(
            global_model=cfg["model_seed"],
            pav_common=30312,
            pav_core=30313,
            recurrent_common=59311,
            recurrent_core=59312,
            spatial_components=31973001,
            spatial_comparison_output=31973002,
            joint_attention=cfg["model_seed"],
            priority_readout=cfg["model_seed"] + 100,
            python_numpy_torch=cfg["model_seed"],
            training_stream=cfg["train_seed"],
            task_scheduler=cfg["scheduler_seed"],
        ),
        prospective_gamma_absent=True,
        comparator=(
            "C_t is concat_heads(sum_j A_t,h,i,j V_t,h,j) before W_O; "
            "the learned Conv([R_t-1,H_t]) comparator is absent."
        ),
        attended_drive="U_t=W_O(C_t), identical JointAttention output computation",
    )
