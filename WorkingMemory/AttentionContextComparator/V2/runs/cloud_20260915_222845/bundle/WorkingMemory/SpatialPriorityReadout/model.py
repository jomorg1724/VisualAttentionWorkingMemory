"""Fully scratch-trained spatial model with a priority-map terminal readout."""
import hashlib
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from WorkingMemory.PreUpdateAttention.model import (
    AttentionMemory,
)

VERSION = "spatial_priority_readout_scratch_v2"
ARM = "spatial_priority_readout_scratch"


class SpatialPriorityHead(nn.Module):
    """Task-conditioned spatial selection over local class evidence."""

    def __init__(self, task_classes, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 100)
            self.shared = nn.Sequential(
                nn.Conv2d(192, 96, 1, bias=False),
                nn.GroupNorm(8, 96),
                nn.SiLU(),
                nn.Conv2d(96, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
            )
            self.selection = nn.ModuleDict(
                {task: nn.Conv2d(64, 1, 1) for task in task_classes}
            )
            self.evidence = nn.ModuleDict(
                {
                    task: nn.Conv2d(64, classes, 1)
                    for task, classes in task_classes.items()
                }
            )
            for layer in self.selection.values():
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.capture = False
        self.last_priority = None
        self.last_evidence = None

    def forward(self, field, rates, comparison, task):
        if task not in self.selection:
            raise KeyError("Unknown priority task " + task)
        if field.shape != rates.shape or field.shape != comparison.shape:
            raise ValueError("H_T, R_T and C_T must share [B,64,13,13]")
        if tuple(field.shape[1:]) != (64, 13, 13):
            raise ValueError("Expected aligned 64x13x13 terminal fields")
        features = self.shared(torch.cat((field, rates, comparison), 1))
        selection_logits = self.selection[task](features).flatten(1)
        priority = selection_logits.softmax(1).reshape(-1, 1, 13, 13)
        evidence = self.evidence[task](features)
        logits = (priority * evidence).sum((2, 3))
        if self.capture:
            self.last_priority = priority[:, 0].detach()
            self.last_evidence = evidence.detach()
        return logits, priority


class SpatialPriorityMemory(AttentionMemory):
    """Original attention/E-I/comparator classes with a spatial terminal readout."""

    def __init__(self, cfg):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["model_seed"])
            super().__init__(
                seed=cfg["model_seed"],
                activation_checkpoint=cfg["activation_checkpoint"],
            )
            del self.readout.trunk
            del self.readout.heads
            del self.memory_output
            del self.comparison_output
            self.priority_readout = SpatialPriorityHead(
                cfg["task_classes"], cfg["model_seed"]
            )

    def forward(self, images, task, diagnostic=False):
        traces = ()
        state = None
        records = []
        states = []
        first_field = None
        local = None
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
            attended, attention_stats = self.attention(field, old, diagnostic)
            local = self.comparator(torch.cat((old, field), 1))
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
        logits, priority = self.priority_readout(field, rates, local, task)
        if diagnostic:
            return logits, dict(
                states=states,
                first_field=first_field,
                records=records,
                priority=priority,
                terminal_fields=(field, rates, local),
            )
        return logits


def groups(model, cfg):
    grouped = {}
    high_prefixes = (
        "memory.",
        "memory_input.",
        "comparator.",
        "attention.",
        "priority_readout.",
    )
    for name, parameter in model.named_parameters():
        high = name.startswith(high_prefixes)
        decay = (
            cfg["weight_decay"]
            if parameter.ndim >= 2
            and name.endswith("weight")
            and "norm" not in name
            else 0.0
        )
        grouped.setdefault(
            (cfg["new_lr"] if high else cfg["parent_lr"], decay), []
        ).append(parameter)
    return [
        dict(params=parameters, lr=key[0], weight_decay=key[1])
        for key, parameters in grouped.items()
    ]


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def initialize_scratch(arm, cfg):
    """Construct every model/optimizer tensor fresh from documented seeds."""
    if arm != ARM or cfg.get("battery") != "spatial":
        raise ValueError("Expected scratch spatial-priority arm and five-task recipe")
    model = SpatialPriorityMemory(cfg)
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
        attention=(
            "Original JointAttention class and source_bias - "
            "softplus(raw_locality)*distance_squared computation, freshly initialized."
        ),
        architecture=(
            "All components train from scratch. Terminal concat[H_T,R_T,C_T] at "
            "13x13 uses shared 1x1 192->96 and 3x3 96->64 convolutions, then "
            "per-task selection/evidence maps and spatial-softmax weighted evidence."
        ),
    )
