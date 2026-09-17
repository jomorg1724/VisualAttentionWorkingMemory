"""Spatially preserving priority-map readout from the intact attention8400 parent."""
import copy
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from WorkingMemory.PreUpdateAttention.model import (
    AttentionMemory,
    groups as parent_groups,
)

VERSION = "spatial_priority_readout_v1"
ARM = "spatial_priority_readout"
PARENT_VERSION = "spatial_preupdate_joint_attention_v1"
PARENT_STEP = 8400
PARENT_SHA256 = "e37602aa20ccfc400ea8fe9d98c11f29c508069388897c55803b97f2ccdf1bc9"
REPLACED_PREFIXES = (
    "readout.trunk.",
    "readout.heads.",
    "memory_output.",
    "comparison_output.",
)


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
    """Original attention/E-I/comparator path with only the terminal readout replaced."""

    def __init__(self, cfg):
        super().__init__(seed=cfg["model_seed"], activation_checkpoint=cfg["activation_checkpoint"])
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


def migrate(parent, arm, cfg, parent_sha256):
    """Accept only the exact intact parent and initialize only the new readout."""
    if parent_sha256 != PARENT_SHA256:
        raise ValueError("Expected exact intact attention8400 SHA256")
    if parent.get("version") == VERSION:
        raise ValueError("Refusing an already migrated priority-readout checkpoint")
    if "prospective" in str(parent.get("version", "")).lower():
        raise ValueError("Refusing prospective-query lineage")
    if (
        parent.get("version") != PARENT_VERSION
        or parent.get("step") != PARENT_STEP
        or arm != ARM
        or cfg.get("battery") != "spatial"
    ):
        raise ValueError("Expected original attention8400 and fixed five-task recipe")

    old = AttentionMemory(activation_checkpoint=False)
    old.load_state_dict(parent["model"], strict=True)
    old_optimizer = torch.optim.Adam(
        parent_groups(old, parent["config"]), eps=parent["config"]["adam_eps"]
    )
    old_optimizer.load_state_dict(parent["optimizer"])

    model = SpatialPriorityMemory(cfg)
    state = model.state_dict()
    retained = []
    dropped = []
    for name, value in parent["model"].items():
        if name.startswith(REPLACED_PREFIXES):
            dropped.append(name)
            continue
        if name not in state or state[name].shape != value.shape:
            raise RuntimeError("Unexpected retained tensor mismatch " + name)
        state[name] = value
        retained.append(name)
    new_tensors = set(state) - set(retained)
    if not new_tensors or not all(
        name.startswith("priority_readout.") for name in new_tensors
    ):
        raise RuntimeError("Only priority-readout tensors may be newly initialized")
    if "attention.gamma" in state:
        raise RuntimeError("Prospective gamma must be absent")
    model.load_state_dict(state, strict=True)
    for name in retained:
        if not torch.equal(model.state_dict()[name], parent["model"][name]):
            raise RuntimeError("Inherited tensor changed " + name)

    optimizer = torch.optim.Adam(groups(model, cfg), eps=cfg["adam_eps"])
    old_names = dict(old.named_parameters())
    inherited_adam = []
    for name, parameter in model.named_parameters():
        if name in old_names and old_names[name] in old_optimizer.state:
            optimizer.state[parameter] = copy.deepcopy(
                old_optimizer.state[old_names[name]]
            )
            inherited_adam.append(name)
    if set(inherited_adam) != {
        name
        for name in retained
        if name in old_names and old_names[name] in old_optimizer.state
    }:
        raise RuntimeError("Compatible Adam-state transfer is incomplete")

    new_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("priority_readout.")
    )
    dropped_parameters = sum(
        value.numel()
        for name, value in parent["model"].items()
        if name.startswith(REPLACED_PREFIXES)
        and not name.endswith("distance_squared")
    )
    return model, optimizer, dict(
        version=VERSION,
        parent_version=PARENT_VERSION,
        parent_step=PARENT_STEP,
        parent_sha256=PARENT_SHA256,
        retained_tensors=retained,
        retained_adam_names=inherited_adam,
        dropped_parent_tensors=sorted(dropped),
        new_tensors=sorted(new_tensors),
        new_readout_parameters=new_parameters,
        dropped_readout_parameters=dropped_parameters,
        prospective_gamma_absent=True,
        attention_change="none; original JointAttention query, source_bias and locality",
        architecture_change=(
            "Replace three terminal global mean/max branches with concat[H_T,R_T,C_T] "
            "at 13x13; shared 1x1 192->96 and 3x3 96->64 convolutions; per-task "
            "1x1 selection/evidence maps; spatial softmax weighted evidence sum."
        ),
        initialization=(
            "Original model_seed lineage; shared/evidence convolutions use deterministic "
            "PyTorch initialization at model_seed+100 and selection logits start at zero "
            "(uniform 1/169 priority)."
        ),
    )
