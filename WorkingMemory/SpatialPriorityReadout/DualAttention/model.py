"""Scratch dual-attention memory with an exclusive terminal priority readout."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from WorkingMemory.SpatialPriorityReadout.model import (
    SpatialPriorityMemory,
)

VERSION = "dual_attention_spatial_priority_scratch_v1"
ARM = "dual_attention_spatial_priority_scratch"


def _grid_distance(source_count):
    xy = torch.stack(
        torch.meshgrid(torch.arange(13), torch.arange(13), indexing="ij"), -1
    ).reshape(169, 2).float()
    distance = (xy[:, None] - xy[None, :]).square().sum(-1)
    return distance.repeat(1, source_count)


class PreUpdateAttention(nn.Module):
    """One width-64 head from previous memory into current sensory/memory tokens."""

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(64)
        self.key_norm = nn.LayerNorm(64)
        self.position = nn.Parameter(torch.empty(169, 64))
        self.source = nn.Parameter(torch.empty(2, 64))
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.output = nn.Linear(64, 64, bias=False)
        self.raw_locality = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
        self.source_bias = nn.Parameter(torch.tensor([2.0, 0.0]))
        self.register_buffer("distance_squared", _grid_distance(2))
        with torch.no_grad():
            self.position.normal_(0, 0.02)
            self.source.normal_(0, 0.02)
            nn.init.xavier_uniform_(self.query.weight, gain=0.1)
            nn.init.xavier_uniform_(self.key.weight, gain=0.1)
            nn.init.eye_(self.value.weight)
            nn.init.eye_(self.output.weight)
        self.last_query = None
        self.last_key = None
        self.last_value = None
        self.last_weights = None
        self.last_context = None

    def forward(self, field, old, diagnostic=False):
        batch = field.shape[0]
        visual = field.flatten(2).transpose(1, 2)
        memory = old.flatten(2).transpose(1, 2)
        query = self.query(
            self.query_norm(memory) + self.position + self.source[1]
        )
        raw = torch.cat((visual, memory), 1)
        identities = torch.cat(
            (self.position + self.source[0], self.position + self.source[1]), 0
        )
        key = self.key(self.key_norm(raw) + identities)
        value = self.value(raw)
        logits = (query @ key.transpose(-1, -2)) / math.sqrt(64)
        bias = (
            self.source_bias.repeat_interleave(169)[None, None, :]
            - F.softplus(self.raw_locality)
            * self.distance_squared[None]
        )
        weights = (logits + bias).softmax(-1)
        context = weights @ value
        drive = self.output(context).transpose(1, 2).reshape(batch, 64, 13, 13)
        context_field = context.transpose(1, 2).reshape(batch, 64, 13, 13)
        if diagnostic:
            for tensor in (query, key, value, weights, context):
                if tensor.requires_grad:
                    tensor.retain_grad()
            self.last_query = query
            self.last_key = key
            self.last_value = value
            self.last_weights = weights[:, None]
            self.last_context = context_field
            stats = dict(
                pre_memory_mass=float(weights[:, :, 169:].detach().sum(-1).mean()),
                pre_locality=float(F.softplus(self.raw_locality).detach()),
            )
        else:
            self.last_query = self.last_key = self.last_value = None
            self.last_weights = self.last_context = None
            stats = {}
        return drive, context_field, stats


class PostUpdateAttention(nn.Module):
    """One width-64 head from updated memory into C/H/R token banks."""

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(64)
        self.key_norm = nn.LayerNorm(64)
        self.position = nn.Parameter(torch.empty(169, 64))
        self.query_identity = nn.Parameter(torch.empty(64))
        self.source = nn.Parameter(torch.empty(3, 64))
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.output = nn.Linear(64, 64, bias=False)
        self.raw_locality = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
        self.source_bias = nn.Parameter(torch.tensor([0.0, 2.0, 0.0]))
        self.register_buffer("distance_squared", _grid_distance(3))
        with torch.no_grad():
            self.position.normal_(0, 0.02)
            self.query_identity.normal_(0, 0.02)
            self.source.normal_(0, 0.02)
            nn.init.xavier_uniform_(self.query.weight, gain=0.1)
            nn.init.xavier_uniform_(self.key.weight, gain=0.1)
            nn.init.eye_(self.value.weight)
            nn.init.eye_(self.output.weight)
        self.last_query = None
        self.last_key = None
        self.last_value = None
        self.last_weights = None
        self.last_priority_field = None

    def forward(self, context, field, rates, diagnostic=False):
        batch = field.shape[0]
        context_tokens = context.flatten(2).transpose(1, 2)
        visual_tokens = field.flatten(2).transpose(1, 2)
        rate_tokens = rates.flatten(2).transpose(1, 2)
        query = self.query(
            self.query_norm(rate_tokens) + self.position + self.query_identity
        )
        raw = torch.cat((context_tokens, visual_tokens, rate_tokens), 1)
        identities = torch.cat(
            tuple(self.position + self.source[index] for index in range(3)), 0
        )
        key = self.key(self.key_norm(raw) + identities)
        value = self.value(raw)
        logits = (query @ key.transpose(-1, -2)) / math.sqrt(64)
        bias = (
            self.source_bias.repeat_interleave(169)[None, None, :]
            - F.softplus(self.raw_locality)
            * self.distance_squared[None]
        )
        weights = (logits + bias).softmax(-1)
        merged = weights @ value
        priority = self.output(merged).transpose(1, 2).reshape(
            batch, 64, 13, 13
        )
        if diagnostic:
            for tensor in (query, key, value, weights, priority):
                if tensor.requires_grad:
                    tensor.retain_grad()
            self.last_query = query
            self.last_key = key
            self.last_value = value
            self.last_weights = weights[:, None]
            self.last_priority_field = priority
            stats = dict(
                post_source_mass=weights.detach()
                .reshape(batch, 169, 3, 169)
                .sum(-1)
                .mean((0, 1))
                .tolist(),
                post_locality=float(F.softplus(self.raw_locality).detach()),
            )
        else:
            self.last_query = self.last_key = self.last_value = None
            self.last_weights = self.last_priority_field = None
            stats = {}
        return priority, stats


class TerminalPriorityReadout(nn.Module):
    """Convolutional task-local selection that accepts only P_T."""

    def __init__(self, control_readout, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 200)
            self.shared = nn.Sequential(
                nn.Conv2d(64, 96, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(96, 64, 3, padding=1, bias=False),
            )
        self.selection = control_readout.selection
        self.evidence = control_readout.evidence
        self.capture = False
        self.last_priority = None
        self.last_evidence = None
        self.last_input = None

    def forward(self, priority_field, task):
        if tuple(priority_field.shape[1:]) != (64, 13, 13):
            raise ValueError("Terminal decoder expects only P_T [B,64,13,13]")
        features = self.shared(priority_field)
        selection_logits = self.selection[task](features).flatten(1)
        priority = selection_logits.softmax(1).reshape(-1, 1, 13, 13)
        evidence = self.evidence[task](features)
        logits = (priority * evidence).sum((2, 3))
        if self.capture:
            self.last_input = priority_field.detach()
            self.last_priority = priority[:, 0].detach()
            self.last_evidence = evidence.detach()
        return logits, priority


class DualAttentionMemory(SpatialPriorityMemory):
    """Pre-update attention, unchanged E/I update, post-update attention, P-only head."""

    def __init__(self, cfg):
        super().__init__(cfg)
        control_readout = self.priority_readout
        del self.comparator
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["model_seed"] + 300)
            self.pre_attention = PreUpdateAttention()
            self.post_attention = PostUpdateAttention()
        del self.attention
        self.priority_readout = TerminalPriorityReadout(
            control_readout, cfg["model_seed"]
        )

    def forward(self, images, task, diagnostic=False):
        traces = ()
        state = None
        records = []
        states = []
        first_field = None
        context = None
        priority_field = None
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
            drive, context, pre_stats = self.pre_attention(field, old, diagnostic)
            rates, state, memory_stats = self.memory(
                self.memory_input(drive), state, diagnostic
            )
            priority_field, post_stats = self.post_attention(
                context, field, rates, diagnostic
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
                        **pre_stats,
                        **post_stats,
                    }
                )
        logits, priority = self.priority_readout(priority_field, task)
        if diagnostic:
            return logits, dict(
                states=states,
                first_field=first_field,
                records=records,
                priority=priority,
                terminal_priority_field=priority_field,
                pre_context=context,
                pre_weights=self.pre_attention.last_weights,
                post_weights=self.post_attention.last_weights,
                post_query=self.post_attention.last_query,
                post_key=self.post_attention.last_key,
                post_value=self.post_attention.last_value,
            )
        return logits


def groups(model, cfg):
    grouped = {}
    high_prefixes = (
        "memory.",
        "memory_input.",
        "pre_attention.",
        "post_attention.",
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
    if arm != ARM or cfg.get("battery") != "spatial":
        raise ValueError("Expected scratch dual-attention five-task arm")
    model = DualAttentionMemory(cfg)
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
            dual_attention=cfg["model_seed"] + 300,
            priority_decoder=cfg["model_seed"] + 200,
            python_numpy_torch=cfg["model_seed"],
            training_stream=cfg["train_seed"],
            task_scheduler=cfg["scheduler_seed"],
        ),
        prospective_gamma_absent=True,
        architecture=(
            "One-head width-64 pre attention produces raw C and W_O_pre(C) memory "
            "drive; unchanged spatial E/I update; one-head width-64 post attention "
            "queries R over [C,H,R]; terminal P alone enters the convolutional "
            "task-specific spatial-softmax priority decoder."
        ),
    )
