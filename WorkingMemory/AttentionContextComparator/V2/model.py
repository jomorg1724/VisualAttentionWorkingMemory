"""AV-context v2: five explicit changes over attention_context_comparator_scratch_v1.

Every change is implemented as an override on the unchanged v1 classes, so the
pinned v1 sources (accumulators.py, WorkingMemory/model.py,
RecurrentComparison/model.py, PreUpdateAttention/model.py,
SpatialPriorityReadout/model.py, AttentionContextComparator/model.py) keep
their byte-identical hashes for the arms that are still training against them.

Change A  restore the discarded opponent signal: energy_channels returns
          (normalized, raw_opponent, magnitude); the accumulator feature width
          grows 72 -> 77 and the five new input columns start at zero.
Change B  mixed pooling: each scale contributes adaptive average AND max
          pooling (32 -> 64 channels per scale); readout.fusion grows
          ConvNormAct(96,64) -> ConvNormAct(192,64), max-half weights zero.
Change C  attention head 1 becomes wide (lambda = 0.02) and source-neutral;
          head 0 keeps lambda = 4 and the +2 visual bias.
Change D  priority selection maps: xavier_uniform(gain=.5), zero bias,
          instead of all-zero weights.
Change E  one learning rate (new_lr) for every parameter of the scratch arm;
          applied through the recipe (parent_lr := new_lr), see protocol.py.

With C and D disabled, v2 logits are bit-identical to v1 on any input.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from PreAttentiveVision.decoder import ConvNormAct
from PreAttentiveVision.TemporalIntegration.accumulators import OpponentAccumulator
from WorkingMemory.AttentionContextComparator.model import (
    AttentionContextComparatorMemory,
    ContextJointAttention,
)
from WorkingMemory.SpatialPriorityReadout.model import groups as v1_groups
from WorkingMemory.SpatialPriorityReadout.model import state_sha256

VERSION = "attention_context_comparator_v2"
ARM = "attention_context_comparator_v2"
CHANGE_KEYS = ("A", "B", "C", "D", "E")
ALL_CHANGES = {key: True for key in CHANGE_KEYS}
CHANGE_TEXT = dict(
    A=(
        "opponent raw_opponent[4] + magnitude[1] channels restored; accumulator "
        "output Conv2d(77,32,1), new columns zero"
    ),
    B=(
        "mixed adaptive avg+max pooling per scale; readout.fusion "
        "ConvNormAct(192,64), max-half zero"
    ),
    C="attention head 1 lambda=0.02 (sigma=5 cells) and source-neutral; head 0 unchanged",
    D="priority selection maps xavier_uniform(gain=.5), zero bias",
    E="single learning rate new_lr for all parameters (recipe sets parent_lr = new_lr)",
)
SELECTION_INIT_SEED_OFFSET = 200
NEW_OPPONENT_CHANNELS = 5
FAR_CELLS = 3.0


def inverse_softplus(value):
    return math.log(math.expm1(value))


class OpponentAccumulatorV2(OpponentAccumulator):
    """Same quadrature bank; energy_channels also returns raw opponent and magnitude."""

    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(72 + NEW_OPPONENT_CHANNELS, 32, 1)

    def energy_channels(self, fast, slow):
        batch, channels, height, width = fast.shape
        response = F.conv2d(
            F.pad(torch.cat((fast, slow), 0), (3,) * 4, mode="reflect"),
            self.kernels,
            groups=32,
        )
        response = response.view(2 * batch, channels, 4, 2, height, width)
        fast_response, slow_response = response.split(batch, 0)
        fe, fo = fast_response.unbind(3)
        se, so = slow_response.unbind(3)
        positive = (fe + so).square() + (fo - se).square()
        negative = (fe - so).square() + (fo + se).square()
        energies = torch.stack((positive.mean(1), negative.mean(1)), 2).flatten(1, 2)
        pooled = F.avg_pool2d(F.pad(energies, (1,) * 4, mode="reflect"), 3, stride=1)
        normalized = pooled / (1e-4 + pooled.mean(1, keepdim=True))
        raw_opponent = positive.mean(1) - negative.mean(1)  # [B,4,H,W]
        # Total motion energy over feature channels and the four axis/frequency
        # pairs: one unsigned magnitude map [B,1,H,W].
        magnitude = (positive + negative).mean((1, 2)).unsqueeze(1)
        return normalized, raw_opponent, magnitude

    def features(self, fast, slow):
        energies, raw_opponent, magnitude = self.energy_channels(fast, slow)
        return torch.cat(
            (0.5 * (fast + slow), fast - slow, energies, raw_opponent, magnitude), 1
        )

    def forward(self, current, state=None):
        if state is None:
            fast = slow = current
        else:
            if len(state) != 2 or any(x.shape != current.shape for x in state):
                raise ValueError("Opponent trace shapes do not match current batch/map")
            fast = 0.25 * state[0] + 0.75 * current
            slow = 0.75 * state[1] + 0.25 * current
        return self.output(self.features(fast, slow)), (fast, slow)


class ContextJointAttentionV2(ContextJointAttention):
    """v1 attention with per-head locality diagnostics (mass beyond FAR_CELLS)."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "far_mask",
            (self.distance_squared > FAR_CELLS ** 2).float(),
            persistent=False,
        )

    def forward(self, field, old, diagnostic=False):
        attended, context_field, stats = super().forward(field, old, diagnostic)
        if diagnostic:
            weights = self.last_weights.detach()
            far = (weights * self.far_mask).sum(-1).mean((0, 2))
            visual = weights[:, :, :, :169].sum(-1).mean((0, 2))
            stats.update(
                attention_mass_beyond_3_by_head=far.tolist(),
                visual_attention_mass_by_head=visual.tolist(),
                source_bias=self.source_bias.detach().tolist(),
            )
        return attended, context_field, stats


class AttentionContextComparatorV2(AttentionContextComparatorMemory):
    def __init__(self, cfg):
        super().__init__(cfg)
        changes = dict(ALL_CHANGES)
        changes.update(cfg.get("v2_changes", {}))
        unknown = set(changes) - set(CHANGE_KEYS)
        if unknown:
            raise ValueError("Unknown v2 change flags " + repr(sorted(unknown)))
        self.changes = changes
        seed = cfg["model_seed"]

        # Diagnostics-only subclass swap; parameters are the v1 tensors.
        attention = ContextJointAttentionV2()
        attention.load_state_dict(self.attention.state_dict(), strict=True)
        self.attention = attention

        if changes["A"]:
            cores = nn.ModuleList()
            for core in self.accumulators:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    new = OpponentAccumulatorV2()
                with torch.no_grad():
                    new.kernels.copy_(core.kernels)
                    new.output.weight.zero_()
                    new.output.weight[:, :72] = core.output.weight
                    new.output.bias.copy_(core.output.bias)
                cores.append(new)
            self.accumulators = cores

        if changes["B"]:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                fusion = ConvNormAct(192, 64)
            old = self.readout.fusion
            with torch.no_grad():
                fusion[0].weight.zero_()
                for scale in range(3):
                    fusion[0].weight[:, 64 * scale : 64 * scale + 32] = old[0].weight[
                        :, 32 * scale : 32 * scale + 32
                    ]
                fusion[1].weight.copy_(old[1].weight)
                fusion[1].bias.copy_(old[1].bias)
            self.readout.fusion = fusion

        if changes["C"]:
            with torch.no_grad():
                self.attention.raw_locality.copy_(
                    torch.tensor([inverse_softplus(4.0), inverse_softplus(0.02)])
                )
                self.attention.source_bias.copy_(
                    torch.tensor([[2.0, 0.0], [0.0, 0.0]])
                )

        if changes["D"]:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    seed + SELECTION_INIT_SEED_OFFSET
                )
                for layer in self.priority_readout.selection.values():
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

    # Change A: emit all three opponent quantities.
    def _emit(self, state):
        if not self.changes["A"]:
            return super()._emit(state)
        return [
            core.output(core.features(fast, slow))
            for core, (fast, slow) in zip(self.accumulators, state)
        ]

    # Change B: mixed pooling before fusion.
    def _sensory(self, frame, *old):
        if not self.changes["B"]:
            return super()._sensory(frame, *old)
        current = self._encode(frame)
        state = (
            tuple((u, u) for u in current)
            if not old
            else tuple(
                (0.25 * old[2 * j] + 0.75 * u, 0.75 * old[2 * j + 1] + 0.25 * u)
                for j, u in enumerate(current)
            )
        )
        emitted = self._emit(state)
        scales = []
        for u, o, layer in zip(current, emitted, self.readout.local):
            x = layer(torch.cat((u, o), 1))
            scales.append(
                torch.cat(
                    (
                        F.adaptive_avg_pool2d(x, (13, 13)),
                        F.adaptive_max_pool2d(x, (13, 13)),
                    ),
                    1,
                )
            )
        field = self.readout.fusion(torch.cat(scales, 1))
        return (field,) + tuple(x for pair in state for x in pair)

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
            attended, context, attention_stats = self.attention(field, old, diagnostic)
            rates, state, memory_stats = self.memory(
                self.memory_input(attended), state, diagnostic
            )
            if diagnostic:
                if timestep == 0:
                    if field.requires_grad:
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
        capture = self.priority_readout.capture
        self.priority_readout.capture = capture or diagnostic
        logits, priority = self.priority_readout(field, rates, context, task)
        self.priority_readout.capture = capture
        if diagnostic:
            return logits, dict(
                states=states,
                first_field=first_field,
                records=records,
                priority=priority,
                evidence=self.priority_readout.last_evidence,
                terminal_fields=(field, rates, context),
                attention_context=context,
                attention_weights=self.attention.last_weights,
            )
        return logits


def groups(model, cfg):
    return v1_groups(model, cfg)


def learning_rate_summary(model, cfg):
    summary = {}
    for group in groups(model, cfg):
        key = f"{group['lr']:g}"
        summary[key] = summary.get(key, 0) + sum(p.numel() for p in group["params"])
    return summary


def initialize_scratch(arm, cfg):
    if arm != ARM or cfg.get("battery") != "spatial":
        raise ValueError("Expected the AV-context v2 arm and five-task recipe")
    model = AttentionContextComparatorV2(cfg)
    if hasattr(model, "comparator"):
        raise RuntimeError("Old learned comparator must be absent")
    if "attention.gamma" in model.state_dict():
        raise RuntimeError("Prospective gamma must be absent")
    if model.changes["E"] and cfg["parent_lr"] != cfg["new_lr"]:
        raise RuntimeError("Change E requires parent_lr == new_lr in the recipe")
    optimizer = torch.optim.Adam(groups(model, cfg), eps=cfg["adam_eps"])
    if optimizer.state:
        raise RuntimeError("Scratch optimizer must have no inherited state")
    return model, optimizer, dict(
        version=VERSION,
        base_version="attention_context_comparator_scratch_v1",
        initialization_kind="full_model_from_scratch",
        loaded_parent=False,
        loaded_model_tensors=0,
        loaded_optimizer_states=0,
        initial_model_sha256=state_sha256(model),
        changes={
            key: dict(enabled=bool(model.changes[key]), text=CHANGE_TEXT[key])
            for key in CHANGE_KEYS
        },
        learning_rate_parameter_counts=learning_rate_summary(model, cfg),
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
            priority_selection_v2=cfg["model_seed"] + SELECTION_INIT_SEED_OFFSET,
            python_numpy_torch=cfg["model_seed"],
            training_stream=cfg["train_seed"],
            task_scheduler=cfg["scheduler_seed"],
        ),
        prospective_gamma_absent=True,
        comparator=(
            "C_t is concat_heads(sum_j A_t,h,i,j V_t,h,j) before W_O; "
            "the learned Conv([R_t-1,H_t]) comparator is absent."
        ),
    )
