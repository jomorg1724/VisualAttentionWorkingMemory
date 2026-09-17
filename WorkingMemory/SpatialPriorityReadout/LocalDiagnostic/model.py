"""Analysis-only probes over cached terminal H_T/R_T/C_T fields."""
from __future__ import annotations

import math

import torch
from torch import nn


class PooledProbe(nn.Module):
    """Capacity-matched global mean/max baseline."""

    def __init__(self, classes: int = 4, seed: int = 731101):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.hidden = nn.Linear(192 * 2, 191)
            self.output = nn.Linear(191, classes)

    def forward(self, fields: torch.Tensor):
        pooled = torch.cat((fields.mean((2, 3)), fields.amax((2, 3))), 1)
        return self.output(torch.nn.functional.silu(self.hidden(pooled)))


class SpatialPriorityProbe(nn.Module):
    """Single-task counterpart of the cloud priority-map readout."""

    def __init__(self, classes: int = 4, seed: int = 731101):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.shared = nn.Sequential(
                nn.Conv2d(192, 96, 1, bias=False),
                nn.GroupNorm(8, 96),
                nn.SiLU(),
                nn.Conv2d(96, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
            )
            self.selection = nn.Conv2d(64, 1, 1)
            self.evidence = nn.Conv2d(64, classes, 1)
            nn.init.zeros_(self.selection.weight)
            nn.init.zeros_(self.selection.bias)

    def forward(self, fields: torch.Tensor, return_maps: bool = False):
        features = self.shared(fields)
        priority = self.selection(features).flatten(1).softmax(1)
        priority = priority.reshape(-1, 1, 13, 13)
        evidence = self.evidence(features)
        logits = (priority * evidence).sum((2, 3))
        return (logits, priority[:, 0], evidence) if return_maps else logits


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def terminal_fields(model, images: torch.Tensor) -> torch.Tensor:
    """Reproduce the frozen BiasedMemory forward path and return concat[H,R,C]."""
    traces = ()
    state = None
    field = rates = comparison = None
    for timestep in range(images.shape[1]):
        result = model._sensory(images[:, timestep], *traces)
        field, traces = result[0], result[1:]
        old = torch.zeros_like(field) if state is None else state[0]
        attended, _ = model.attention(field, old, False)
        comparison = model.comparator(torch.cat((old, field), 1))
        rates, state, _ = model.memory(model.memory_input(attended), state, False)
    if field is None or rates is None or comparison is None:
        raise RuntimeError("Empty sequence")
    fields = torch.cat((field, rates, comparison), 1)
    if tuple(fields.shape[1:]) != (192, 13, 13):
        raise RuntimeError(f"Unexpected terminal field shape {tuple(fields.shape)}")
    return fields


def original_logits_from_fields(model, fields: torch.Tensor, task: str):
    """Reconstruct the historical pooled output for extraction equivalence checks."""
    field, rates, comparison = fields.split(64, 1)
    sensory = model.readout.trunk(
        torch.cat((field.mean((2, 3)), field.amax((2, 3))), 1)
    )
    retained = torch.cat((rates.mean((2, 3)), rates.amax((2, 3))), 1)
    comp = torch.cat(
        (comparison.mean((2, 3)), comparison.amax((2, 3))), 1
    )
    return model.classify(
        sensory
        + model.memory_output(retained)
        + model.comparison_output(comp),
        task,
    )


def target_region_mask(targets: torch.Tensor, device=None):
    """Evaluation-only 3x3 neighborhoods around projected stimulus centers."""
    centers = torch.tensor(
        [[27.0, 27.0], [73.0, 27.0], [27.0, 73.0], [73.0, 73.0]],
        device=device,
    )
    projected = torch.round(centers * (12.0 / 99.0)).long()
    yy, xx = torch.meshgrid(
        torch.arange(13, device=device),
        torch.arange(13, device=device),
        indexing="ij",
    )
    chosen = projected[targets.to(device)]
    return (
        (xx[None] - chosen[:, 0, None, None]).abs() <= 1
    ) & ((yy[None] - chosen[:, 1, None, None]).abs() <= 1)

