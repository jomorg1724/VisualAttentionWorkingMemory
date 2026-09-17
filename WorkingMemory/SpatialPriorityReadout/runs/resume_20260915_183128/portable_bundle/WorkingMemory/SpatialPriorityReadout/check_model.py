"""Focused scratch-construction and architecture checks; no training run."""
import copy
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from WorkingMemory.PreUpdateAttention.model import JointAttention
from WorkingMemory.SpatialComparison.model import LocalNorm, SpatialEI
from WorkingMemory.SpatialPriorityReadout.model import (
    ARM,
    SpatialPriorityMemory,
    initialize_scratch,
    state_sha256,
)
from WorkingMemory.SpatialPriorityReadout.protocol import (
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_TARGETS,
    assert_fixed,
    recipe,
)


def main():
    cfg = recipe()
    assert_fixed(cfg)
    model_a, optimizer_a, lineage_a = initialize_scratch(ARM, cfg)
    model_b, optimizer_b, lineage_b = initialize_scratch(ARM, copy.deepcopy(cfg))
    digest = state_sha256(model_a)
    assert digest == state_sha256(model_b)
    assert digest == lineage_a["initial_model_sha256"]
    assert lineage_a == lineage_b
    assert not optimizer_a.state and not optimizer_b.state
    assert lineage_a["loaded_parent"] is False
    assert lineage_a["loaded_model_tensors"] == 0
    assert lineage_a["loaded_optimizer_states"] == 0

    alternate = copy.deepcopy(cfg)
    alternate["model_seed"] += 1
    different, _, _ = initialize_scratch(ARM, alternate)
    assert state_sha256(different) != digest

    assert isinstance(model_a, SpatialPriorityMemory)
    assert isinstance(model_a.attention, JointAttention)
    assert isinstance(model_a.memory, SpatialEI)
    assert isinstance(model_a.memory_input, LocalNorm)
    assert type(model_a.comparator) is torch.nn.Sequential
    assert not hasattr(model_a.attention, "gamma")
    assert model_a.attention.source_bias.requires_grad
    assert model_a.attention.raw_locality.requires_grad
    assert not hasattr(model_a.readout, "trunk")
    assert not hasattr(model_a.readout, "heads")
    assert not hasattr(model_a, "memory_output")
    assert not hasattr(model_a, "comparison_output")
    forward_source = inspect.getsource(SpatialPriorityMemory.forward)
    assert ".mean((2, 3))" not in forward_source
    assert ".amax((2, 3))" not in forward_source

    batch = 2
    field = torch.randn(batch, 64, 13, 13, requires_grad=True)
    rates = torch.randn_like(field, requires_grad=True)
    comparison = torch.randn_like(field, requires_grad=True)
    task_shapes = {}
    losses = []
    for task, classes in cfg["task_classes"].items():
        logits, priority = model_a.priority_readout(field, rates, comparison, task)
        assert logits.shape == (batch, classes)
        assert priority.shape == (batch, 1, 13, 13)
        assert torch.allclose(priority.sum((2, 3)), torch.ones(batch, 1))
        assert torch.isfinite(logits).all() and torch.isfinite(priority).all()
        task_shapes[task] = list(logits.shape)
        losses.append(logits.square().mean())
    sum(losses).backward()
    gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in model_a.priority_readout.named_parameters()
        if parameter.grad is not None
    }
    assert gradients and all(torch.isfinite(torch.tensor(v)) for v in gradients.values())
    assert field.grad is not None and float(field.grad.norm()) > 0
    assert rates.grad is not None and float(rates.grad.norm()) > 0
    assert comparison.grad is not None and float(comparison.grad.norm()) > 0
    assert len({id(layer) for layer in model_a.priority_readout.selection.values()}) == 5
    assert len({id(layer) for layer in model_a.priority_readout.evidence.values()}) == 5

    model_a.zero_grad(set_to_none=True)
    model_a.eval()
    images = torch.rand(1, 2, 3, 100, 100)
    full_logits, diagnostics = model_a(
        images, next(iter(cfg["task_classes"])), diagnostic=True
    )
    assert torch.isfinite(full_logits).all()
    assert diagnostics["priority"].shape == (1, 1, 13, 13)
    full_logits.square().mean().backward()
    full_gradient_parameters = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model_a.parameters()
    )
    assert full_gradient_parameters > 0

    result = dict(
        status="passed",
        version=lineage_a["version"],
        initialization_kind=lineage_a["initialization_kind"],
        initial_model_sha256=digest,
        deterministic_same_seed=True,
        different_model_seed_changes_initialization=True,
        loaded_parent=False,
        loaded_model_tensors=0,
        loaded_optimizer_states=0,
        fresh_optimizer_state_empty=True,
        original_joint_attention_class=True,
        original_spatial_ei_class=True,
        original_comparator_class=True,
        source_and_locality_trainable=True,
        prospective_gamma_absent=True,
        global_terminal_pooling_absent=True,
        no_terminal_classifier_bypass=True,
        task_output_shapes=task_shapes,
        priority_maps_normalized=True,
        task_specific_selection_and_evidence=True,
        finite_full_forward_backward=True,
        full_gradient_parameter_count=int(full_gradient_parameters),
        priority_gradient_norms=gradients,
        updates=UPDATES,
        episodes=TOTAL_EPISODES,
        validation_targets=list(VALIDATION_TARGETS),
        total_parameters=sum(parameter.numel() for parameter in model_a.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model_a.parameters()
            if parameter.requires_grad
        ),
        lineage=lineage_a,
    )
    output = Path(__file__).with_name("construction_checks.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
