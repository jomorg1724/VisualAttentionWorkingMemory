"""Focused CPU construction and migration checks; this script performs no training run."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from WorkingMemory.PreUpdateAttention.model import JointAttention
from WorkingMemory.SpatialPriorityReadout.model import (
    ARM,
    PARENT_SHA256,
    SpatialPriorityMemory,
    migrate,
)
from WorkingMemory.SpatialPriorityReadout.protocol import (
    TOTAL_EPISODES,
    UPDATES,
    assert_fixed,
    recipe,
)

PARENT = Path(
    os.environ.get(
        "SPATIAL_PRIORITY_PARENT",
        str(
            ROOT
            / "WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/"
            "retrieved/remote_results/preupdate_attention/checkpoint_008400.pt"
        ),
    )
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_equal(left, right):
    if left.keys() != right.keys():
        return False
    return all(
        torch.equal(left[key], right[key])
        if torch.is_tensor(left[key])
        else left[key] == right[key]
        for key in left
    )


def main():
    if sha(PARENT) != PARENT_SHA256:
        raise RuntimeError("Parent SHA mismatch")
    cfg = recipe()
    assert_fixed(cfg)
    parent = torch.load(PARENT, map_location="cpu")
    model, optimizer, lineage = migrate(parent, ARM, cfg, sha(PARENT))
    model.eval()
    assert isinstance(model.attention, JointAttention)
    assert not hasattr(model.attention, "gamma")
    assert model.attention.source_bias.requires_grad
    assert model.attention.raw_locality.requires_grad

    parent_names = dict(
        SpatialPriorityMemory(cfg).named_parameters()
    )
    old_model = __import__(
        "WorkingMemory.PreUpdateAttention.model", fromlist=["AttentionMemory"]
    ).AttentionMemory(activation_checkpoint=False)
    old_model.load_state_dict(parent["model"])
    old_optimizer = torch.optim.Adam(
        __import__(
            "WorkingMemory.PreUpdateAttention.model", fromlist=["groups"]
        ).groups(old_model, parent["config"]),
        eps=parent["config"]["adam_eps"],
    )
    old_optimizer.load_state_dict(parent["optimizer"])
    old_names = dict(old_model.named_parameters())
    for name in lineage["retained_adam_names"]:
        assert state_equal(
            optimizer.state[dict(model.named_parameters())[name]],
            old_optimizer.state[old_names[name]],
        )
    for name in lineage["retained_tensors"]:
        assert torch.equal(model.state_dict()[name], parent["model"][name])

    batch = 2
    field = torch.randn(batch, 64, 13, 13, requires_grad=True)
    rates = torch.randn_like(field, requires_grad=True)
    comparison = torch.randn_like(field, requires_grad=True)
    task_shapes = {}
    priorities = {}
    losses = []
    for task, classes in cfg["task_classes"].items():
        logits, priority = model.priority_readout(field, rates, comparison, task)
        assert logits.shape == (batch, classes)
        assert priority.shape == (batch, 1, 13, 13)
        assert torch.allclose(priority.sum((2, 3)), torch.ones(batch, 1))
        assert torch.isfinite(logits).all() and torch.isfinite(priority).all()
        task_shapes[task] = list(logits.shape)
        priorities[task] = priority.detach().clone()
        losses.append(logits.square().mean())
    sum(losses).backward()
    gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in model.priority_readout.named_parameters()
        if parameter.grad is not None
    }
    assert gradients and all(torch.isfinite(torch.tensor(v)) for v in gradients.values())
    assert field.grad is not None and float(field.grad.norm()) > 0
    assert rates.grad is not None and float(rates.grad.norm()) > 0
    assert comparison.grad is not None and float(comparison.grad.norm()) > 0
    assert len({id(layer) for layer in model.priority_readout.selection.values()}) == 5
    assert len({id(layer) for layer in model.priority_readout.evidence.values()}) == 5

    bad = copy.deepcopy(parent)
    bad["version"] = "spatial_five_task_prospective_query_v1"
    rejected_prospective = False
    try:
        migrate(bad, ARM, cfg, PARENT_SHA256)
    except ValueError:
        rejected_prospective = True
    rejected_sha = False
    try:
        migrate(parent, ARM, cfg, "0" * 64)
    except ValueError:
        rejected_sha = True
    assert rejected_prospective and rejected_sha

    result = dict(
        status="passed",
        parent_sha256=PARENT_SHA256,
        parent_version=parent["version"],
        parent_step=parent["step"],
        updates=UPDATES,
        episodes=TOTAL_EPISODES,
        all_inherited_tensors_exact=True,
        all_compatible_adam_states_exact=True,
        prospective_gamma_absent=True,
        prospective_and_wrong_sha_rejected=True,
        original_attention_class=True,
        source_and_locality_trainable=True,
        task_output_shapes=task_shapes,
        priority_maps_normalized=True,
        task_specific_selection_and_evidence=True,
        finite_forward_backward=True,
        gradient_norms=gradients,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        lineage=lineage,
    )
    output = Path(__file__).with_name("construction_checks.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
