"""Essential construction, pairing, shape and gradient checks."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from WorkingMemory.AttentionContextComparator.model import (
    ARM,
    ContextJointAttention,
    initialize_scratch,
    state_sha256,
)
from WorkingMemory.AttentionContextComparator.protocol import (
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_TARGETS,
    assert_fixed,
    recipe,
)
from WorkingMemory.SpatialPriorityReadout.model import SpatialPriorityMemory

CONTROL_SHA256 = "97b7e6ce513592fc094724854ea1d4d329c93fd1ae0d86aa20ed50e4fd653f76"
CONTROL = Path(
    os.environ.get(
        "ATTENTION_CONTEXT_CONTROL_INIT",
        str(
            ROOT
            / "WorkingMemory/AttentionContextComparator/fixtures/"
            "control_checkpoint_000000.pt"
        ),
    )
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(CONTROL) != CONTROL_SHA256:
        raise RuntimeError("Control checkpoint-0 fixture SHA mismatch")
    control = torch.load(CONTROL, map_location="cpu")
    if (
        control["version"] != "spatial_priority_readout_scratch_v2"
        or control["step"] != 0
        or control["counts"] != {"episodes": 0, "frames": 0}
        or control["optimizer"]["state"]
    ):
        raise RuntimeError("Expected untouched scratch control checkpoint 0")

    cfg = recipe()
    assert_fixed(cfg)
    control_model = SpatialPriorityMemory(cfg)
    model, optimizer, lineage = initialize_scratch(ARM, cfg)
    repeated, repeated_optimizer, repeated_lineage = initialize_scratch(
        ARM, copy.deepcopy(cfg)
    )
    assert state_sha256(model) == state_sha256(repeated)
    assert lineage == repeated_lineage
    assert not optimizer.state and not repeated_optimizer.state
    assert isinstance(model.attention, ContextJointAttention)
    assert not hasattr(model, "comparator")
    assert not any(name.startswith("comparator.") for name in model.state_dict())
    assert not hasattr(model.attention, "gamma")

    state = model.state_dict()
    reconstructed_control_state = control_model.state_dict()
    control_state = control["model"]
    assert set(state).issubset(reconstructed_control_state)
    removed = sorted(set(reconstructed_control_state) - set(state))
    assert removed and all(name.startswith("comparator.") for name in removed)
    for name, value in state.items():
        if not torch.equal(value, reconstructed_control_state[name]):
            raise RuntimeError(
                "Shared scratch tensor differs from reconstructed control: " + name
            )
    checkpoint_shared_equal = all(
        torch.equal(value, control_state[name]) for name, value in state.items()
    )
    if os.environ.get("REQUIRE_CONTROL_CHECKPOINT_EQUALITY") == "1":
        if not checkpoint_shared_equal:
            raise RuntimeError(
                "Pinned-runtime shared tensor differs from control checkpoint 0"
            )

    batch = 2
    field = torch.randn(batch, 64, 13, 13, requires_grad=True)
    old = torch.randn_like(field, requires_grad=True)
    attended, context, _ = model.attention(field, old, diagnostic=True)
    assert attended.shape == context.shape == (batch, 64, 13, 13)
    rates = torch.randn_like(context)
    task = next(iter(cfg["task_classes"]))
    logits, priority = model.priority_readout(
        field.detach(), rates, context, task
    )
    assert logits.shape == (batch, cfg["task_classes"][task])
    assert priority.shape == (batch, 1, 13, 13)
    assert torch.allclose(priority.sum((2, 3)), torch.ones(batch, 1))
    loss = logits.square().mean()
    loss.backward()
    weights = model.attention.last_weights
    assert weights.grad is not None and float(weights.grad.norm()) > 0
    context_gradient_norms = {}
    for name in ("query.weight", "key.weight", "value.weight"):
        parameter = dict(model.attention.named_parameters())[name]
        assert parameter.grad is not None and float(parameter.grad.norm()) > 0
        context_gradient_norms[name] = float(parameter.grad.norm())
    assert model.attention.output.weight.grad is None

    model.zero_grad(set_to_none=True)
    model.eval()
    images = torch.rand(1, 2, 3, 100, 100)
    full_logits, diagnostics = model(images, task, diagnostic=True)
    assert torch.isfinite(full_logits).all()
    assert diagnostics["terminal_fields"][2].shape == (1, 64, 13, 13)
    assert diagnostics["priority"].shape == (1, 1, 13, 13)
    full_logits.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    result = dict(
        status="passed",
        version=lineage["version"],
        control_checkpoint_sha256=CONTROL_SHA256,
        deterministic_scratch_initialization=True,
        loaded_parent=False,
        loaded_model_tensors=0,
        loaded_optimizer_states=0,
        shared_tensor_count=len(state),
        all_shared_tensors_exact_reconstructed_control=True,
        all_shared_tensors_exact_control_checkpoint0=checkpoint_shared_equal,
        checkpoint_equality_required=(
            os.environ.get("REQUIRE_CONTROL_CHECKPOINT_EQUALITY") == "1"
        ),
        cross_platform_checkpoint_comparison=(
            "Exact equality required remotely under the control runtime; local "
            "Windows construction may differ from the Linux checkpoint."
        ),
        removed_control_tensors=removed,
        old_comparator_parameters_absent=True,
        prospective_gamma_absent=True,
        attention_context_shape=[batch, 64, 13, 13],
        priority_shape=[batch, 1, 13, 13],
        priority_normalized=True,
        attention_weight_gradient_norm=float(weights.grad.norm()),
        context_path_gradient_norms=context_gradient_norms,
        context_path_bypasses_W_O=True,
        finite_full_forward_backward=True,
        updates=UPDATES,
        episodes=TOTAL_EPISODES,
        validation_targets=list(VALIDATION_TARGETS),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        initial_model_sha256=lineage["initial_model_sha256"],
        lineage=lineage,
    )
    output = Path(__file__).with_name("construction_checks.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
