"""Essential shape, finite, gradient, lineage, and scratch-parity checks."""
import copy
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from WorkingMemory.SpatialPriorityReadout.DualAttention.model import (
    ARM,
    DualAttentionMemory,
    initialize_scratch,
    state_sha256,
)
from WorkingMemory.SpatialPriorityReadout.DualAttention.protocol import (
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
        "DUAL_ATTENTION_CONTROL_INIT",
        str(ROOT / "WorkingMemory/AttentionContextComparator/fixtures/control_checkpoint_000000.pt"),
    )
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grad_norm(parameters):
    values = [value.grad.square().sum() for value in parameters if value.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main():
    if sha(CONTROL) != CONTROL_SHA256:
        raise RuntimeError("Control checkpoint-0 fixture SHA mismatch")
    control_checkpoint = torch.load(CONTROL, map_location="cpu")
    if (
        control_checkpoint["version"] != "spatial_priority_readout_scratch_v2"
        or control_checkpoint["step"] != 0
        or control_checkpoint["counts"] != {"episodes": 0, "frames": 0}
        or control_checkpoint["optimizer"]["state"]
    ):
        raise RuntimeError("Expected untouched scratch control checkpoint 0")

    cfg = recipe()
    assert_fixed(cfg)
    model, optimizer, lineage = initialize_scratch(ARM, cfg)
    repeated, repeated_optimizer, repeated_lineage = initialize_scratch(
        ARM, copy.deepcopy(cfg)
    )
    assert state_sha256(model) == state_sha256(repeated)
    assert lineage == repeated_lineage
    assert not optimizer.state and not repeated_optimizer.state
    assert not hasattr(model, "attention")
    assert not hasattr(model, "comparator")
    assert not hasattr(model, "memory_output")
    assert not hasattr(model, "comparison_output")
    assert not hasattr(model.pre_attention, "gamma")
    assert not hasattr(model.post_attention, "gamma")

    control_model = SpatialPriorityMemory(cfg)
    control_state = control_model.state_dict()
    shared = {}
    excluded = ("attention.", "comparator.", "priority_readout.shared.")
    model_state = model.state_dict()
    for name, value in model_state.items():
        if name.startswith(excluded) or name.startswith(("pre_attention.", "post_attention.")):
            continue
        if name in control_state and value.shape == control_state[name].shape:
            if not torch.equal(value, control_state[name]):
                raise RuntimeError("Shared scratch tensor differs: " + name)
            shared[name] = value
    checkpoint_shared_equal = all(
        name in control_checkpoint["model"]
        and torch.equal(value, control_checkpoint["model"][name])
        for name, value in shared.items()
    )
    if os.environ.get("REQUIRE_CONTROL_CHECKPOINT_EQUALITY") == "1":
        if not checkpoint_shared_equal:
            raise RuntimeError("Shared tensor differs from control checkpoint 0")

    batch = 2
    field = torch.randn(batch, 64, 13, 13, requires_grad=True)
    old = torch.randn_like(field, requires_grad=True)
    drive, context, _ = model.pre_attention(field, old, diagnostic=True)
    assert drive.shape == context.shape == (batch, 64, 13, 13)
    rates = torch.randn_like(field, requires_grad=True)
    priority_field, _ = model.post_attention(context, field, rates, diagnostic=True)
    assert model.post_attention.last_query.shape == (batch, 169, 64)
    assert model.post_attention.last_key.shape == (batch, 507, 64)
    assert model.post_attention.last_value.shape == (batch, 507, 64)
    assert model.post_attention.last_weights.shape == (batch, 1, 169, 507)
    assert priority_field.shape == (batch, 64, 13, 13)
    task = next(iter(cfg["task_classes"]))
    logits, spatial_priority = model.priority_readout(priority_field, task)
    assert logits.shape == (batch, cfg["task_classes"][task])
    assert spatial_priority.shape == (batch, 1, 13, 13)
    assert torch.allclose(spatial_priority.sum((2, 3)), torch.ones(batch, 1), atol=1e-6)
    logits.square().mean().backward()
    isolated_gradients = {
        "pre_q": grad_norm(model.pre_attention.query.parameters()),
        "pre_k": grad_norm(model.pre_attention.key.parameters()),
        "pre_v": grad_norm(model.pre_attention.value.parameters()),
        "post_q": grad_norm(model.post_attention.query.parameters()),
        "post_k": grad_norm(model.post_attention.key.parameters()),
        "post_v": grad_norm(model.post_attention.value.parameters()),
    }
    assert all(value > 0 for value in isolated_gradients.values())

    model.zero_grad(set_to_none=True)
    model.eval()
    images = torch.rand(1, 3, 3, 100, 100)
    full_logits, diagnostics = model(images, task, diagnostic=True)
    assert torch.isfinite(full_logits).all()
    assert diagnostics["post_query"].shape == (1, 169, 64)
    assert diagnostics["post_key"].shape == (1, 507, 64)
    assert diagnostics["post_value"].shape == (1, 507, 64)
    assert diagnostics["post_weights"].shape == (1, 1, 169, 507)
    assert diagnostics["terminal_priority_field"].shape == (1, 64, 13, 13)
    full_logits.square().mean().backward()
    full_gradients = {
        "pre_q": grad_norm(model.pre_attention.query.parameters()),
        "pre_k": grad_norm(model.pre_attention.key.parameters()),
        "pre_v": grad_norm(model.pre_attention.value.parameters()),
        "post_q": grad_norm(model.post_attention.query.parameters()),
        "post_k": grad_norm(model.post_attention.key.parameters()),
        "post_v": grad_norm(model.post_attention.value.parameters()),
    }
    assert all(value > 0 for value in full_gradients.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    forward_source = inspect.getsource(DualAttentionMemory.forward)
    decoder_source = inspect.getsource(model.priority_readout.forward)
    assert "self.priority_readout(priority_field, task)" in forward_source
    assert "field" not in inspect.signature(model.priority_readout.forward).parameters
    assert ".mean((2, 3))" not in forward_source + decoder_source
    assert ".amax((2, 3))" not in forward_source + decoder_source

    result = dict(
        status="passed",
        version=lineage["version"],
        exact_post_shapes=dict(
            query=[batch, 169, 64],
            key=[batch, 507, 64],
            value=[batch, 507, 64],
            weights=[batch, 1, 169, 507],
            priority_field=[batch, 64, 13, 13],
        ),
        finite_forward_backward=True,
        output_loss_gradient_norms=full_gradients,
        isolated_path_gradient_norms=isolated_gradients,
        decoder_terminal_input_only="P_T",
        decoder_signature=list(inspect.signature(model.priority_readout.forward).parameters),
        priority_weights_normalized=True,
        loaded_parent=False,
        loaded_model_tensors=0,
        loaded_optimizer_states=0,
        optimizer_state_empty=True,
        deterministic_scratch_initialization=True,
        shared_control_tensor_count=len(shared),
        all_shared_tensors_exact_reconstructed_control=True,
        all_shared_tensors_exact_control_checkpoint0=checkpoint_shared_equal,
        checkpoint_equality_required=(
            os.environ.get("REQUIRE_CONTROL_CHECKPOINT_EQUALITY") == "1"
        ),
        control_checkpoint_sha256=CONTROL_SHA256,
        prospective_gamma_absent=True,
        old_comparator_absent=True,
        global_pooling_absent=True,
        total_parameters=sum(value.numel() for value in model.parameters()),
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        initial_model_sha256=lineage["initial_model_sha256"],
        updates=UPDATES,
        episodes=TOTAL_EPISODES,
        validation_targets=list(VALIDATION_TARGETS),
        lineage=lineage,
    )
    output = Path(__file__).with_name("construction_checks.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
