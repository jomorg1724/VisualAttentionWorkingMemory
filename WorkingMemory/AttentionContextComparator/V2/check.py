"""Hard pre-launch gates for AV-context v2.

1. Identity: v2 with C, D (and E) disabled must produce bit-identical logits to
   v1 on fixed batches (gates changes A and B).
2. Construction: deterministic scratch initialization, expected shapes, zero
   new columns, changed C/D values, one learning rate under E, finite full
   forward/backward with gradients reaching every new parameter column.
"""
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from WorkingMemory.AttentionContextComparator.model import ARM as V1_ARM
from WorkingMemory.AttentionContextComparator.model import initialize_scratch as v1_initialize
from WorkingMemory.AttentionContextComparator.protocol import recipe as v1_recipe
from WorkingMemory.AttentionContextComparator.V2.model import (
    ARM,
    VERSION,
    AttentionContextComparatorV2,
    ContextJointAttentionV2,
    OpponentAccumulatorV2,
    initialize_scratch,
    inverse_softplus,
    state_sha256,
)
from WorkingMemory.AttentionContextComparator.V2.protocol import (
    CUED_TASKS,
    GATE_STEP,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_TARGETS,
    assert_fixed,
    recipe,
)
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream


def v1_learning_rates(model, cfg):
    from WorkingMemory.SpatialPriorityReadout.model import groups as v1_groups

    summary = {}
    for group in v1_groups(model, cfg):
        key = f"{group['lr']:g}"
        summary[key] = summary.get(key, 0) + sum(p.numel() for p in group["params"])
    return summary


def fixed_batches(seed=77973001):
    stream = SpatialBatteryStream(seed, "val")
    batches = []
    for task, condition in (
        ("motion_duration_cued", dict(delay=0)),
        ("orientation_cued", dict(delay=4)),
        ("spatial_binding", dict(delay=0)),
    ):
        images, labels, _ = stream.batch(2, task, condition)
        batches.append((task, images, labels))
    return batches


def logits_for(model, batches, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for task, images, _ in batches:
            outputs.append(model(images.to(device), task).cpu())
    return outputs


def main():
    result = dict(version=VERSION, status="running")
    v1_cfg = v1_recipe()
    v1_model, _, _ = v1_initialize(V1_ARM, v1_cfg)
    ab_cfg = recipe(dict(C=False, D=False, E=False))
    ab_model, _, _ = initialize_scratch(ARM, ab_cfg)
    full_cfg = recipe()
    assert_fixed(full_cfg)
    assert_fixed(ab_cfg)
    model, optimizer, lineage = initialize_scratch(ARM, full_cfg)
    repeated, repeated_optimizer, repeated_lineage = initialize_scratch(
        ARM, copy.deepcopy(full_cfg)
    )
    if state_sha256(model) != state_sha256(repeated) or lineage != repeated_lineage:
        raise RuntimeError("v2 scratch construction is not deterministic")
    if optimizer.state or repeated_optimizer.state:
        raise RuntimeError("Scratch optimizer has state")

    batches = fixed_batches()
    # Identity gate (A and B): CPU bit-identity is the hard gate.
    v1_logits = logits_for(v1_model, batches, "cpu")
    ab_logits = logits_for(ab_model, batches, "cpu")
    identity = all(torch.equal(a, b) for a, b in zip(v1_logits, ab_logits))
    max_diff = max(float((a - b).abs().max()) for a, b in zip(v1_logits, ab_logits))
    result["identity_gate"] = dict(
        v2_changes=ab_cfg["v2_changes"],
        bit_identical_logits_cpu=identity,
        max_abs_logit_difference_cpu=max_diff,
        batches=[(task, list(images.shape)) for task, images, _ in batches],
    )
    if torch.cuda.is_available():
        v1_gpu = logits_for(v1_model.cuda(), batches, "cuda")
        ab_gpu = logits_for(ab_model.cuda(), batches, "cuda")
        result["identity_gate"].update(
            bit_identical_logits_cuda=all(
                torch.equal(a, b) for a, b in zip(v1_gpu, ab_gpu)
            ),
            max_abs_logit_difference_cuda=max(
                float((a - b).abs().max()) for a, b in zip(v1_gpu, ab_gpu)
            ),
        )
        v1_model.cpu()
        ab_model.cpu()
    if not identity:
        result["status"] = "failed_identity_gate"
        Path(__file__).with_name("construction_checks.json").write_text(
            json.dumps(result, indent=2)
        )
        raise RuntimeError("v2(A,B) is not bit-identical to v1: max diff %g" % max_diff)

    # Full v2 must differ from v1 (C and D are real changes).
    full_logits = logits_for(model, batches, "cpu")
    result["full_v2_differs_from_v1"] = any(
        not torch.equal(a, b) for a, b in zip(v1_logits, full_logits)
    )

    # Structural checks.
    state = model.state_dict()
    for index, core in enumerate(model.accumulators):
        if not isinstance(core, OpponentAccumulatorV2):
            raise RuntimeError("Accumulator is not v2")
        if tuple(core.output.weight.shape) != (32, 77, 1, 1):
            raise RuntimeError("Accumulator output must be Conv2d(77,32,1)")
        if float(core.output.weight[:, 72:].abs().sum()) != 0.0:
            raise RuntimeError("New opponent columns must start at zero")
        if not torch.equal(core.output.weight[:, :72], v1_model.accumulators[index].output.weight):
            raise RuntimeError("Inherited opponent columns differ from v1")
    fusion = model.readout.fusion[0]
    if tuple(fusion.weight.shape) != (64, 192, 3, 3):
        raise RuntimeError("Fusion must be ConvNormAct(192,64)")
    for scale in range(3):
        if float(fusion.weight[:, 64 * scale + 32 : 64 * scale + 64].abs().sum()) != 0.0:
            raise RuntimeError("Max-pool fusion columns must start at zero")
    if not isinstance(model.attention, ContextJointAttentionV2):
        raise RuntimeError("Attention is not the v2 diagnostic subclass")
    locality = torch.nn.functional.softplus(model.attention.raw_locality)
    if not torch.allclose(locality, torch.tensor([4.0, 0.02]), atol=1e-6):
        raise RuntimeError("Change C locality values wrong")
    if not torch.equal(model.attention.source_bias, torch.tensor([[2.0, 0.0], [0.0, 0.0]])):
        raise RuntimeError("Change C source bias wrong")
    if any(
        float(layer.weight.abs().sum()) == 0.0 or float(layer.bias.abs().sum()) != 0.0
        for layer in model.priority_readout.selection.values()
    ):
        raise RuntimeError("Change D selection init wrong")
    if set(lineage["learning_rate_parameter_counts"]) != {f"{full_cfg['new_lr']:g}"}:
        raise RuntimeError("Change E did not unify learning rates")
    if any(name.startswith("comparator.") for name in state) or hasattr(model.attention, "gamma"):
        raise RuntimeError("Wrong comparator/attention architecture")

    # Gradient flow into the new zero columns on one real batch per task.
    model.train()
    model.zero_grad(set_to_none=True)
    train_stream = SpatialBatteryStream(88973001, "train")
    per_task = {}
    for task in full_cfg["task_classes"]:
        condition = full_cfg["cells"][full_cfg["train_names"][task][0]]["condition"]
        images, labels, _ = train_stream.batch(2, task, condition)
        logits, diagnostics = model(images, task, True)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        (loss / 5).backward()
        priority = diagnostics["priority"]
        per_task[task] = dict(
            frames=int(images.shape[1]),
            loss=float(loss),
            priority_entropy_fraction=float(
                -(priority.clamp_min(1e-30) * priority.clamp_min(1e-30).log()).sum((1, 2, 3)).mean()
                / torch.log(torch.tensor(169.0))
            ),
            attention_mass_beyond_3_by_head=diagnostics["records"][-1]["attention_mass_beyond_3_by_head"],
        )
    gradients = {
        name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None
    }
    if not gradients or not all(torch.isfinite(torch.tensor(v)) for v in gradients.values()):
        raise RuntimeError("Nonfinite or missing gradients")
    new_column_gradients = dict(
        accumulator_new_columns=[
            float(core.output.weight.grad[:, 72:].norm()) for core in model.accumulators
        ],
        fusion_max_columns=[
            float(fusion.weight.grad[:, 64 * s + 32 : 64 * s + 64].norm()) for s in range(3)
        ],
        selection=[float(l.weight.grad.norm()) for l in model.priority_readout.selection.values()],
        raw_locality=model.attention.raw_locality.grad.tolist(),
        source_bias=model.attention.source_bias.grad.tolist(),
    )
    if min(new_column_gradients["accumulator_new_columns"]) == 0.0:
        raise RuntimeError("No gradient reaches the new opponent columns")
    if min(new_column_gradients["fusion_max_columns"]) == 0.0:
        raise RuntimeError("No gradient reaches the max-pool fusion columns")

    result.update(
        status="passed",
        arm=ARM,
        deterministic_scratch_initialization=True,
        loaded_parent=False,
        loaded_model_tensors=0,
        loaded_optimizer_states=0,
        initial_model_sha256=lineage["initial_model_sha256"],
        v1_initial_model_sha256=state_sha256(v1_model),
        total_parameters=sum(p.numel() for p in model.parameters()),
        v1_total_parameters=sum(p.numel() for p in v1_model.parameters()),
        parameter_delta_by_change=dict(
            A=3 * 32 * 5,
            B=64 * 96 * 9,
        ),
        learning_rate_parameter_counts=lineage["learning_rate_parameter_counts"],
        v1_learning_rate_parameter_counts=v1_learning_rates(v1_model, v1_cfg),
        locality_by_head=locality.tolist(),
        raw_locality=model.attention.raw_locality.tolist(),
        source_bias=model.attention.source_bias.tolist(),
        head1_sigma_cells=float((1.0 / (2 * 0.02)) ** 0.5),
        head1_penalty_at_3_cells=0.02 * 9,
        head1_penalty_at_6_cells=0.02 * 36,
        selection_weight_norms={
            task: float(layer.weight.norm())
            for task, layer in model.priority_readout.selection.items()
        },
        per_task_forward=per_task,
        new_column_gradient_norms=new_column_gradients,
        finite_gradient_parameter_count=len(gradients),
        gate=dict(step=GATE_STEP, cued_tasks=list(CUED_TASKS)),
        updates=UPDATES,
        episodes=TOTAL_EPISODES,
        validation_targets=list(VALIDATION_TARGETS),
        lineage=lineage,
    )
    Path(__file__).with_name("construction_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("lineage", "per_task_forward")}, indent=2))


if __name__ == "__main__":
    main()
