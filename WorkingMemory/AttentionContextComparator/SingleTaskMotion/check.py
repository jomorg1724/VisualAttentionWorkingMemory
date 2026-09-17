"""Essential scratch-lineage, canonical-task, and finite-gradient checks."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from WorkingMemory.AttentionContextComparator.model import (
    ARM,
    VERSION,
    initialize_scratch,
    state_sha256,
)
from WorkingMemory.AttentionContextComparator.SingleTaskMotion.protocol import (
    BATCH_SIZE,
    DELAYS,
    TASK,
    TOTAL_EPISODES,
    UPDATES,
    recipe,
    training_name,
)
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream


def main():
    cfg = recipe()
    model, optimizer, lineage = initialize_scratch(ARM, cfg)
    repeated, repeated_optimizer, repeated_lineage = initialize_scratch(
        ARM, copy.deepcopy(cfg)
    )
    if (
        VERSION != "attention_context_comparator_scratch_v1"
        or lineage != repeated_lineage
        or state_sha256(model) != state_sha256(repeated)
        or optimizer.state
        or repeated_optimizer.state
    ):
        raise RuntimeError("Scratch construction is not exact and deterministic")
    if hasattr(model, "comparator") or hasattr(model.attention, "gamma"):
        raise RuntimeError("Wrong comparator architecture")

    schedule = [training_name(cfg, step) for step in range(UPDATES)]
    allowed = set(cfg["train_names"][TASK])
    if set(schedule) != allowed:
        raise RuntimeError("Motion scheduler emitted a noncanonical condition")
    for offset in range(0, UPDATES, len(DELAYS)):
        if set(schedule[offset : offset + len(DELAYS)]) != allowed:
            raise RuntimeError("Each scheduler block must cover all delays once")

    stream = SpatialBatteryStream(cfg["train_seed"], "train")
    images, labels, metadata = stream.batch(1, TASK, dict(delay=0))
    meta = metadata[0]
    if (
        images.shape != (1, 11, 3, 100, 100)
        or labels.shape != (1,)
        or meta["motion_transitions"] != 8
        or meta["dots_per_patch"] != 32
        or meta["cue_frames"] != list(range(10))
        or meta["moving_frames"] != list(range(2, 10))
        or meta["report_frame"] != 10
        or len(meta["directions_by_patch"]) != 4
    ):
        raise RuntimeError("Canonical four-patch motion renderer changed")

    model.train()
    logits, diagnostics = model(images, TASK, True)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if (
        logits.shape != (1, 4)
        or not torch.isfinite(logits).all()
        or not gradients
        or not all(np.isfinite(value) for value in gradients.values())
        or diagnostics["attention_context"].shape != (1, 64, 13, 13)
        or diagnostics["priority"].shape != (1, 1, 13, 13)
    ):
        raise RuntimeError("Nonfinite or malformed full motion forward/backward")
    other_head_gradients = [
        name
        for name in gradients
        if (
            name.startswith("priority_readout.selection.")
            or name.startswith("priority_readout.evidence.")
        )
        and f".{TASK}." not in name
    ]
    if other_head_gradients:
        raise RuntimeError("A non-motion task head received gradients")

    result = dict(
        status="passed",
        architecture_version=VERSION,
        exact_deterministic_scratch_initialization=True,
        initial_model_sha256=lineage["initial_model_sha256"],
        loaded_parent=False,
        loaded_model_tensors=0,
        inherited_optimizer_states=0,
        old_comparator_absent=True,
        prospective_gamma_absent=True,
        only_training_task=TASK,
        classes=4,
        canonical_delays=list(DELAYS),
        canonical_renderer=dict(
            shape=list(images.shape),
            patches=4,
            cue_frames=meta["cue_frames"],
            moving_frames=meta["moving_frames"],
            report_frame=meta["report_frame"],
            motion_transitions=meta["motion_transitions"],
            dots_per_patch=meta["dots_per_patch"],
        ),
        full_bptt_finite=True,
        finite_gradient_parameter_count=len(gradients),
        non_motion_head_gradients=[],
        attention_context_shape=list(diagnostics["attention_context"].shape),
        priority_shape=list(diagnostics["priority"].shape),
        scheduler_complete_delay_block=True,
        updates=UPDATES,
        batch_size=BATCH_SIZE,
        episodes=TOTAL_EPISODES,
        lineage=lineage,
    )
    output = Path(__file__).with_name("construction_checks.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

