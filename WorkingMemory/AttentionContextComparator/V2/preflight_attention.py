"""Pre-flight 2: attention mass versus query-key distance on a trained v1 checkpoint.

Forward-only. Loads the hash-verified local scratch motion-only step-3200
checkpoint (attention_context_comparator_scratch_v1), runs cued motion-duration
D0 validation trials with per-timestep attention capture, and measures, per
head and per source bank, how attention mass falls with cell distance. It also
reports the mass that queries at the cued patch place on the cue-ring cells.
The v2 initialization (change C) is measured on the same trials for contrast.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import sha, write
from WorkingMemory.AttentionContextComparator.model import ARM as V1_ARM
from WorkingMemory.AttentionContextComparator.model import VERSION as V1_VERSION
from WorkingMemory.AttentionContextComparator.model import initialize_scratch as v1_initialize
from WorkingMemory.AttentionContextComparator.V2.model import ARM, initialize_scratch
from WorkingMemory.AttentionContextComparator.V2.protocol import SPATIAL_VAL_SEED, recipe
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

CHECKPOINT = (
    ROOT
    / "WorkingMemory/AttentionContextComparator/SingleTaskMotion/runs/"
    "scratch_motion_20260915_194230/training/checkpoint_003200.pt"
)
TASK = "motion_duration_cued"
N_TRIALS = 64
BATCH = 8
CELL_PX = 100.0 / 13.0
APERTURE_PX = 11.5
RING_PX = 14.0
BINS = ((0.0, 0.0), (0.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 6.5), (6.5, 99.0))


def verified_checkpoint():
    index = CHECKPOINT.parent / "checkpoint_index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    row = next(r for r in rows if r["file"] == CHECKPOINT.name)
    digest = sha(CHECKPOINT)
    if digest != row["sha256"]:
        raise RuntimeError("Checkpoint hash does not match its index")
    saved = torch.load(CHECKPOINT, map_location="cpu")
    if saved["version"] != V1_VERSION or saved["arm"] != V1_ARM or saved["step"] != 3200:
        raise RuntimeError("Unexpected checkpoint identity")
    return saved, digest


def cell_sets(center_xy):
    """Return index sets over the 13x13 grid for the cued patch and its ring cue."""
    x, y = center_xy
    patch, cue = [], []
    for row in range(13):
        for col in range(13):
            cx, cy = (col + 0.5) * CELL_PX, (row + 0.5) * CELL_PX
            distance = math.hypot(cx - x, cy - y)
            index = row * 13 + col
            if distance <= APERTURE_PX:
                patch.append(index)
            elif abs(distance - RING_PX) <= CELL_PX * 0.75:
                cue.append(index)
    return patch, cue


class Capture:
    def __init__(self, attention):
        self.weights = []
        self.handle = attention.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        if module.last_weights is not None:
            self.weights.append(module.last_weights.detach().cpu())

    def reset(self):
        self.weights = []


def measure(model, device, label):
    model.eval()
    capture = Capture(model.attention)
    distance = model.attention.distance_squared.sqrt().cpu()  # [169,338]
    stream = SpatialBatteryStream(SPATIAL_VAL_SEED, "val")
    bin_mass = np.zeros((2, 2, len(BINS)))  # head, source, bin
    beyond3 = np.zeros((2, 2))
    patch_to_cue = np.zeros((2, 2))
    patch_to_patch = np.zeros((2, 2))
    cue_to_patch = np.zeros((2, 2))
    phase_beyond3 = dict(cue_evidence=np.zeros(2), report=np.zeros(2))
    phase_counts = dict(cue_evidence=0, report=0)
    total_weights = 0
    trials = 0
    for offset in range(0, N_TRIALS, BATCH):
        images, labels, metadata = stream.batch(min(BATCH, N_TRIALS - offset), TASK, dict(delay=0))
        capture.reset()
        with torch.enable_grad():
            model(images.to(device), TASK, True)
        weights = torch.stack(capture.weights, 1)  # [B,T,2,169,338]
        batch, timesteps = weights.shape[:2]
        for source, (lo, hi) in enumerate(((0, 169), (169, 338))):
            block = weights[..., lo:hi]  # [B,T,2,169,169]
            d = distance[:, lo:hi]
            for b_index, (lo_d, hi_d) in enumerate(BINS):
                mask = ((d > lo_d) & (d <= hi_d)) if lo_d < hi_d else (d == 0)
                bin_mass[:, source, b_index] += (block * mask).sum(-1).mean((0, 1, 3)).numpy() * batch * timesteps
            beyond3[:, source] += (block * (d > 3)).sum(-1).mean((0, 1, 3)).numpy() * batch * timesteps
            for trial in range(batch):
                patch, cue = cell_sets(metadata[trial]["positions_xy"][metadata[trial]["target_location"]])
                w = block[trial]  # [T,2,169,169]
                patch_to_cue[:, source] += w[:, :, patch][:, :, :, cue].sum(-1).mean((0, 2)).numpy() * timesteps
                patch_to_patch[:, source] += w[:, :, patch][:, :, :, patch].sum(-1).mean((0, 2)).numpy() * timesteps
                cue_to_patch[:, source] += w[:, :, cue][:, :, :, patch].sum(-1).mean((0, 2)).numpy() * timesteps
        far_all = (weights * (distance > 3)).sum(-1).mean(-1)  # [B,T,2]
        phase_beyond3["cue_evidence"] += far_all[:, :10].sum((0, 1)).numpy()
        phase_counts["cue_evidence"] += batch * 10
        phase_beyond3["report"] += far_all[:, 10:].sum((0, 1)).numpy()
        phase_counts["report"] += batch * (timesteps - 10)
        total_weights += batch * timesteps
        trials += batch
    capture.handle.remove()
    locality = torch.nn.functional.softplus(model.attention.raw_locality).detach().cpu().tolist()
    return dict(
        model=label,
        trials=trials,
        query_timesteps=total_weights,
        locality_by_head=locality,
        source_bias=model.attention.source_bias.detach().cpu().tolist(),
        distance_bins_cells=[list(b) for b in BINS],
        mass_by_head_source_bin=(bin_mass / total_weights).tolist(),
        mass_beyond_3_by_head_source=(beyond3 / total_weights).tolist(),
        mass_beyond_3_by_head=(beyond3.sum(1) / total_weights).tolist(),
        mass_beyond_3_by_phase_by_head={
            key: (value / phase_counts[key]).tolist() for key, value in phase_beyond3.items()
        },
        patch_query_to_cue_key_mass_by_head_source=(patch_to_cue / total_weights).tolist(),
        patch_query_to_patch_key_mass_by_head_source=(patch_to_patch / total_weights).tolist(),
        cue_query_to_patch_key_mass_by_head_source=(cue_to_patch / total_weights).tolist(),
        predicted_locality_penalty_at_3_cells_by_head=[9 * l for l in locality],
        predicted_exp_penalty_at_3_cells_by_head=[math.exp(-9 * l) for l in locality],
    )


def main():
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved, digest = verified_checkpoint()
    cfg = saved["config"]
    v1, _, _ = v1_initialize(V1_ARM, cfg)
    v1.load_state_dict(saved["model"], strict=True)
    v1.to(device)
    trained = measure(v1, device, "v1 local scratch motion-only step 3200")
    del v1
    v2, _, _ = initialize_scratch(ARM, recipe())
    v2.to(device)
    initial = measure(v2, device, "v2 initialization (all changes)")
    result = dict(
        status="completed",
        checkpoint=str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        checkpoint_sha256=digest,
        checkpoint_step=3200,
        checkpoint_training="scratch v1, motion_duration_cued only, 25,600 episodes",
        task=TASK,
        condition="D0 (cue present through evidence, report frame 10 blank)",
        evaluation_stream=dict(seed=SPATIAL_VAL_SEED, split="val"),
        geometry=dict(
            cell_px=CELL_PX,
            patch_cells="cell centers within the 11.5 px aperture of the cued patch",
            cue_cells="cell centers within 0.75 cell of the 14 px cue ring, excluding patch cells",
        ),
        trained_checkpoint=trained,
        v2_initialization=initial,
        wall_seconds=time.time() - started,
        device=device,
        interpretation=(
            "Forward-only measurement; no training. Mass beyond 3 cells is the "
            "fraction of each query's attention placed on keys more than 3 grid "
            "cells away, averaged over queries, timesteps and trials. Sources are "
            "the current sensory bank (0) and the previous memory bank (1)."
        ),
    )
    out = Path(__file__).with_name("preflight") / "preflight_attention.json"
    write(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("trained_checkpoint", "v2_initialization")}, indent=2))
    for block in (trained, initial):
        print(block["model"])
        print("  locality", block["locality_by_head"], "source_bias", block["source_bias"])
        print("  mass beyond 3 by head", block["mass_beyond_3_by_head"])
        print("  mass beyond 3 by phase", block["mass_beyond_3_by_phase_by_head"])
        print("  patch->cue by head/source", block["patch_query_to_cue_key_mass_by_head_source"])
        print("  bins by head/source", json.dumps(block["mass_by_head_source_bin"]))


if __name__ == "__main__":
    main()
