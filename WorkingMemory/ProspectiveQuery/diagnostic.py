"""Frozen paired-cue diagnostic and optional split pre-pooling sensory probe."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from PreAttentiveVision.train import sha
from WorkingMemory.SpatialTaskBattery.BiasedTraining.model import BiasedMemory, VERSION
from WorkingMemory.SpatialTaskBattery.stimuli import CENTERS, SpatialBatteryStream

BASELINE = (
    ROOT
    / "WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/"
    "biased_20260913_194506/retrieved/biased_results/spatial_biased/"
    "training/checkpoint_010000.pt"
)
BASELINE_SHA256 = "35281f264131e01678ab5725a5b66816d47302583f78b9751f775f18adf22942"


def load_frozen(path, device):
    digest = sha(path)
    if digest != BASELINE_SHA256:
        raise RuntimeError("Expected frozen original-bias five-task checkpoint10000")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("version") != VERSION or checkpoint.get("step") != 10000:
        raise RuntimeError("Baseline checkpoint version/step mismatch")
    model = BiasedMemory(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, digest


def annulus(center):
    yy, xx = np.mgrid[:100, :100]
    x, y = center
    return np.abs(np.sqrt((xx - x) ** 2 + (yy - y) ** 2) - 14) < 0.8


RINGS = [torch.from_numpy(annulus(center)) for center in CENTERS]


def changed_cue_pair(stream):
    """Return identical motion evidence with two valid target-ring assignments."""
    for _ in range(128):
        images, labels, metadata = stream.batch(
            1, "motion_duration_cued", {"delay": 0}
        )
        meta = metadata[0]
        old_target = int(meta["target_location"])
        winners = list(map(int, meta["winner_by_patch"]))
        alternatives = [
            index
            for index, winner in enumerate(winners)
            if index != old_target and winner != winners[old_target]
        ]
        if not alternatives:
            continue
        new_target = alternatives[0]
        changed = images.clone()
        union = RINGS[old_target] | RINGS[new_target]
        for frame in meta["cue_frames"]:
            changed[0, frame, :, RINGS[old_target]] = 0.5
            changed[0, frame, :, RINGS[new_target]] = 0.95
        difference = (images - changed).abs()
        if float(difference[:, :, :, ~union].max()) != 0.0:
            raise AssertionError("Counterfactual changed pixels outside cue rings")
        if float(difference[:, :, :, union].max()) == 0.0:
            raise AssertionError("Counterfactual failed to change cue pixels")
        return dict(
            original=images,
            counterfactual=changed,
            original_label=int(labels[0]),
            counterfactual_label=winners[new_target],
            original_target=old_target,
            counterfactual_target=new_target,
            metadata=meta,
            evidence_sha256=hashlib.sha256(
                images[:, :, :, ~union].contiguous().numpy().tobytes()
            ).hexdigest(),
        )
    raise RuntimeError("Could not draw a cue pair with different valid answers")


def attention_weights(module, field, old):
    batch = field.shape[0]
    visual = field.flatten(2).transpose(1, 2)
    memory = old.flatten(2).transpose(1, 2)
    q = module.query(module.query_norm(memory) + module.position + module.source[1])
    raw = torch.cat((visual, memory), 1)
    identities = torch.cat(
        (module.position + module.source[0], module.position + module.source[1]), 0
    )
    k = module.key(module.key_norm(raw) + identities)
    v = module.value(raw)
    q = q.reshape(batch, 169, 2, 32).transpose(1, 2)
    k = k.reshape(batch, 338, 2, 32).transpose(1, 2)
    v = v.reshape(batch, 338, 2, 32).transpose(1, 2)
    logits = q @ k.transpose(-1, -2) / math.sqrt(32)
    bias = (
        module.source_bias.repeat_interleave(169, dim=1)[:, None, :]
        - F.softplus(module.raw_locality)[:, None, None] * module.distance_squared
    )
    weights = (logits + bias[None]).softmax(-1)
    attended = module.output(
        (weights @ v).transpose(1, 2).reshape(batch, 169, 64)
    )
    return attended.transpose(1, 2).reshape(batch, 64, 13, 13), weights


@torch.no_grad()
def frozen_forward(model, images, task):
    traces = ()
    state = None
    routes = []
    for frame in range(images.shape[1]):
        result = model._sensory(images[:, frame], *traces)
        field, traces = result[0], result[1:]
        old = torch.zeros_like(field) if state is None else state[0]
        attended, weights = attention_weights(model.attention, field, old)
        routes.append(weights.cpu())
        z = model.memory_input(attended)
        local = model.comparator(torch.cat((old, field), 1))
        comparison = torch.cat((local.mean((2, 3)), local.amax((2, 3))), 1)
        rates, state, _ = model.memory(z, state, False)
    sensory = model.readout.trunk(
        torch.cat((field.mean((2, 3)), field.amax((2, 3))), 1)
    )
    retained = torch.cat((rates.mean((2, 3)), rates.amax((2, 3))), 1)
    logits = model.classify(
        sensory
        + model.memory_output(retained)
        + model.comparison_output(comparison),
        task,
    )
    return logits.cpu(), torch.stack(routes, 1)


def paired_diagnostic(model, device, count, seed):
    stream = SpatialBatteryStream(seed, "test")
    rows = []
    for pair_index in range(count):
        pair = changed_cue_pair(stream)
        first_logits, first_routes = frozen_forward(
            model, pair["original"].to(device), "motion_duration_cued"
        )
        second_logits, second_routes = frozen_forward(
            model, pair["counterfactual"].to(device), "motion_duration_cued"
        )
        first_probability = first_logits.softmax(1)[0]
        second_probability = second_logits.softmax(1)[0]
        route_delta = (second_routes - first_routes).abs()
        rows.append(
            dict(
                pair_id=pair_index,
                evidence_sha256=pair["evidence_sha256"],
                original_target=pair["original_target"],
                counterfactual_target=pair["counterfactual_target"],
                original_label=pair["original_label"],
                counterfactual_label=pair["counterfactual_label"],
                original_prediction=int(first_probability.argmax()),
                counterfactual_prediction=int(second_probability.argmax()),
                original_correct_probability=float(
                    first_probability[pair["original_label"]]
                ),
                counterfactual_correct_probability=float(
                    second_probability[pair["counterfactual_label"]]
                ),
                prediction_changed=bool(
                    first_probability.argmax() != second_probability.argmax()
                ),
                routing_mean_abs_change=float(route_delta.mean()),
                routing_max_abs_change=float(route_delta.max()),
                routing_change_by_frame=route_delta.mean((0, 2, 3, 4)).tolist(),
            )
        )
    return dict(
        pair_count=count,
        independent_base_movies=count,
        presentations=2 * count,
        evidence_identity=(
            "Raster-exact outside the union of old/new valid target cue rings; "
            "motion schedules, dots, nuisance draws and all non-cue pixels are paired"
        ),
        prediction_change_rate=float(np.mean([row["prediction_changed"] for row in rows])),
        mean_routing_absolute_change=float(
            np.mean([row["routing_mean_abs_change"] for row in rows])
        ),
        rows=rows,
    )


@torch.no_grad()
def sensory_features(model, device, split, seed, episodes, batch_size=8):
    stream = SpatialBatteryStream(seed, split)
    features = []
    labels = []
    targets = []
    for offset in range(0, episodes, batch_size):
        images, _, metadata = stream.batch(
            min(batch_size, episodes - offset),
            "motion_duration_cued",
            {"delay": 0},
        )
        images = images.to(device)
        traces = ()
        field = None
        for frame in range(10):
            result = model._sensory(images[:, frame], *traces)
            field, traces = result[0], result[1:]
        for batch_index, meta in enumerate(metadata):
            for location, (x, y) in enumerate(CENTERS):
                row = int(round(y / 99 * 12))
                column = int(round(x / 99 * 12))
                patch = field[
                    batch_index,
                    :,
                    max(0, row - 1) : min(13, row + 2),
                    max(0, column - 1) : min(13, column + 2),
                ]
                features.append(patch.mean((1, 2)).cpu().numpy())
                labels.append(int(meta["winner_by_patch"][location]))
                targets.append(location == int(meta["target_location"]))
    return (
        np.asarray(features, np.float64),
        np.asarray(labels, np.int64),
        np.asarray(targets, bool),
    )


def ridge_fit(features, labels, penalty):
    classes = 4
    design = np.c_[features, np.ones(len(features))]
    target = np.eye(classes)[labels]
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[-1, -1] = 0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ target)


def accuracy(features, labels, weights, mask=None):
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    return float(np.mean((np.c_[features, np.ones(len(features))] @ weights).argmax(1)[mask] == labels[mask]))


def split_probe(model, device, sizes):
    split_data = {}
    for split, seed, episodes in zip(
        ("train", "val", "test"), (73173001, 74173001, 75173001), sizes
    ):
        split_data[split] = sensory_features(
            model, device, split, seed, episodes
        )
    train_x, train_y, _ = split_data["train"]
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-8] = 1
    standardized = {
        split: ((values[0] - mean) / scale, values[1], values[2])
        for split, values in split_data.items()
    }
    candidates = []
    for penalty in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        weights = ridge_fit(standardized["train"][0], train_y, penalty)
        candidates.append(
            dict(
                penalty=penalty,
                validation_accuracy=accuracy(
                    standardized["val"][0], standardized["val"][1], weights
                ),
                weights=weights,
            )
        )
    selected = max(candidates, key=lambda row: (row["validation_accuracy"], -row["penalty"]))
    test_x, test_y, test_target = standardized["test"]
    return dict(
        target="per-patch longest-duration direction from final moving-frame H",
        representation=(
            "Frozen 64-channel pre-global-pooling sensory field; local 3x3 mean "
            "at each of four patch centers"
        ),
        split_unit="Independent base episode; no episode appears across splits",
        split_episodes=dict(zip(("train", "val", "test"), sizes)),
        split_patch_examples={name: 4 * size for name, size in zip(("train", "val", "test"), sizes)},
        scaling="Train-only channel mean/std",
        selection="Ridge penalty selected on validation accuracy only",
        selected_penalty=selected["penalty"],
        validation_accuracy=selected["validation_accuracy"],
        test_accuracy=accuracy(test_x, test_y, selected["weights"]),
        test_target_patch_accuracy=accuracy(
            test_x, test_y, selected["weights"], test_target
        ),
        test_foil_patch_accuracy=accuracy(
            test_x, test_y, selected["weights"], ~test_target
        ),
        chance_accuracy=0.25,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("diagnostic_results.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=72173001)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--probe-train", type=int, default=256)
    parser.add_argument("--probe-val", type=int, default=64)
    parser.add_argument("--probe-test", type=int, default=128)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    torch.set_num_threads(2)
    model, checkpoint, before = load_frozen(args.checkpoint, args.device)
    result = dict(
        status="completed",
        analysis_only=True,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=before,
        checkpoint_version=checkpoint["version"],
        checkpoint_step=checkpoint["step"],
        paired=paired_diagnostic(model, args.device, args.pairs, args.seed),
    )
    if not args.skip_probe:
        result["sensory_probe"] = split_probe(
            model,
            args.device,
            (args.probe_train, args.probe_val, args.probe_test),
        )
    if sha(args.checkpoint) != before:
        raise RuntimeError("Frozen checkpoint changed during analysis")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "paired"}, indent=2))


if __name__ == "__main__":
    main()
