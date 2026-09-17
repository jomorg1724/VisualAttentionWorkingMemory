"""Bounded frozen-feature extraction and paired diagnostic-probe fitting."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from WorkingMemory.SpatialPriorityReadout.LocalDiagnostic.model import (
    PooledProbe,
    SpatialPriorityProbe,
    original_logits_from_fields,
    parameter_count,
    target_region_mask,
    terminal_fields,
)
from WorkingMemory.SpatialPriorityReadout.LocalDiagnostic.protocol import (
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINT,
    CHECKPOINT_SHA256,
    CHECKPOINT_STEP,
    CHECKPOINT_VERSION,
    CLIP,
    CPU_THREADS,
    DELAYS,
    LEARNING_RATE,
    LOOKS,
    MODEL_SEED,
    PRODUCTION_PER_DELAY,
    SMOKE_PER_DELAY,
    SPLIT_SEEDS,
    TASK,
    WEIGHT_DECAY,
)
from WorkingMemory.SpatialTaskBattery.BiasedTraining.model import BiasedMemory
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def check_deadline(deadline: float, reserve: float = 10.0):
    if time.time() > deadline - reserve:
        raise TimeoutError("Local diagnostic reached its absolute wall deadline")


def load_frozen(device):
    checkpoint_path = ROOT / CHECKPOINT
    if sha(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if (
        payload["version"] != CHECKPOINT_VERSION
        or payload["step"] != CHECKPOINT_STEP
    ):
        raise RuntimeError("Unexpected frozen checkpoint lineage")
    model = BiasedMemory(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen model is not in eval/frozen mode")
    return model, payload


def extract_split(model, split, n_per_delay, out, device, deadline):
    started = time.time()
    features = []
    labels = []
    delays = []
    targets = []
    base_ids = []
    trial_ids = []
    references = {}
    forward_equivalence = []
    batch_size = 8
    for delay in DELAYS:
        stream = SpatialBatteryStream(SPLIT_SEEDS[split], split)
        offset = 0
        while offset < n_per_delay:
            check_deadline(deadline, 20)
            size = min(batch_size, n_per_delay - offset)
            images, y, metadata = stream.batch(size, TASK, {"delay": delay})
            with torch.no_grad():
                images_gpu = images.to(device)
                field_batch = terminal_fields(model, images_gpu)
                if offset == 0:
                    direct = model(images_gpu, TASK)
                    reconstructed = original_logits_from_fields(
                        model, field_batch, TASK
                    )
                    difference = float((direct - reconstructed).abs().max())
                    if difference > 2e-6:
                        raise RuntimeError(
                            f"Terminal extraction changed logits: {difference}"
                        )
                    forward_equivalence.append(difference)
            features.append(field_batch.cpu())
            labels.append(y)
            delays.extend([delay] * size)
            for local_index, metadata_row in enumerate(metadata):
                ordinal = offset + local_index
                base_id = f"{split}/{SPLIT_SEEDS[split]}/{ordinal}"
                target = int(metadata_row["target_location"])
                signature = (
                    int(metadata_row["label"]),
                    target,
                    metadata_row["directions_by_patch"],
                    metadata_row["winner_by_patch"],
                    metadata_row["step_pixels"],
                )
                if delay == DELAYS[0]:
                    references[ordinal] = signature
                elif references.get(ordinal) != signature:
                    raise RuntimeError("Across-delay base movie mismatch")
                targets.append(target)
                base_ids.append(base_id)
                trial_ids.append(f"{base_id}/D{delay}")
            offset += size
    value = {
        "features": torch.cat(features).float(),
        "labels": torch.cat(labels).long(),
        "delays": torch.tensor(delays, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "base_ids": base_ids,
        "trial_ids": trial_ids,
    }
    if not torch.isfinite(value["features"]).all():
        raise FloatingPointError("Nonfinite cached field")
    cache_path = out / f"features_{split}.pt"
    torch.save(value, cache_path)
    receipt = {
        "split": split,
        "seed": SPLIT_SEEDS[split],
        "independent_base_episodes": n_per_delay,
        "delay_presentations": n_per_delay * len(DELAYS),
        "per_delay": n_per_delay,
        "shape": list(value["features"].shape),
        "dtype": str(value["features"].dtype),
        "feature_tensor_sha256": tensor_digest(value["features"]),
        "cache_sha256": sha(cache_path),
        "paired_motion_movie_metadata_across_delays": True,
        "max_original_logit_reconstruction_error": max(forward_equivalence),
        "seconds": time.time() - started,
    }
    return receipt


def train_scaling(train):
    x = train["features"].double()
    mean = x.mean((0, 2, 3)).float()
    variance = x.var((0, 2, 3), unbiased=False).float()
    std = variance.sqrt().clamp_min(1e-6)
    return mean, std


def normalize(x, mean, std, device):
    return (
        x.to(device, non_blocking=True) - mean[None, :, None, None]
    ) / std[None, :, None, None]


def classification(y, probabilities):
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pred = probabilities.argmax(1)
    matrix = confusion_matrix(y, pred, labels=np.arange(4))
    recalls = np.divide(
        np.diag(matrix),
        matrix.sum(1),
        out=np.full(4, np.nan),
        where=matrix.sum(1) > 0,
    )
    try:
        auc = float(
            roc_auc_score(
                y, probabilities, labels=np.arange(4), multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = None
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_ovr_auc": auc,
        "confusion_true_rows_pred_columns": matrix.tolist(),
        "class_counts": np.bincount(y, minlength=4).tolist(),
        "prediction_counts": np.bincount(pred, minlength=4).tolist(),
    }


def evaluate_probe(model, data, mean, std, device, maps=False):
    model.eval()
    probabilities = []
    priority = []
    with torch.no_grad():
        for offset in range(0, len(data["labels"]), BATCH_SIZE):
            x = normalize(
                data["features"][offset : offset + BATCH_SIZE], mean, std, device
            )
            if maps:
                logits, values, _ = model(x, True)
                priority.append(values.cpu())
            else:
                logits = model(x)
            probabilities.append(logits.softmax(1).cpu())
    result = torch.cat(probabilities).numpy()
    maps_result = torch.cat(priority).numpy() if priority else None
    return result, maps_result


def selection_rank(probabilities, data):
    per_delay = {}
    for delay in DELAYS:
        rows = data["delays"].numpy() == delay
        per_delay[str(delay)] = classification(
            data["labels"].numpy()[rows], probabilities[rows]
        )
    mean_ba = float(
        np.mean([value["balanced_accuracy"] for value in per_delay.values()])
    )
    mean_auc = float(
        np.mean([value["macro_ovr_auc"] for value in per_delay.values()])
    )
    return (mean_ba, mean_auc), per_delay


def fit_probe(name, model, train, val, mean, std, out, device, deadline):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    curves = []
    best = None
    updates = 0
    n = len(train["labels"])
    for epoch in range(1, max(LOOKS) + 1):
        check_deadline(deadline, 60)
        model.train()
        generator = torch.Generator().manual_seed(MODEL_SEED + epoch)
        order = torch.randperm(n, generator=generator)
        losses = []
        for offset in range(0, n, BATCH_SIZE):
            indices = order[offset : offset + BATCH_SIZE]
            x = normalize(train["features"][indices], mean, std, device)
            y = train["labels"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite probe loss/gradient")
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        if epoch in LOOKS:
            probabilities, _ = evaluate_probe(
                model, val, mean, std, device, name == "spatial"
            )
            rank, per_delay = selection_rank(probabilities, val)
            record = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "training_presentations": epoch * n,
                "mean_loss": float(np.mean(losses)),
                "selection_rank_mean_delay_ba_then_auc": list(rank),
                "per_delay": per_delay,
            }
            curves.append(record)
            candidate = (rank[0], rank[1], -epoch)
            if best is None or candidate > best["candidate"]:
                best = {
                    "candidate": candidate,
                    "epoch": epoch,
                    "optimizer_updates": updates,
                    "training_presentations": epoch * n,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
    model.load_state_dict(best["state"])
    probe_path = out / f"{name}_selected.pt"
    torch.save(
        {
            "probe": name,
            "state": best["state"],
            "selected_epoch": best["epoch"],
            "optimizer_updates_at_selection": best["optimizer_updates"],
            "training_presentations_at_selection": best["training_presentations"],
            "parameter_count": parameter_count(model),
        },
        probe_path,
    )
    del best["state"]
    best["candidate"] = list(best["candidate"])
    return model, {
        "probe": name,
        "parameters": parameter_count(model),
        "validation_looks": curves,
        "selected": best,
        "terminal_epoch": max(LOOKS),
        "total_optimizer_updates": updates,
        "total_training_presentations": max(LOOKS) * n,
        "checkpoint": str(probe_path.relative_to(ROOT)),
        "checkpoint_sha256": sha(probe_path),
    }


def stratified_base_bootstrap(
    labels, base_ids, probabilities_a, probabilities_b, replicates, seed
):
    labels = np.asarray(labels)
    base_ids = np.asarray(base_ids)
    unique = np.unique(base_ids)
    base_label = {base: int(labels[np.flatnonzero(base_ids == base)[0]]) for base in unique}
    strata = {
        label: np.array([base for base in unique if base_label[base] == label])
        for label in range(4)
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        chosen = []
        for label in range(4):
            values = strata[label]
            chosen.extend(rng.choice(values, len(values), replace=True).tolist())
        indices = np.concatenate(
            [np.flatnonzero(base_ids == base) for base in chosen]
        )
        metric_a = classification(labels[indices], probabilities_a[indices])
        metric_b = classification(labels[indices], probabilities_b[indices])
        if metric_a["macro_ovr_auc"] is None or metric_b["macro_ovr_auc"] is None:
            continue
        samples.append(
            [
                metric_a["accuracy"],
                metric_a["balanced_accuracy"],
                metric_a["macro_ovr_auc"],
                metric_b["accuracy"],
                metric_b["balanced_accuracy"],
                metric_b["macro_ovr_auc"],
            ]
        )
    values = np.asarray(samples)
    names = ("accuracy", "balanced_accuracy", "macro_ovr_auc")
    result = {"valid_replicates": int(len(values))}
    for index, name in enumerate(names):
        result[f"pooled_{name}_ci95"] = np.quantile(
            values[:, index], [0.025, 0.975]
        ).tolist()
        result[f"spatial_{name}_ci95"] = np.quantile(
            values[:, index + 3], [0.025, 0.975]
        ).tolist()
        result[f"spatial_minus_pooled_{name}_ci95"] = np.quantile(
            values[:, index + 3] - values[:, index], [0.025, 0.975]
        ).tolist()
    return result


def alignment(priority, targets, base_ids, delays):
    priorities = torch.from_numpy(priority)
    target_tensor = torch.as_tensor(targets)
    mask = target_region_mask(target_tensor)
    mass = (priorities * mask).sum((1, 2)).numpy()
    peaks = priorities.flatten(1).argmax(1)
    hit = mask.flatten(1).gather(1, peaks[:, None]).squeeze(1).numpy().astype(float)
    result = {
        "definition": (
            "Priority mass and argmax hit inside an evaluation-only 3x3 grid "
            "neighborhood around the linearly projected target-patch center."
        ),
        "uniform_expected_mass_and_hit": 9 / 169,
        "overall_mass": float(mass.mean()),
        "overall_peak_hit_rate": float(hit.mean()),
        "per_delay": {},
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + 300)
    ids = np.asarray(base_ids)
    unique = np.unique(ids)
    boot_mass = []
    boot_hit = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(ids == value) for value in chosen])
        boot_mass.append(mass[indices].mean())
        boot_hit.append(hit[indices].mean())
    result["overall_mass_ci95"] = np.quantile(
        boot_mass, [0.025, 0.975]
    ).tolist()
    result["overall_peak_hit_rate_ci95"] = np.quantile(
        boot_hit, [0.025, 0.975]
    ).tolist()
    delays_array = np.asarray(delays)
    for delay in DELAYS:
        rows = delays_array == delay
        result["per_delay"][str(delay)] = {
            "n": int(rows.sum()),
            "mass": float(mass[rows].mean()),
            "peak_hit_rate": float(hit[rows].mean()),
        }
    return result


def source_hashes():
    paths = [
        "WorkingMemory/SpatialPriorityReadout/model.py",
        "WorkingMemory/SpatialPriorityReadout/protocol.py",
        "WorkingMemory/SpatialPriorityReadout/check_model.py",
        "WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/model.py",
        "WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/protocol.py",
        "WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/run.py",
        "WorkingMemory/SpatialTaskBattery/BiasedTraining/model.py",
        "WorkingMemory/SpatialTaskBattery/stimuli.py",
        "WorkingMemory/PreUpdateAttention/model.py",
        "WorkingMemory/SpatialComparison/model.py",
    ]
    return {path: sha(ROOT / path) for path in paths}


def make_report(result):
    pooled = result["test"]["pooled"]
    spatial = result["test"]["spatial"]
    comparison = result["test"]["paired"]
    lines = [
        "# Complementary local frozen-core readout diagnostic",
        "",
        "**Status: complete; analysis-only/post-hoc.** No deployed model weight was "
        "updated and the active cloud run was not queried or modified.",
        "",
        "## Design and exposure",
        "",
        f"The frozen source was motion-only step {CHECKPOINT_STEP}, SHA256 "
        f"`{CHECKPOINT_SHA256}`. It had received 30,400 cumulative motion "
        "episodes. Encoder, opponent traces, original pre-update attention, "
        "spatial E/I memory and comparator stayed in eval mode. Both fresh "
        "probes consumed the identical cached float32 `H_T/R_T/C_T` fields.",
        "",
        f"Independent base episodes per delay were "
        f"{result['samples']['train']}/{result['samples']['val']}/"
        f"{result['samples']['test']} for train/validation/test, using seeds "
        f"{SPLIT_SEEDS['train']}/{SPLIT_SEEDS['val']}/{SPLIT_SEEDS['test']}. "
        "The four delay presentations of each base movie were grouped. "
        f"Probe A has {result['probes']['pooled']['parameters']:,} parameters; "
        f"probe B has {result['probes']['spatial']['parameters']:,} "
        "parameters (70, 0.094%, more).",
        "",
        "Both probes selected epoch 20 at validation-only look 3: 640 "
        "optimizer updates and 40,960 repeated training presentations each. "
        "Both were run through the fixed epoch-40 validation schedule (1,280 "
        "updates / 81,920 presentations each) before restoring the selected "
        "states. These presentations reuse 512 independent training base "
        "episodes across epochs and four delays; they are not fresh episodes.",
        "",
        "## Once-only held-out test",
        "",
        "| Probe | Accuracy | Balanced accuracy (95% CI) | Macro OVR AUC (95% CI) |",
        "|---|---:|---:|---:|",
        f"| Pooled | {pooled['overall']['accuracy']:.2%} | "
        f"{pooled['overall']['balanced_accuracy']:.2%} "
        f"[{comparison['overall']['bootstrap']['pooled_balanced_accuracy_ci95'][0]:.2%}, "
        f"{comparison['overall']['bootstrap']['pooled_balanced_accuracy_ci95'][1]:.2%}] | "
        f"{pooled['overall']['macro_ovr_auc']:.4f} "
        f"[{comparison['overall']['bootstrap']['pooled_macro_ovr_auc_ci95'][0]:.4f}, "
        f"{comparison['overall']['bootstrap']['pooled_macro_ovr_auc_ci95'][1]:.4f}] |",
        f"| Spatial priority | {spatial['overall']['accuracy']:.2%} | "
        f"{spatial['overall']['balanced_accuracy']:.2%} "
        f"[{comparison['overall']['bootstrap']['spatial_balanced_accuracy_ci95'][0]:.2%}, "
        f"{comparison['overall']['bootstrap']['spatial_balanced_accuracy_ci95'][1]:.2%}] | "
        f"{spatial['overall']['macro_ovr_auc']:.4f} "
        f"[{comparison['overall']['bootstrap']['spatial_macro_ovr_auc_ci95'][0]:.4f}, "
        f"{comparison['overall']['bootstrap']['spatial_macro_ovr_auc_ci95'][1]:.4f}] |",
        "",
        f"Paired spatial-minus-pooled BA is "
        f"{comparison['overall']['point']['balanced_accuracy']:+.2%}, "
        f"95% grouped bootstrap CI "
        f"[{comparison['overall']['bootstrap']['spatial_minus_pooled_balanced_accuracy_ci95'][0]:+.2%}, "
        f"{comparison['overall']['bootstrap']['spatial_minus_pooled_balanced_accuracy_ci95'][1]:+.2%}].",
        "",
        "| Delay | Pooled BA | Spatial BA | Spatial−pooled BA (95% CI) | "
        "Pooled / spatial AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for delay in DELAYS:
        key = str(delay)
        pair = comparison["per_delay"][key]
        ci = pair["bootstrap"]["spatial_minus_pooled_balanced_accuracy_ci95"]
        lines.append(
            f"| D{delay} | {pooled['per_delay'][key]['balanced_accuracy']:.2%} | "
            f"{spatial['per_delay'][key]['balanced_accuracy']:.2%} | "
            f"{pair['point']['balanced_accuracy']:+.2%} "
            f"[{ci[0]:+.2%}, {ci[1]:+.2%}] | "
            f"{pooled['per_delay'][key]['macro_ovr_auc']:.4f} / "
            f"{spatial['per_delay'][key]['macro_ovr_auc']:.4f} |"
        )
    align = result["priority_alignment"]
    lines += [
        "",
        "## Priority-map alignment",
        "",
        f"Mean target-region mass was {align['overall_mass']:.2%} "
        f"(95% CI {align['overall_mass_ci95'][0]:.2%}–"
        f"{align['overall_mass_ci95'][1]:.2%}); peak hit rate was "
        f"{align['overall_peak_hit_rate']:.2%} "
        f"(95% CI {align['overall_peak_hit_rate_ci95'][0]:.2%}–"
        f"{align['overall_peak_hit_rate_ci95'][1]:.2%}). Uniform-map "
        "expectation is 5.33%. Target coordinates were used only for this "
        "post-test metric, not training. Maps are exposed in "
        "`priority_maps_test.npz` and `predictions_test.jsonl`.",
        "",
        "| Delay | Target-region mass | Peak-hit rate |",
        "|---|---:|---:|",
        *[
            f"| D{delay} | {align['per_delay'][str(delay)]['mass']:.2%} | "
            f"{align['per_delay'][str(delay)]['peak_hit_rate']:.2%} |"
            for delay in DELAYS
        ],
        "",
        "## Interpretation and limits",
        "",
        "This is one frozen checkpoint, one stimulus generator and one probe "
        "initialization. A post-hoc probe can demonstrate decodable information "
        "for its function class, not that the deployed output uses it, that the "
        "map is causal attention, or that end-to-end cloud training will obtain "
        "the same result. The pooled comparator is capacity-matched within "
        "0.094%, not architecturally identical in inductive bias. The 3×3 cue "
        "region is an approximate feature-grid projection rather than pixel "
        "attribution. Across-delay presentations repeat base evidence and are "
        "not counted as independent episodes.",
        "",
        f"Production computation took {result['production_wall_seconds']:.1f} "
        f"seconds; total bounded wall time from the first smoke extraction "
        f"through finalization was {result['total_experiment_wall_seconds']:.1f} "
        "seconds. "
        "See `results.json`, `run_manifest.json`, feature receipts and selected "
        "probe checkpoints for reproducible details.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()

    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("One local CUDA worker is required")
    device = torch.device("cuda:0")
    started = time.time()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = SMOKE_PER_DELAY if args.mode == "smoke" else PRODUCTION_PER_DELAY
    hashes = source_hashes()
    checkpoint_path = ROOT / CHECKPOINT
    before_checkpoint_sha = sha(checkpoint_path)
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)

    model, payload = load_frozen(device)
    before_state = state_digest(model)
    extraction = {}
    for split in ("train", "val", "test"):
        extraction[split] = extract_split(
            model, split, counts[split], out, device, args.deadline_unix
        )
    after_state = state_digest(model)
    if after_state != before_state or sha(checkpoint_path) != before_checkpoint_sha:
        raise RuntimeError("Frozen model/checkpoint changed during extraction")
    del model, payload
    torch.cuda.empty_cache()

    if args.mode == "smoke":
        result = {
            "status": "passed",
            "mode": "smoke",
            "started_unix": started,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
            "extraction": extraction,
            "checkpoint_immutable": True,
            "frozen_model_state_immutable": True,
            "source_hashes": hashes,
        }
        json_write(out / "smoke_result.json", result)
        print(json.dumps(result, indent=2))
        return

    train = torch.load(out / "features_train.pt", map_location="cpu")
    val = torch.load(out / "features_val.pt", map_location="cpu")
    mean, std = train_scaling(train)
    scaling_path = out / "train_scaling.pt"
    torch.save({"mean": mean, "std": std}, scaling_path)

    probes = {}
    trained = {}
    for name, probe in (
        ("pooled", PooledProbe(seed=MODEL_SEED)),
        ("spatial", SpatialPriorityProbe(seed=MODEL_SEED)),
    ):
        trained[name], probes[name] = fit_probe(
            name, probe, train, val, mean.to(device), std.to(device), out,
            device, args.deadline_unix,
        )
    del train, val
    torch.cuda.empty_cache()

    # The test labels and fields are first loaded after both selections are fixed.
    test = torch.load(out / "features_test.pt", map_location="cpu")
    probabilities = {}
    priority_maps = None
    for name in ("pooled", "spatial"):
        probabilities[name], maps = evaluate_probe(
            trained[name], test, mean.to(device), std.to(device), device,
            name == "spatial",
        )
        if maps is not None:
            priority_maps = maps
    labels = test["labels"].numpy()
    delay_values = test["delays"].numpy()
    base_ids = test["base_ids"]

    summaries = {}
    for name in ("pooled", "spatial"):
        overall = classification(labels, probabilities[name])
        per_delay = {
            str(delay): classification(
                labels[delay_values == delay],
                probabilities[name][delay_values == delay],
            )
            for delay in DELAYS
        }
        summaries[name] = {"overall": overall, "per_delay": per_delay}

    paired = {"per_delay": {}}
    point_a = summaries["pooled"]["overall"]
    point_b = summaries["spatial"]["overall"]
    paired["overall"] = {
        "point": {
            key: point_b[key] - point_a[key]
            for key in ("accuracy", "balanced_accuracy", "macro_ovr_auc")
        },
        "bootstrap": stratified_base_bootstrap(
            labels, base_ids, probabilities["pooled"], probabilities["spatial"],
            BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
        ),
    }
    for delay in DELAYS:
        rows = delay_values == delay
        a = summaries["pooled"]["per_delay"][str(delay)]
        b = summaries["spatial"]["per_delay"][str(delay)]
        paired["per_delay"][str(delay)] = {
            "point": {
                key: b[key] - a[key]
                for key in ("accuracy", "balanced_accuracy", "macro_ovr_auc")
            },
            "bootstrap": stratified_base_bootstrap(
                labels[rows], np.asarray(base_ids)[rows],
                probabilities["pooled"][rows], probabilities["spatial"][rows],
                BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + delay + 1,
            ),
        }

    alignment_result = alignment(
        priority_maps, test["targets"].numpy(), base_ids, delay_values
    )
    np.savez_compressed(
        out / "priority_maps_test.npz",
        priority=priority_maps.astype(np.float32),
        labels=labels,
        delays=delay_values,
        targets=test["targets"].numpy(),
        base_ids=np.asarray(base_ids),
        trial_ids=np.asarray(test["trial_ids"]),
    )
    with (out / "predictions_test.jsonl").open("w") as handle:
        for index in range(len(labels)):
            handle.write(
                json.dumps(
                    {
                        "trial_id": test["trial_ids"][index],
                        "base_id": base_ids[index],
                        "delay": int(delay_values[index]),
                        "label": int(labels[index]),
                        "target_location": int(test["targets"][index]),
                        "pooled_probabilities": probabilities["pooled"][index].tolist(),
                        "spatial_probabilities": probabilities["spatial"][index].tolist(),
                        "priority_map": priority_maps[index].tolist(),
                    }
                )
                + "\n"
            )
    del test

    config = {
        "task": TASK,
        "delays": DELAYS,
        "split_seeds": SPLIT_SEEDS,
        "independent_base_episodes_per_delay": counts,
        "looks": LOOKS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip": CLIP,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cpu_threads": CPU_THREADS,
        "gpu_workers": 1,
    }
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "status": "completed",
        "analysis_only_post_hoc": True,
        "task": TASK,
        "chance": 0.25,
        "checkpoint": CHECKPOINT,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_sha256_before": before_checkpoint_sha,
        "checkpoint_sha256_after": sha(checkpoint_path),
        "checkpoint_immutable": sha(checkpoint_path) == before_checkpoint_sha,
        "frozen_model_state_sha256_before": before_state,
        "frozen_model_state_sha256_after": after_state,
        "frozen_model_state_immutable": before_state == after_state,
        "frozen_components": [
            "encoder", "opponent traces", "original pre-update attention",
            "spatial E/I memory", "comparator",
        ],
        "samples": counts,
        "split_seeds": SPLIT_SEEDS,
        "delay_presentations": {
            split: counts[split] * len(DELAYS) for split in counts
        },
        "extraction": extraction,
        "train_scaling_sha256": sha(scaling_path),
        "probes": probes,
        "test": {
            **summaries,
            "paired": paired,
            "test_read_once_after_both_probe_selections": True,
        },
        "priority_alignment": alignment_result,
        "source_hashes": hashes,
        "config": config,
        "config_sha256": config_sha,
        "started_unix": started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "absolute_deadline_unix": args.deadline_unix,
        "within_1800_second_shared_cap": time.time() <= args.deadline_unix,
        "cloud_calls": 0,
        "deployed_weight_updates": 0,
    }
    json_write(out / "results.json", result)
    (out / "report.md").write_text(make_report(result), encoding="utf-8")
    manifest = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            manifest[str(path.relative_to(out)).replace("\\", "/")] = sha(path)
    json_write(
        out / "run_manifest.json",
        {
            "status": "verified",
            "files": manifest,
            "source_hashes": hashes,
            "checkpoint_sha256": before_checkpoint_sha,
            "config_sha256": config_sha,
        },
    )
    print(json.dumps({
        "status": result["status"],
        "wall_seconds": result["wall_seconds"],
        "test": summaries,
        "paired": paired,
        "priority_alignment": alignment_result,
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
