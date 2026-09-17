"""Pre-flight 3: linear probes for motion direction on frozen H and R fields.

Forward-only; no model parameter is trained. Loads the hash-verified local
scratch motion-only step-3200 v1 checkpoint, generates cued motion-duration D0
trials, caches terminal H_T, R_T, C_T and the last-evidence-frame field H_9,
and fits regularized multinomial logistic probes with sklearn. If a probe
decodes the longest-duration direction of the cued patch above chance, the
information survives pooling and the fault lies in routing/readout.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import sha, write
from WorkingMemory.AttentionContextComparator.model import ARM as V1_ARM
from WorkingMemory.AttentionContextComparator.model import VERSION as V1_VERSION
from WorkingMemory.AttentionContextComparator.model import initialize_scratch as v1_initialize
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

CHECKPOINT = (
    ROOT
    / "WorkingMemory/AttentionContextComparator/SingleTaskMotion/runs/"
    "scratch_motion_20260915_194230/training/checkpoint_003200.pt"
)
TASK = "motion_duration_cued"
FIT_SEED = 71973001
TEST_SEED = 72973001
N_FIT = 768
N_TEST = 256
BATCH = 8
C_GRID = (1e-3, 1e-2, 1e-1)
CELL_PX = 100.0 / 13.0


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


def region_slice(center_xy):
    col = min(max(int(center_xy[0] / CELL_PX), 1), 11)
    row = min(max(int(center_xy[1] / CELL_PX), 1), 11)
    return slice(row - 1, row + 2), slice(col - 1, col + 2)


def extract(model, device, seed, split, count):
    stream = SpatialBatteryStream(seed, split)
    fields = []
    model.eval()
    captured = []
    handle = model.readout.fusion.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().cpu())
    )
    labels_all, targets, centers = [], [], []
    for offset in range(0, count, BATCH):
        images, labels, metadata = stream.batch(min(BATCH, count - offset), TASK, dict(delay=0))
        captured.clear()
        with torch.enable_grad():
            _, diagnostics = model(images.to(device), TASK, True)
        h_t, r_t, c_t = [x.detach().cpu() for x in diagnostics["terminal_fields"]]
        h_9 = captured[9]  # field on the last moving frame
        fields.append(dict(H_T=h_t, R_T=r_t, C_T=c_t, H_9=h_9))
        labels_all.extend(labels.tolist())
        targets.extend(m["target_location"] for m in metadata)
        centers.extend(m["positions_xy"][m["target_location"]] for m in metadata)
        del diagnostics
    handle.remove()
    stacked = {key: torch.cat([f[key] for f in fields]) for key in fields[0]}
    return stacked, np.array(labels_all), np.array(targets), centers


def feature_sets(stacked, centers):
    out = {}
    for key, tensor in stacked.items():
        out[key + "_full"] = tensor.flatten(1).numpy()
        out[key + "_pooled_mean_max"] = torch.cat((tensor.mean((2, 3)), tensor.amax((2, 3))), 1).numpy()
        local = []
        for index, center in enumerate(centers):
            rows, cols = region_slice(center)
            local.append(tensor[index, :, rows, cols].flatten().numpy())
        out[key + "_cued_3x3"] = np.stack(local)
    out["H_T+R_T_cued_3x3"] = np.concatenate((out["H_T_cued_3x3"], out["R_T_cued_3x3"]), 1)
    out["H_9+R_T_cued_3x3"] = np.concatenate((out["H_9_cued_3x3"], out["R_T_cued_3x3"]), 1)
    return out


def wilson(k, n, z=1.96):
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def fit_probe(x_fit, y_fit, x_test, y_test, seed=0):
    scaler = StandardScaler().fit(x_fit)
    x_fit_s, x_test_s = scaler.transform(x_fit), scaler.transform(x_test)
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    cv = {}
    for c in C_GRID:
        scores = []
        for train_index, valid_index in folds.split(x_fit_s, y_fit):
            probe = LogisticRegression(C=c, max_iter=3000, multi_class="multinomial")
            probe.fit(x_fit_s[train_index], y_fit[train_index])
            scores.append(float(probe.score(x_fit_s[valid_index], y_fit[valid_index])))
        cv[c] = float(np.mean(scores))
    best_c = max(cv, key=cv.get)
    probe = LogisticRegression(C=best_c, max_iter=3000, multi_class="multinomial")
    probe.fit(x_fit_s, y_fit)
    predictions = probe.predict(x_test_s)
    correct = int((predictions == y_test).sum())
    per_class = []
    for k in range(4):
        mask = y_test == k
        per_class.append(float((predictions[mask] == k).mean()) if mask.any() else None)
    return dict(
        dimensions=int(x_fit.shape[1]),
        cv_accuracy_by_C={f"{c:g}": v for c, v in cv.items()},
        chosen_C=best_c,
        fit_accuracy=float(probe.score(x_fit_s, y_fit)),
        test_accuracy=correct / len(y_test),
        test_accuracy_wilson95=wilson(correct, len(y_test)),
        test_balanced_accuracy=float(np.mean([v for v in per_class if v is not None])),
        per_class_recall=per_class,
        predicted_class_counts=np.bincount(predictions, minlength=4).tolist(),
    )


def main():
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved, digest = verified_checkpoint()
    model, _, _ = v1_initialize(V1_ARM, saved["config"])
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    fit_fields, y_fit, _, fit_centers = extract(model, device, FIT_SEED, "train", N_FIT)
    test_fields, y_test, _, test_centers = extract(model, device, TEST_SEED, "val", N_TEST)
    extraction_seconds = time.time() - started
    fit_features = feature_sets(fit_fields, fit_centers)
    test_features = feature_sets(test_fields, test_centers)
    probes = {}
    for name in fit_features:
        tick = time.time()
        probes[name] = fit_probe(fit_features[name], y_fit, test_features[name], y_test)
        probes[name]["seconds"] = time.time() - tick
        print(name, json.dumps({k: probes[name][k] for k in ("dimensions", "chosen_C", "fit_accuracy", "test_accuracy", "test_accuracy_wilson95")}), flush=True)
    best = max(probes, key=lambda k: probes[k]["test_accuracy"])
    rng = np.random.default_rng(0)
    control = fit_probe(fit_features[best], rng.permutation(y_fit), test_features[best], y_test)
    result = dict(
        status="completed",
        checkpoint=str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        checkpoint_sha256=digest,
        checkpoint_step=3200,
        task=TASK,
        condition="D0",
        fit=dict(seed=FIT_SEED, split="train", n=int(len(y_fit)), label_counts=np.bincount(y_fit, minlength=4).tolist()),
        test=dict(seed=TEST_SEED, split="val", n=int(len(y_test)), label_counts=np.bincount(y_test, minlength=4).tolist()),
        chance=0.25,
        probe="StandardScaler + multinomial LogisticRegression (lbfgs), C by 3-fold CV over " + str(C_GRID),
        fields=dict(
            H_T="terminal sensory field (report frame 10, blank)",
            R_T="terminal E/I firing rates",
            C_T="terminal attention context (pre-W_O)",
            H_9="sensory field on the last moving frame (frame 9)",
            cued_3x3="the 3x3 cells around the cued patch centre (uses the cue location as a probe-side oracle)",
        ),
        probes=probes,
        best_probe=best,
        permuted_label_control_for_best=control,
        extraction_seconds=extraction_seconds,
        wall_seconds=time.time() - started,
        device=device,
        interpretation=(
            "Forward-only; no model training. Above-chance decoding shows the "
            "direction information is present in that field at that time; chance "
            "decoding under these probes does not prove absence."
        ),
    )
    out = Path(__file__).with_name("preflight") / "preflight_probe.json"
    write(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "probes"}, indent=2))


if __name__ == "__main__":
    main()
