"""Bounded local GPU worker for AV-context v2 with live instrumentation.

Kinds: profile (3 updates), train (to a target step), eval (val/test split).
Every DIAGNOSTIC_EVERY updates a diagnostic forward records, per task:
priority entropy as a fraction of ln(169), attention mass beyond 3 cells per
head, spatial variance of evidence and of H_T, softplus(raw_locality) per head
and source_bias. Evaluation also records the same quantities on the validation
trials so that the pre-registered gate reads measured validation values.
"""
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.V2.protocol import (
    CHECKPOINT_EVERY,
    DIAGNOSTIC_EVERY,
    UNIFORM_PRIORITY_ENTROPY,
)

LN169 = UNIFORM_PRIORITY_ENTROPY


def entropy_fraction(priority):
    p = priority.clamp_min(1e-30)
    return (-(p * p.log()).sum((1, 2, 3)) / LN169)


def run(job):
    import numpy as np
    import torch

    from WorkingMemory.AttentionContextComparator.V2.model import (
        ARM,
        VERSION,
        initialize_scratch,
    )
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    from WorkingMemory.UnbiasedAttention.protocol import summarize_spatial

    deadline = float(job["deadline"])
    timer = threading.Timer(max(0.01, deadline - time.time()), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name, digest in job["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned source changed " + name)
    if job["arm"] != ARM:
        raise ValueError("Unauthorized arm")

    cfg = job["config"]
    device = torch.device(job["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["model_seed"])
    np.random.seed(cfg["model_seed"])
    torch.manual_seed(cfg["model_seed"])
    torch.cuda.manual_seed_all(cfg["model_seed"])
    model, optimizer, lineage = initialize_scratch(ARM, cfg)
    stream = SpatialBatteryStream(cfg["train_seed"], "train")
    step = 0
    counts = Counter(episodes=0, frames=0)
    cell_counts = Counter()

    def rng_state():
        return dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        )

    checkpoint_path = job.get("checkpoint")
    if checkpoint_path:
        path = Path(checkpoint_path)
        rows = [
            json.loads(line)
            for line in (path.parent / "checkpoint_index.jsonl").read_text().splitlines()
        ]
        if not any(row["file"] == path.name and row["sha256"] == sha(path) for row in rows):
            raise RuntimeError("Resume checkpoint missing valid hash index")
        saved = torch.load(path, map_location="cpu")
        if (
            saved["version"] != VERSION
            or saved["arm"] != ARM
            or saved["config"] != cfg
            or saved["source_hashes"] != job["source_hashes"]
            or saved["lineage"] != lineage
        ):
            raise RuntimeError("Resume checkpoint lineage/config mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        stream.load_state_dict(saved["stream"])
        step = int(saved["step"])
        counts = Counter(saved["counts"])
        cell_counts = Counter(saved["cell_counts"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and saved["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        del saved

    model.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)

    initialization_path = out / "initialization.json"
    if not initialization_path.exists():
        write(
            initialization_path,
            dict(
                lineage=lineage,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                loaded_parent=False,
                loaded_model_tensors=0,
                inherited_adam_states=0,
                v2_changes=cfg["v2_changes"],
                learning_rates=dict(new_lr=cfg["new_lr"], parent_lr=cfg["parent_lr"]),
            ),
        )

    def save_checkpoint():
        path = out / f"checkpoint_{step:06d}.pt"
        if not path.exists():
            payload = dict(
                version=VERSION,
                arm=ARM,
                config=cfg,
                source_hashes=job["source_hashes"],
                lineage=lineage,
                step=step,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                stream=stream.state_dict(),
                counts=dict(counts),
                cell_counts=dict(cell_counts),
                rng=rng_state(),
            )
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
            with (out / "checkpoint_index.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(file=path.name, step=step, sha256=sha(path))) + "\n")
        return str(path)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def diagnostic_detail(diagnostics, task, field):
        records = diagnostics["records"]
        far = np.mean([r["attention_mass_beyond_3_by_head"] for r in records], 0)
        visual = np.mean([r["visual_attention_mass_by_head"] for r in records], 0)
        evidence = diagnostics["evidence"]
        priority = diagnostics["priority"]
        return dict(
            priority_entropy_fraction=float(entropy_fraction(priority).mean()),
            priority_peak=float(priority.amax((2, 3)).mean()),
            attention_mass_beyond_3_by_head=far.tolist(),
            visual_attention_mass_by_head=visual.tolist(),
            evidence_spatial_variance=float(evidence.flatten(2).var(-1).mean()),
            evidence_spatial_std_over_mean_abs=float(
                (evidence.flatten(2).std(-1) / (evidence.flatten(2).abs().mean(-1) + 1e-8)).mean()
            ),
            field_spatial_variance=float(field.detach().flatten(2).var(-1).mean()),
            locality_by_head=records[-1]["locality"],
            source_bias=records[-1]["source_bias"],
            selection_weight_norm=float(model.priority_readout.selection[task].weight.norm()),
            priority_gradient_norm=float(
                sum(p.grad.square().sum() for p in model.priority_readout.parameters() if p.grad is not None).sqrt()
            ),
            attention_gradient_norm=float(
                sum(p.grad.square().sum() for p in model.attention.parameters() if p.grad is not None).sqrt()
            ),
        )

    def task_microbatch(name, diagnostic=False):
        cell = cfg["cells"][name]
        images, labels, _ = stream.batch(cfg["batch_size"], cell["task"], cell["condition"])
        images = images.to(device)
        started = time.time()
        if diagnostic:
            logits, diagnostics = model(images, cell["task"], True)
        else:
            logits = model(images, cell["task"])
            diagnostics = None
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        (loss / 5).backward()
        detail = {}
        if diagnostic:
            detail = diagnostic_detail(diagnostics, cell["task"], diagnostics["terminal_fields"][0])
            del diagnostics
        synchronize()
        return dict(
            name=name,
            task=cell["task"],
            loss=float(loss.detach()),
            accuracy=float((logits.argmax(1).cpu() == labels).float().mean()),
            frames=int(images.shape[1]),
            seconds=time.time() - started,
            diagnostics=detail,
        )

    def complete_update(record=False):
        nonlocal step
        optimizer.zero_grad(set_to_none=True)
        parts = []
        for task_index, task in enumerate(cfg["task_classes"]):
            names = cfg["train_names"][task]
            generator = np.random.default_rng(
                cfg["scheduler_seed"] + 100003 * task_index + step // len(names)
            )
            chosen = names[int(generator.permutation(len(names))[step % len(names)])]
            part = task_microbatch(chosen, record)
            parts.append(part)
            cell_counts[chosen] += cfg["batch_size"]
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        if not torch.isfinite(gradient_norm) or not all(np.isfinite(p["loss"]) for p in parts):
            raise FloatingPointError("Nonfinite loss/gradient")
        optimizer.step()
        step += 1
        counts["episodes"] += 40
        counts["frames"] += sum(cfg["batch_size"] * p["frames"] for p in parts)
        return dict(
            step=step,
            episodes=counts["episodes"],
            frames_total=counts["frames"],
            loss=float(np.mean([p["loss"] for p in parts])),
            accuracy=float(np.mean([p["accuracy"] for p in parts])),
            gradient_norm=float(gradient_norm),
            clipped=int(float(gradient_norm) > cfg["clip"]),
            seconds=float(sum(p["seconds"] for p in parts)),
            task_losses=json.dumps({p["task"]: p["loss"] for p in parts}),
            task_accuracies=json.dumps({p["task"]: p["accuracy"] for p in parts}),
            cells=json.dumps({p["task"]: p["name"] for p in parts}),
            diagnostics=json.dumps({p["task"]: p["diagnostics"] for p in parts}),
        )

    started = time.time()
    kind = job["kind"]
    if kind == "profile":
        rows = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        for index in range(job.get("profile_updates", 3)):
            rows.append(complete_update(record=index == 0))
        result = dict(
            status="completed",
            rows=rows,
            mean_update_seconds=float(np.mean([r["seconds"] for r in rows])),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            parameters=sum(p.numel() for p in model.parameters()),
            profile_exposure_discarded=True,
        )
    elif kind == "train":
        if not checkpoint_path:
            save_checkpoint()
        metrics_path = out / "metrics.csv"
        live_path = out / "live_diagnostics.jsonl"
        columns = (
            "step", "episodes", "frames_total", "loss", "accuracy", "gradient_norm",
            "clipped", "seconds", "task_losses", "task_accuracies", "cells", "diagnostics",
        )
        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            if handle.tell() == 0:
                writer.writeheader()
            model.train()
            while step < int(job["target"]):
                if time.time() > deadline - 20:
                    break
                record = step % DIAGNOSTIC_EVERY == 0
                value = complete_update(record=record)
                writer.writerow(value)
                handle.flush()
                if record:
                    with live_path.open("a") as live:
                        live.write(
                            json.dumps(
                                dict(
                                    step=value["step"],
                                    episodes=value["episodes"],
                                    unix=time.time(),
                                    loss=value["loss"],
                                    accuracy=value["accuracy"],
                                    task_losses=json.loads(value["task_losses"]),
                                    task_accuracies=json.loads(value["task_accuracies"]),
                                    diagnostics=json.loads(value["diagnostics"]),
                                )
                            )
                            + "\n"
                        )
                if step % 32 == 0:
                    print(
                        json.dumps({k: value[k] for k in ("step", "episodes", "loss", "accuracy", "task_losses", "seconds")}),
                        flush=True,
                    )
                if step % CHECKPOINT_EVERY == 0:
                    save_checkpoint()
        result = dict(
            status="completed" if step == int(job["target"]) else "budget_stopped",
            step=step,
            checkpoint=save_checkpoint(),
            counts=dict(counts),
            cell_counts=dict(cell_counts),
        )
    elif kind == "eval":
        if not checkpoint_path:
            raise ValueError("Evaluation requires a checkpoint")
        model.eval()
        rows = []
        entropy_by_cell = defaultdict(list)
        far_by_task = defaultdict(list)
        visual_by_task = defaultdict(list)
        priority_path = out / "priority_maps.jsonl"
        with (out / "predictions.jsonl").open("w") as prediction_handle, priority_path.open("w") as priority_handle:
            for name, cell in cfg["cells"].items():
                number = (
                    100
                    if job["split"] == "val" and cell["task"] == "krauzlis_cued_motion"
                    else (400 if job["split"] != "val" and cell["task"] == "krauzlis_cued_motion" else job["n"])
                )
                if job.get("smoke_n"):
                    number = int(job["smoke_n"])  # smoke tests only; never used by run.py
                evaluation = SpatialBatteryStream(job["eval_seed"], job["split"])
                for offset in range(0, number, cfg["batch_size"]):
                    images, labels, metadata = evaluation.batch(
                        min(cfg["batch_size"], number - offset), cell["task"], cell["condition"]
                    )
                    with torch.no_grad():
                        logits, diagnostics = model(images.to(device), cell["task"], True)
                        probabilities = logits.softmax(1).cpu().tolist()
                        priority = diagnostics["priority"]
                        entropy_by_cell[name].extend(entropy_fraction(priority).cpu().tolist())
                        far_by_task[cell["task"]].append(
                            np.mean([r["attention_mass_beyond_3_by_head"] for r in diagnostics["records"]], 0)
                        )
                        visual_by_task[cell["task"]].append(
                            np.mean([r["visual_attention_mass_by_head"] for r in diagnostics["records"]], 0)
                        )
                        maps = priority[:, 0].cpu().tolist()
                        del diagnostics
                    for index, (probability, label, meta, priority_map) in enumerate(
                        zip(probabilities, labels.tolist(), metadata, maps)
                    ):
                        trial = f"{name}/{offset + index}"
                        row = dict(
                            task=cell["task"], condition=name, protocol=cell["task"], label=label,
                            probabilities=probability, base_id=None, paired_base_id=trial,
                            trial_id=trial, metadata=meta,
                        )
                        rows.append(row)
                        prediction_handle.write(json.dumps(row) + "\n")
                        priority_handle.write(
                            json.dumps(dict(task=cell["task"], condition=name, trial_id=trial, priority=priority_map)) + "\n"
                        )
                    prediction_handle.flush()
                    priority_handle.flush()
                    if time.time() > deadline - 20:
                        raise TimeoutError("Evaluation deadline")
                print(json.dumps(dict(evaluated=name, n=number)), flush=True)
        result = summarize_spatial(rows)
        by_task = defaultdict(list)
        for name, values in entropy_by_cell.items():
            by_task[cfg["cells"][name]["task"]].extend(values)
        far_all = np.mean(np.concatenate([np.stack(v) for v in far_by_task.values()]), 0)
        result.update(
            status="completed",
            step=step,
            predictions=str(out / "predictions.jsonl"),
            priority_maps=str(priority_path),
            checkpoint=job["checkpoint"],
            priority_map_sha256=sha(priority_path),
            priority_entropy_fraction_by_cell={k: float(np.mean(v)) for k, v in entropy_by_cell.items()},
            priority_entropy_fraction_by_task={k: float(np.mean(v)) for k, v in by_task.items()},
            priority_entropy_fraction_min_by_task={k: float(np.min(v)) for k, v in by_task.items()},
            attention_mass_beyond_3_by_head=far_all.tolist(),
            attention_mass_beyond_3_by_task_by_head={k: np.mean(np.stack(v), 0).tolist() for k, v in far_by_task.items()},
            visual_attention_mass_by_task_by_head={k: np.mean(np.stack(v), 0).tolist() for k, v in visual_by_task.items()},
            locality_by_head=torch.nn.functional.softplus(model.attention.raw_locality).detach().cpu().tolist(),
            source_bias=model.attention.source_bias.detach().cpu().tolist(),
        )
        write(out / "summary.json", result)
    else:
        raise ValueError(kind)
    result["worker_seconds"] = time.time() - started
    write(job["result"], result)
    timer.cancel()


if __name__ == "__main__":
    run(read(sys.argv[1]))
