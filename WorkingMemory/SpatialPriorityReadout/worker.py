"""Bounded GPU worker for the spatial-priority-readout arm."""
import copy
import csv
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write


def run(job):
    import numpy as np
    import torch

    from WorkingMemory.SpatialPriorityReadout.model import (
        ARM,
        VERSION,
        groups,
        initialize_scratch,
    )
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    from WorkingMemory.UnbiasedAttention.protocol import summarize_spatial

    deadline = float(job["deadline"])
    timer = threading.Timer(
        max(0.01, deadline - time.time()), lambda: os._exit(124)
    )
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
    model.cuda()
    if optimizer.state:
        raise RuntimeError("Scratch optimizer unexpectedly has loaded state")

    checkpoint_path = job.get("checkpoint")
    if checkpoint_path:
        path = Path(checkpoint_path)
        rows = [
            json.loads(line)
            for line in (path.parent / "checkpoint_index.jsonl").read_text().splitlines()
        ]
        if not any(
            row["file"] == path.name and row["sha256"] == sha(path) for row in rows
        ):
            raise RuntimeError("Resume checkpoint missing valid hash index")
        saved = torch.load(path, map_location="cpu")
        if (
            saved["version"] != VERSION
            or saved["arm"] != ARM
            or saved["config"] != cfg
            or saved["source_hashes"] != job["source_hashes"]
            or saved["lineage"] != lineage
        ):
            raise RuntimeError("Resume checkpoint scratch-lineage/config mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        stream.load_state_dict(saved["stream"])
        step = saved["step"]
        counts = Counter(saved["counts"])
        cell_counts = Counter(saved["cell_counts"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state(saved["rng"]["cuda"])
        del saved

    write(
        out / "initialization.json",
        dict(
            lineage=lineage,
            total_parameters=sum(parameter.numel() for parameter in model.parameters()),
            trainable_parameters=sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            loaded_parent=False,
            loaded_model_tensors=0,
            inherited_adam_states=0,
            fresh_optimizer_states=len(list(model.parameters())),
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
                rng=dict(
                    python=random.getstate(),
                    numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(),
                ),
            )
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
            with (out / "checkpoint_index.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(file=path.name, step=step, sha256=sha(path))
                    )
                    + "\n"
                )
        return str(path)

    def task_microbatch(name, diagnostic=False):
        cell = cfg["cells"][name]
        images, labels, metadata = stream.batch(
            cfg["batch_size"], cell["task"], cell["condition"]
        )
        started = time.time()
        if diagnostic:
            logits, diagnostics = model(images.cuda(), cell["task"], True)
        else:
            logits = model(images.cuda(), cell["task"])
            diagnostics = None
        loss = torch.nn.functional.cross_entropy(logits, labels.cuda().long())
        (loss / 5).backward()
        detail = {}
        if diagnostic:
            detail = dict(
                priority_entropy=float(
                    -(
                        diagnostics["priority"].clamp_min(1e-30)
                        * diagnostics["priority"].clamp_min(1e-30).log()
                    )
                    .sum((2, 3))
                    .mean()
                    .detach()
                ),
                priority_peak=float(
                    diagnostics["priority"].amax((2, 3)).mean().detach()
                ),
                priority_gradient_norm=float(
                    sum(
                        parameter.grad.square().sum()
                        for parameter in model.priority_readout.parameters()
                        if parameter.grad is not None
                    )
                    .sqrt()
                    .detach()
                ),
                attention_gradient_norm=float(
                    sum(
                        parameter.grad.square().sum()
                        for parameter in model.attention.parameters()
                        if parameter.grad is not None
                    )
                    .sqrt()
                    .detach()
                ),
            )
        torch.cuda.synchronize()
        return dict(
            name=name,
            task=cell["task"],
            loss=float(loss.detach()),
            accuracy=float(
                (logits.argmax(1).cpu() == labels).float().mean()
            ),
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
                cfg["scheduler_seed"]
                + 100003 * task_index
                + step // len(names)
            )
            chosen = names[
                int(
                    generator.permutation(len(names))[
                        step % len(names)
                    ]
                )
            ]
            part = task_microbatch(chosen, record)
            parts.append(part)
            cell_counts[chosen] += cfg["batch_size"]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["clip"]
        )
        if not torch.isfinite(gradient_norm) or not all(
            np.isfinite(part["loss"]) for part in parts
        ):
            raise FloatingPointError("Nonfinite loss/gradient")
        optimizer.step()
        step += 1
        counts["episodes"] += 40
        counts["frames"] += sum(
            cfg["batch_size"] * part["frames"] for part in parts
        )
        return dict(
            step=step,
            episodes=counts["episodes"],
            frames_total=counts["frames"],
            loss=float(np.mean([part["loss"] for part in parts])),
            accuracy=float(np.mean([part["accuracy"] for part in parts])),
            gradient_norm=float(gradient_norm),
            clipped=int(float(gradient_norm) > cfg["clip"]),
            seconds=float(sum(part["seconds"] for part in parts)),
            task_losses=json.dumps(
                {part["task"]: part["loss"] for part in parts}
            ),
            task_accuracies=json.dumps(
                {part["task"]: part["accuracy"] for part in parts}
            ),
            cells=json.dumps({part["task"]: part["name"] for part in parts}),
            diagnostics=json.dumps(
                {part["task"]: part["diagnostics"] for part in parts}
            ),
        )

    started = time.time()
    kind = job["kind"]
    if kind == "profile":
        rows = []
        torch.cuda.reset_peak_memory_stats()
        model.train()
        for index in range(job.get("profile_updates", 3)):
            rows.append(complete_update(record=index == 0))
        result = dict(
            status="completed",
            rows=rows,
            mean_update_seconds=float(np.mean([row["seconds"] for row in rows])),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            checkpoint=save_checkpoint(),
        )
    elif kind == "train":
        if not checkpoint_path:
            save_checkpoint()
        metrics_path = out / "metrics.csv"
        columns = (
            "step",
            "episodes",
            "frames_total",
            "loss",
            "accuracy",
            "gradient_norm",
            "clipped",
            "seconds",
            "task_losses",
            "task_accuracies",
            "cells",
            "diagnostics",
        )
        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            if handle.tell() == 0:
                writer.writeheader()
            model.train()
            while step < job["target"]:
                if time.time() > deadline - 15:
                    break
                value = complete_update(record=step % 128 == 0)
                writer.writerow(value)
                handle.flush()
                if step % 32 == 0:
                    print(
                        json.dumps(
                            {
                                key: value[key]
                                for key in (
                                    "step",
                                    "episodes",
                                    "loss",
                                    "accuracy",
                                    "task_losses",
                                )
                            }
                        ),
                        flush=True,
                    )
                if step % 256 == 0:
                    save_checkpoint()
        result = dict(
            status="completed" if step == job["target"] else "budget_stopped",
            step=step,
            checkpoint=save_checkpoint(),
            counts=dict(counts),
            cell_counts=dict(cell_counts),
        )
    elif kind == "eval":
        model.eval()
        rows = []
        priority_path = out / "priority_maps.jsonl"
        with (out / "predictions.jsonl").open("w") as prediction_handle, priority_path.open(
            "w"
        ) as priority_handle:
            for name, cell in cfg["cells"].items():
                number = (
                    100
                    if job["split"] == "val"
                    and cell["task"] == "krauzlis_cued_motion"
                    else (
                        400
                        if job["split"] != "val"
                        and cell["task"] == "krauzlis_cued_motion"
                        else job["n"]
                    )
                )
                evaluation = SpatialBatteryStream(job["eval_seed"], job["split"])
                for offset in range(0, number, cfg["batch_size"]):
                    images, labels, metadata = evaluation.batch(
                        min(cfg["batch_size"], number - offset),
                        cell["task"],
                        cell["condition"],
                    )
                    model.priority_readout.capture = True
                    with torch.no_grad():
                        probabilities = (
                            model(images.cuda(), cell["task"])
                            .softmax(1)
                            .cpu()
                            .tolist()
                        )
                    maps = model.priority_readout.last_priority.cpu().tolist()
                    model.priority_readout.capture = False
                    for index, (probability, label, meta, priority) in enumerate(
                        zip(probabilities, labels.tolist(), metadata, maps)
                    ):
                        trial = f"{name}/{offset + index}"
                        row = dict(
                            task=cell["task"],
                            condition=name,
                            protocol=cell["task"],
                            label=label,
                            probabilities=probability,
                            base_id=None,
                            paired_base_id=trial,
                            trial_id=trial,
                            metadata=meta,
                        )
                        rows.append(row)
                        prediction_handle.write(json.dumps(row) + "\n")
                        priority_handle.write(
                            json.dumps(
                                dict(
                                    task=cell["task"],
                                    condition=name,
                                    trial_id=trial,
                                    priority=priority,
                                )
                            )
                            + "\n"
                        )
                    prediction_handle.flush()
                    priority_handle.flush()
                    if time.time() > deadline - 15:
                        raise TimeoutError("Evaluation deadline")
                print(json.dumps(dict(evaluated=name, n=number)), flush=True)
        result = summarize_spatial(rows)
        result.update(
            status="completed",
            step=step,
            predictions=str(out / "predictions.jsonl"),
            priority_maps=str(priority_path),
            checkpoint=job["checkpoint"],
            priority_map_sha256=sha(priority_path),
        )
        write(out / "summary.json", result)
    else:
        raise ValueError(kind)
    result["worker_seconds"] = time.time() - started
    write(job["result"], result)
    timer.cancel()


if __name__ == "__main__":
    run(read(sys.argv[1]))
