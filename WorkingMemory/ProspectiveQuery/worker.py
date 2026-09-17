"""Bounded worker for the pinned five-task prospective-query experiment."""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import copy
import csv
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write


def worker(job):
    import numpy as np
    import torch

    from WorkingMemory.ProspectiveQuery.model import VERSION, migrate
    from WorkingMemory.ProspectiveQuery.protocol import assert_same_as_biased_training
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    from WorkingMemory.UnbiasedAttention.protocol import summarize_spatial

    deadline = job["deadline"]
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
    if sha(job["parent"]) != job["parent_sha256"]:
        raise RuntimeError("Parent hash mismatch")
    cfg = job["config"]
    assert_same_as_biased_training(cfg)
    arm = job["arm"]
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)

    def indexed_checkpoints(directory):
        directory = Path(directory)
        index_path = directory / "checkpoint_index.jsonl"
        files = set(directory.glob("checkpoint_*.pt"))
        if not index_path.exists():
            if files:
                raise RuntimeError("Checkpoint files exist without a hash index")
            return []
        rows = [json.loads(line) for line in index_path.read_text().splitlines()]
        seen_files = set()
        seen_steps = set()
        for row in rows:
            if (
                not re.fullmatch(r"checkpoint_[0-9]{6}\.pt", row.get("file", ""))
                or row.get("step") in seen_steps
                or row["file"] in seen_files
            ):
                raise RuntimeError("Malformed or duplicate checkpoint index entry")
            path = directory / row["file"]
            if not path.is_file() or sha(path) != row["sha256"]:
                raise RuntimeError("Missing or hash-mismatched indexed checkpoint")
            if int(path.stem.rsplit("_", 1)[1]) != int(row["step"]):
                raise RuntimeError("Checkpoint filename/step mismatch")
            seen_files.add(row["file"])
            seen_steps.add(row["step"])
        unexpected = {path.name for path in files} - seen_files
        if unexpected:
            raise RuntimeError(
                "Unexpected unindexed checkpoint(s): " + ", ".join(sorted(unexpected))
            )
        return rows

    random.seed(cfg["model_seed"])
    np.random.seed(cfg["model_seed"])
    torch.manual_seed(cfg["model_seed"])
    torch.cuda.manual_seed_all(cfg["model_seed"])
    parent = torch.load(job["parent"], map_location="cpu")
    model, opt, lineage = migrate(parent, arm, cfg, job["parent_sha256"])
    stream = SpatialBatteryStream(cfg["train_seed"], "train")
    step = 8400
    counts = Counter(episodes=0, frames=0)
    cell_counts = Counter()
    model.cuda()

    random.setstate(parent["rng"]["python"])
    np.random.set_state(parent["rng"]["numpy"])
    torch.set_rng_state(parent["rng"]["torch"])
    torch.cuda.set_rng_state(parent["rng"]["cuda"])
    del parent
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.cuda()

    if job.get("checkpoint"):
        checkpoint_path = Path(job["checkpoint"])
        index = indexed_checkpoints(checkpoint_path.parent)
        if not any(
            row["file"] == checkpoint_path.name
            and row["sha256"] == sha(checkpoint_path)
            for row in index
        ):
            raise RuntimeError("Resume checkpoint is absent from its hash index")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if (
            checkpoint["version"] != VERSION
            or checkpoint["arm"] != arm
            or checkpoint["config"] != cfg
            or checkpoint["source_hashes"] != job["source_hashes"]
        ):
            raise RuntimeError("Resume checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(checkpoint["optimizer"])
        stream.load_state_dict(checkpoint["stream"])
        step = checkpoint["step"]
        counts = Counter(checkpoint["counts"])
        cell_counts = Counter(checkpoint["cell_counts"])
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        torch.cuda.set_rng_state(checkpoint["rng"]["cuda"])
        del checkpoint
    if job["kind"] == "train":
        indexed = indexed_checkpoints(out)
        eligible = [row for row in indexed if row["step"] <= job["target"]]
        newest = (
            str(out / max(eligible, key=lambda row: row["step"])["file"])
            if eligible
            else None
        )
        supplied = str(Path(job["checkpoint"])) if job.get("checkpoint") else None
        if supplied != newest:
            raise RuntimeError(
                "Train job did not select newest indexed checkpoint <= target"
            )

    write(
        out / "initialization.json",
        dict(
            lineage=lineage,
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            gamma_initial=0.0,
        ),
    )

    def save():
        path = out / f"checkpoint_{step:06d}.pt"
        indexed = indexed_checkpoints(out)
        existing = [row for row in indexed if row["file"] == path.name]
        if path.exists():
            if len(existing) != 1 or existing[0]["step"] != step:
                raise RuntimeError("Unexpected existing checkpoint; refusing to skip")
            saved = torch.load(path, map_location="cpu")
            if (
                saved.get("version") != VERSION
                or saved.get("arm") != arm
                or saved.get("step") != step
                or saved.get("config") != cfg
                or saved.get("source_hashes") != job["source_hashes"]
            ):
                raise RuntimeError("Existing checkpoint identity mismatch")
        else:
            if existing:
                raise RuntimeError("Checkpoint index points to a missing file")
            data = dict(
                version=VERSION,
                arm=arm,
                config=cfg,
                source_hashes=job["source_hashes"],
                lineage=lineage,
                step=step,
                model=model.state_dict(),
                optimizer=opt.state_dict(),
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
            torch.save(data, temporary)
            os.replace(temporary, path)
            with (out / "checkpoint_index.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(dict(file=path.name, step=step, sha256=sha(path))) + "\n"
                )
        return str(path)

    def microbatch(name, record=False):
        tick = time.time()
        cell = cfg["cells"][name]
        images, labels, _ = stream.batch(
            cfg["batch_size"], cell["task"], cell["condition"]
        )
        model.train()
        if record:
            logits, diagnostic = model(images.cuda(), cell["task"], True)
        else:
            logits = model(images.cuda(), cell["task"])
            diagnostic = None
        loss = torch.nn.functional.cross_entropy(logits, labels.cuda().long())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite microbatch loss")
        (loss / 5).backward()
        detail = {}
        if record:
            detail["gamma"] = float(model.attention.gamma.detach())
            detail["gradient_semantics"] = (
                "Pre-clip cumulative gradients after this task microbatch; "
                "includes all preceding task microbatches in the current update"
            )
            detail["gamma_gradient_cumulative_preclip"] = (
                None
                if model.attention.gamma.grad is None
                else float(model.attention.gamma.grad)
            )
            detail["attention_gradient_norm_cumulative_preclip"] = float(
                sum(
                    p.grad.square().sum()
                    for p in model.attention.parameters()
                    if p.grad is not None
                ).sqrt()
            )
            detail["encoder_gradient_norm_cumulative_preclip"] = float(
                sum(
                    p.grad.square().sum()
                    for p in model.encoder.parameters()
                    if p.grad is not None
                ).sqrt()
            )
            detail["current_task_first_sensory_field_gradient_norm_preclip"] = float(
                diagnostic["first_field"].grad.norm()
            )
            detail["per_frame_attention"] = [
                {
                    key: value
                    for key, value in row.items()
                    if key.startswith("memory_attention")
                    or key.startswith("attention_entropy")
                    or key == "gamma"
                }
                for row in diagnostic["records"]
            ]
        torch.cuda.synchronize()
        return dict(
            loss=float(loss),
            accuracy=float((logits.argmax(1).cpu() == labels).float().mean()),
            frames=int(images.shape[1]),
            seconds=time.time() - tick,
            diagnostics=detail,
        )

    started = time.time()
    kind = job["kind"]
    if kind == "profile":
        rows = []
        torch.cuda.reset_peak_memory_stats()
        for name in job["profile_names"]:
            opt.zero_grad(set_to_none=True)
            value = microbatch(name, True)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            cell = cfg["cells"][name]
            images, _, _ = stream.batch(
                cfg["batch_size"], cell["task"], cell["condition"]
            )
            tick = time.time()
            model.eval()
            with torch.no_grad():
                model(images.cuda(), cell["task"])
            torch.cuda.synchronize()
            value.update(
                name=name,
                gradient_norm=float(norm),
                eval_seconds=time.time() - tick,
            )
            rows.append(value)
            step += 1
            counts["episodes"] += cfg["batch_size"]
        result = dict(
            status="completed",
            rows=rows,
            checkpoint=save(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            parameters=sum(p.numel() for p in model.parameters()),
        )
    elif kind == "train":
        if not job.get("checkpoint"):
            save()
        fields = [
            "step",
            "episodes",
            "frames_total",
            "cell",
            "loss",
            "accuracy",
            "gradient_norm",
            "clipped",
            "frames",
            "seconds",
            "gamma",
            "gamma_gradient_preclip",
            "diagnostics",
        ]
        with (out / "metrics.csv").open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            if handle.tell() == 0:
                writer.writeheader()
            while step < job["target"]:
                if time.time() > deadline - 15:
                    break
                opt.zero_grad(set_to_none=True)
                parts = []
                chosen_names = []
                for task_index, task in enumerate(cfg["task_classes"]):
                    names = cfg["train_names"][task]
                    scheduler = np.random.default_rng(
                        cfg["scheduler_seed"]
                        + 100003 * task_index
                        + (step - 8400) // len(names)
                    )
                    chosen = names[
                        int(scheduler.permutation(len(names))[(step - 8400) % len(names)])
                    ]
                    chosen_names.append(chosen)
                    parts.append(microbatch(chosen, step % 128 == 0))
                    cell_counts[chosen] += cfg["batch_size"]
                gamma_gradient_preclip = (
                    None
                    if model.attention.gamma.grad is None
                    else float(model.attention.gamma.grad)
                )
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite accumulated gradient")
                opt.step()
                step += 1
                counts["episodes"] += 5 * cfg["batch_size"]
                frames = sum(part["frames"] for part in parts)
                counts["frames"] += cfg["batch_size"] * frames
                value = dict(
                    step=step,
                    episodes=counts["episodes"],
                    frames_total=counts["frames"],
                    cell="five_tasks",
                    loss=float(np.mean([part["loss"] for part in parts])),
                    accuracy=float(np.mean([part["accuracy"] for part in parts])),
                    gradient_norm=float(norm),
                    clipped=int(float(norm) > cfg["clip"]),
                    frames=frames,
                    seconds=sum(part["seconds"] for part in parts),
                    gamma=float(model.attention.gamma.detach()),
                    gamma_gradient_preclip=gamma_gradient_preclip,
                    diagnostics=json.dumps(
                        {
                            task: part["diagnostics"]
                            for task, part in zip(cfg["task_classes"], parts)
                        }
                    ),
                )
                writer.writerow(value)
                handle.flush()
                if step % 64 == 0:
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                step=step,
                                episodes=counts["episodes"],
                                loss=value["loss"],
                                gamma=value["gamma"],
                            )
                        ),
                        flush=True,
                    )
                if step % 256 == 0:
                    save()
        result = dict(
            status="completed" if step == job["target"] else "budget_stopped",
            step=step,
            checkpoint=save(),
            counts=dict(counts),
            cell_counts=dict(cell_counts),
            gamma=float(model.attention.gamma.detach()),
        )
    elif kind == "eval":
        model.eval()
        rows = []
        attention_cells = {}
        with (out / "predictions.jsonl").open("w") as handle:
            for name, cell in cfg["cells"].items():
                n = (
                    100 if job["split"] == "val" else 400
                ) if cell["task"] == "krauzlis_cued_motion" else job["n"]
                evaluation = SpatialBatteryStream(job["eval_seed"], job["split"])
                for offset in range(0, n, cfg["batch_size"]):
                    images, labels, metadata = evaluation.batch(
                        min(cfg["batch_size"], n - offset),
                        cell["task"],
                        cell["condition"],
                    )
                    model.attention.capture = True
                    model.attention.captured = []
                    with torch.no_grad():
                        probabilities = (
                            model(images.cuda(), cell["task"]).softmax(1).cpu().tolist()
                        )
                    attention_cells.setdefault(name, []).append(
                        (model.attention.captured, len(labels))
                    )
                    model.attention.capture = False
                    model.attention.captured = []
                    for index, (probability, label, meta) in enumerate(
                        zip(probabilities, labels.tolist(), metadata)
                    ):
                        row = dict(
                            task=cell["task"],
                            condition=name,
                            protocol=cell["task"],
                            label=label,
                            probabilities=probability,
                            base_id=None,
                            paired_base_id=f"{name}/{offset + index}",
                            trial_id=f"{name}/{offset + index}",
                            metadata=meta,
                        )
                        rows.append(row)
                        handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    if time.time() > deadline - 15:
                        raise TimeoutError("Evaluation deadline")
                print(json.dumps(dict(evaluated=name, n=n)), flush=True)
        result = summarize_spatial(rows)
        result["attention_by_condition"] = {
            name: {
                key: np.average(
                    [[row[key] for row in batch] for batch, _ in batches],
                    axis=0,
                    weights=[size for _, size in batches],
                ).tolist()
                for key in (
                    "memory_attention_mass_by_head",
                    "attention_entropy_by_head",
                )
            }
            for name, batches in attention_cells.items()
        }
        result.update(
            status="completed",
            step=step,
            gamma=float(model.attention.gamma.detach()),
            comparison_status=job["comparison_status"],
            comparison_limitation=job["comparison_limitation"],
            predictions=str(out / "predictions.jsonl"),
            checkpoint=job.get("checkpoint") or job["parent"],
        )
        write(out / "summary.json", result)
    else:
        raise ValueError(kind)

    result["worker_seconds"] = time.time() - started
    write(job["result"], result)
    timer.cancel()


if __name__ == "__main__":
    worker(read(sys.argv[1]))
