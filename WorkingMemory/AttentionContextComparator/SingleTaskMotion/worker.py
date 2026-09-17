"""Bounded local worker for scratch AV-context motion-only training."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.SingleTaskMotion.protocol import (
    DELAYS,
    TASK,
    training_name,
)


def run(job):
    import numpy as np
    import torch

    from PreAttentiveVision.evaluate_multitask import summarize
    from WorkingMemory.AttentionContextComparator.model import (
        ARM,
        VERSION,
        initialize_scratch,
    )
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

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

    cfg = job["config"]
    device = torch.device(job["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    random.seed(cfg["model_seed"])
    np.random.seed(cfg["model_seed"])
    torch.manual_seed(cfg["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["model_seed"])
    model, optimizer, lineage = initialize_scratch(ARM, cfg)
    if optimizer.state:
        raise RuntimeError("Scratch optimizer unexpectedly has state")
    stream = SpatialBatteryStream(cfg["train_seed"], "train")
    step = 0
    counts = Counter(episodes=0, frames=0)
    cell_counts = Counter({f"D{delay}": 0 for delay in DELAYS})

    def rng_state():
        return dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=(
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None
            ),
        )

    def restore_rng(saved):
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda"])

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
            or saved["training_task"] != TASK
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
        restore_rng(saved["rng"])
        del saved

    model.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    initialization_path = out / "initialization.json"
    if not initialization_path.exists():
        write(
            initialization_path,
            dict(
                lineage=lineage,
                architecture_version=VERSION,
                training_task=TASK,
                full_suite_architecture_config_retained=True,
                loaded_parent=False,
                loaded_model_tensors=0,
                inherited_optimizer_states=0,
                optimizer_initially_empty=True,
                model_seed=cfg["model_seed"],
                train_seed=cfg["train_seed"],
                scheduler_seed=cfg["scheduler_seed"],
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            ),
        )

    def save_checkpoint():
        path = out / f"checkpoint_{step:06d}.pt"
        if not path.exists():
            payload = dict(
                version=VERSION,
                arm=ARM,
                training_task=TASK,
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
                handle.write(
                    json.dumps(
                        dict(file=path.name, step=step, sha256=sha(path))
                    )
                    + "\n"
                )
        return str(path)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def train_batch(delay, diagnostic=False):
        images, labels, metadata = stream.batch(
            cfg["batch_size"], TASK, dict(delay=delay)
        )
        images = images.to(device)
        labels_device = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        tick = time.time()
        if diagnostic:
            logits, diagnostics = model(images, TASK, True)
        else:
            logits = model(images, TASK)
            diagnostics = None
        loss = torch.nn.functional.cross_entropy(logits, labels_device)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["clip"]
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("Nonfinite loss or gradient")
        detail = {}
        if diagnostic:
            detail = dict(
                first_field_gradient=float(diagnostics["first_field"].grad.norm()),
                attention_gradient_norm=float(
                    sum(
                        p.grad.square().sum()
                        for p in model.attention.parameters()
                        if p.grad is not None
                    )
                    .sqrt()
                    .detach()
                ),
                priority_gradient_norm=float(
                    sum(
                        p.grad.square().sum()
                        for p in model.priority_readout.parameters()
                        if p.grad is not None
                    )
                    .sqrt()
                    .detach()
                ),
                source_bias_gradient=float(model.attention.source_bias.grad.norm()),
                locality_gradient=float(model.attention.raw_locality.grad.norm()),
            )
        optimizer.step()
        synchronize()
        return dict(
            delay=delay,
            loss=float(loss.detach()),
            accuracy=float((logits.argmax(1).cpu() == labels).float().mean()),
            gradient_norm=float(gradient_norm),
            clipped=int(float(gradient_norm) > cfg["clip"]),
            frames=int(images.shape[1]),
            seconds=time.time() - tick,
            diagnostics=detail,
            first_trial_id=metadata[0]["trial_id"],
        )

    started = time.time()
    kind = job["kind"]
    if kind == "profile":
        rows = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        for delay in DELAYS:
            rows.append(train_batch(delay, diagnostic=(delay == DELAYS[0])))
        result = dict(
            status="completed",
            rows=rows,
            mean_update_seconds=float(np.mean([row["seconds"] for row in rows])),
            peak_allocated_bytes=(
                torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            ),
            peak_reserved_bytes=(
                torch.cuda.max_memory_reserved() if device.type == "cuda" else None
            ),
            device=str(device),
            profile_exposure_discarded=True,
        )
    elif kind == "train":
        if not checkpoint_path:
            save_checkpoint()
        metrics_path = out / "metrics.csv"
        columns = (
            "step",
            "episodes",
            "frames_total",
            "condition",
            "delay",
            "loss",
            "accuracy",
            "gradient_norm",
            "clipped",
            "frames",
            "seconds",
            "diagnostics",
            "first_trial_id",
        )
        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            if handle.tell() == 0:
                writer.writeheader()
            model.train()
            while step < int(job["target"]):
                if time.time() > deadline - 20:
                    break
                name = training_name(cfg, step)
                delay = int(cfg["cells"][name]["condition"]["delay"])
                value = train_batch(delay, diagnostic=(step % 128 == 0))
                step += 1
                counts["episodes"] += cfg["batch_size"]
                counts["frames"] += cfg["batch_size"] * value["frames"]
                cell_counts[f"D{delay}"] += cfg["batch_size"]
                value["diagnostics"] = json.dumps(value["diagnostics"])
                writer.writerow(
                    dict(
                        step=step,
                        episodes=counts["episodes"],
                        frames_total=counts["frames"],
                        condition=name,
                        **value,
                    )
                )
                handle.flush()
                if step % 32 == 0:
                    print(
                        json.dumps(
                            dict(
                                step=step,
                                episodes=counts["episodes"],
                                condition=name,
                                loss=value["loss"],
                                accuracy=value["accuracy"],
                            )
                        ),
                        flush=True,
                    )
                if step % 256 == 0:
                    save_checkpoint()
        result = dict(
            status=(
                "completed" if step == int(job["target"]) else "budget_stopped"
            ),
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
        prediction_path = out / "predictions.jsonl"
        with prediction_path.open("w") as handle:
            for delay in DELAYS:
                evaluation = SpatialBatteryStream(job["eval_seed"], job["split"])
                for offset in range(0, int(job["n"]), cfg["batch_size"]):
                    size = min(cfg["batch_size"], int(job["n"]) - offset)
                    images, labels, metadata = evaluation.batch(
                        size, TASK, dict(delay=delay)
                    )
                    with torch.no_grad():
                        probabilities = (
                            model(images.to(device), TASK).softmax(1).cpu().tolist()
                        )
                    for index, (probability, label, meta) in enumerate(
                        zip(probabilities, labels.tolist(), metadata)
                    ):
                        row = dict(
                            task=TASK,
                            condition=f"D{delay}",
                            protocol=TASK,
                            label=label,
                            probabilities=probability,
                            base_id=None,
                            paired_base_id=str(offset + index),
                            trial_id=f"{offset + index}/D{delay}",
                            metadata=meta,
                        )
                        rows.append(row)
                        handle.write(json.dumps(row) + "\n")
                    if time.time() > deadline - 20:
                        raise TimeoutError("Evaluation deadline")
                handle.flush()
        cells = {
            f"D{delay}": summarize(
                [row for row in rows if row["condition"] == f"D{delay}"], 0
            )
            for delay in DELAYS
        }
        result = dict(
            status="completed",
            step=step,
            cells=cells,
            rank=[
                min(cell["balanced_accuracy"] for cell in cells.values()),
                float(np.mean([cell["macro_ovr_auc"] for cell in cells.values()])),
            ],
            predictions=str(prediction_path),
            predictions_sha256=sha(prediction_path),
            episodes=len(rows),
            checkpoint=checkpoint_path,
        )
        write(out / "summary.json", result)
    else:
        raise ValueError(kind)
    result["worker_seconds"] = time.time() - started
    write(job["result"], result)
    timer.cancel()


if __name__ == "__main__":
    run(read(sys.argv[1]))
