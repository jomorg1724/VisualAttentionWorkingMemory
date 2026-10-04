"""CPU-only live-checkpoint diagnostic; never alters model selection or a run.

python -m SecondPass.EarlyValidation.snapshot delayed_gru RUN --output OUT
    --deadline UNIX_SECONDS
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from SecondPass.SequenceKDA16.worker import FreshStream, TASK, atomic_json, cpu_setup, evaluate


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_model(architecture: str, config: dict):
    if architecture == "kda16":
        from SecondPass.SequenceKDA16.model import SequenceKDA
        return SequenceKDA(backend=config.get("backend", "chunk"), chunk_size=config.get("chunk_size", 32))
    if architecture == "delayed_gru":
        from SecondPass.DelayedFrameGRU.model import DelayedFrameGRU
        return DelayedFrameGRU()
    raise ValueError("unknown architecture")


def validate_output(run: Path, output: Path) -> None:
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("snapshot output must be outside the active run directory")


def snapshot(architecture: str, run: Path, output: Path, deadline: float) -> dict:
    cpu_setup()
    run, output = run.resolve(), output.resolve()
    validate_output(run, output)
    started = time.time()
    receipt = json.loads((run / "latest_checkpoint.json").read_text())
    checkpoint = Path(receipt["path"])
    if not checkpoint.resolve().is_relative_to(run) or checkpoint.is_symlink():
        raise ValueError("checkpoint must be an immutable file in this run")
    if checkpoint.stat().st_size != receipt["bytes"] or digest(checkpoint) != receipt["sha256"]:
        raise ValueError("checkpoint integrity mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["state"]["step"] != receipt["step"] or receipt["step"] <= 0:
        raise ValueError("checkpoint lacks persisted training progress")
    config = saved["state"]["config"]
    deadline = min(float(deadline), float(config["deadline"]), started + 1200)
    if time.time() >= deadline:
        raise ValueError("snapshot deadline expired")
    model = make_model(architecture, config)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.eval().requires_grad_(False)
    # The longest native movie provides a conservative per-trial timing signal.
    profile_started = time.perf_counter()
    images, _, metadata = FreshStream("val").batch(1, TASK, "B28")
    with torch.inference_mode():
        logits = model(images, TASK)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite profile logits")
    seconds = time.perf_counter() - profile_started
    del images, logits
    remaining = deadline - time.time()
    count = 100 if 3 * 100 * seconds * 1.35 + 60 < remaining else 50
    if 3 * count * seconds * 1.35 + 30 >= remaining:
        raise RuntimeError("even 50 trials per condition cannot fit the CPU snapshot deadline")
    profile = dict(architecture=architecture, checkpoint=receipt, cpu_threads=2, device="cpu",
        native_profile_cell="B28", native_profile_frames=metadata[0]["frame_count"],
        native_profile_trial_seconds=seconds, trials_per_condition=count,
        preferred_trials_per_condition=100, reduced_for_measured_cost=count < 100,
        profile_draw_repeated_in_evaluation=True, deadline=deadline,
        diagnostic=True, affects_selection=False, training_unchanged=True)
    atomic_json(str(output) + ".profile.json", profile)
    result = evaluate(model, "val", count, 1, "cpu", output, deadline)
    unchanged = checkpoint.stat().st_size == receipt["bytes"] and digest(checkpoint) == receipt["sha256"]
    if not unchanged:
        raise ValueError("immutable checkpoint changed during read-only evaluation")
    result.update(architecture=architecture, checkpoint=receipt, checkpoint_step=receipt["step"],
        checkpoint_sha256=receipt["sha256"], diagnostic=True, affects_selection=False,
        weights_unchanged=True, checkpoint_bytes_unchanged=True, device="cpu", cpu_threads=2,
        training_unchanged=True, no_gpu=True, profile=profile, elapsed_seconds=time.time() - started,
        interpretation="Live validation snapshot only; not selected or final held-out performance.")
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=("kda16", "delayed_gru"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = snapshot(args.architecture, args.run, args.output, args.deadline)
    print(json.dumps({key: result[key] for key in ("architecture", "checkpoint_step", "complete", "n", "summary", "diagnostic", "affects_selection", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
