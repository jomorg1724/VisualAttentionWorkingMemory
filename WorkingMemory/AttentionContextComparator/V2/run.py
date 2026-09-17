"""Finite local supervisor for the AV-context v2 five-task scratch arm.

Sequence: construction/identity gate -> 3-update profile -> train/validate at
every 800 updates -> pre-registered stopping rule at validation 2,400 ->
(if passed) continue to 12,000 -> held-out test of selected and terminal.
"""
import datetime
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.V2.model import ARM, VERSION
from WorkingMemory.AttentionContextComparator.V2.protocol import (
    GATE_STEP,
    RUNTIME_SECONDS,
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    VALIDATION_TARGETS,
    assert_fixed,
    gate,
    recipe,
)

SOURCE_PATHS = (
    "WorkingMemory/AttentionContextComparator/V2/model.py",
    "WorkingMemory/AttentionContextComparator/V2/protocol.py",
    "WorkingMemory/AttentionContextComparator/V2/check.py",
    "WorkingMemory/AttentionContextComparator/V2/worker.py",
    "WorkingMemory/AttentionContextComparator/V2/run.py",
    "WorkingMemory/AttentionContextComparator/model.py",
    "WorkingMemory/AttentionContextComparator/protocol.py",
    "WorkingMemory/SpatialPriorityReadout/model.py",
    "WorkingMemory/PreUpdateAttention/model.py",
    "WorkingMemory/SpatialComparison/model.py",
    "WorkingMemory/RecurrentComparison/model.py",
    "WorkingMemory/model.py",
    "PreAttentiveVision/TemporalIntegration/accumulators.py",
    "PreAttentiveVision/decoder.py",
    "PreAttentiveVision/hybrid_models.py",
    "PreAttentiveVision/models.py",
    "WorkingMemory/SpatialTaskBattery/stimuli.py",
    "WorkingMemory/UnbiasedAttention/protocol.py",
    "WorkingMemory/TrainingExposure/protocol.py",
    "PreAttentiveVision/evaluate_multitask.py",
)


def newest_checkpoint(directory, target):
    directory = Path(directory)
    index = directory / "checkpoint_index.jsonl"
    files = set(directory.glob("checkpoint_*.pt")) if directory.exists() else set()
    if not index.exists():
        if files:
            raise RuntimeError("Unindexed checkpoint exists")
        return None
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    seen = set()
    for row in rows:
        path = directory / row.get("file", "")
        if (
            not re.fullmatch(r"checkpoint_[0-9]{6}\.pt", row.get("file", ""))
            or row["file"] in seen
            or not path.is_file()
            or sha(path) != row.get("sha256")
            or int(path.stem.rsplit("_", 1)[1]) != int(row["step"])
        ):
            raise RuntimeError("Malformed checkpoint index")
        seen.add(row["file"])
    if {p.name for p in files} != seen:
        raise RuntimeError("Checkpoint/index inventory mismatch")
    eligible = [row for row in rows if int(row["step"]) <= int(target)]
    return str(directory / max(eligible, key=lambda r: int(r["step"]))["file"]) if eligible else None


def run():
    import torch

    cfg = recipe()
    assert_fixed(cfg)
    run_root = HERE / "runs" / datetime.datetime.now().strftime("v2_local_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True)
    started = time.time()
    deadline = started + RUNTIME_SECONDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_PATHS}
    budget = dict(
        status="construction", started_unix=started, deadline_unix=deadline,
        runtime_seconds=RUNTIME_SECONDS, device=device, active=None, events=[],
        source_hashes=source_hashes,
    )
    aggregate = dict(
        status="construction",
        run_root=str(run_root),
        arm=ARM,
        version=VERSION,
        base_version="attention_context_comparator_scratch_v1",
        initialization_kind="full_model_and_optimizer_from_scratch",
        loaded_parent=False,
        v2_changes=cfg["v2_changes"],
        learning_rates=dict(new_lr=cfg["new_lr"], parent_lr=cfg["parent_lr"]),
        requested_exposure=dict(updates=UPDATES, episodes=TOTAL_EPISODES),
        validation_targets=list(VALIDATION_TARGETS),
        validation_n_per_cell=VALIDATION_N,
        test_n_per_cell=TEST_N,
        gate_step=GATE_STEP,
        config=cfg,
        source_hashes=source_hashes,
        platform=dict(
            python=platform.python_version(), torch=torch.__version__,
            platform=platform.platform(), device=device,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        ),
        validation=[],
        selection="maximum [minimum chance-normalized task BA, mean task AUC], then earlier step",
        interpretation=(
            "Combined five-change arm (A-E). A positive result shows the combination "
            "works, not which change did it; leave-one-out ablations are deferred "
            "until a positive result exists."
        ),
    )
    write(HERE / "local_run.json", dict(run=str(run_root), deadline_unix=deadline))

    def publish():
        write(run_root / "budget.json", budget)
        write(run_root / "aggregate.json", aggregate)
        write(
            run_root / "live_status.json",
            dict(
                status=aggregate["status"], active=budget["active"],
                selected_step=aggregate.get("selected_step"),
                validation_steps=[v["step"] for v in aggregate["validation"]],
                gate=aggregate.get("gate"), updated_unix=time.time(),
                deadline_unix=deadline,
            ),
        )

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            checkpoint = newest_checkpoint(out, extra["target"])
            if checkpoint:
                extra["checkpoint"] = checkpoint
        job_path = run_root / f"{tag}_job.json"
        result_path = run_root / f"{tag}_result.json"
        log_path = run_root / f"{tag}.log"
        job = dict(
            kind=kind, arm=ARM, config=cfg, out=str(out), result=str(result_path),
            deadline=deadline - 120, device=device, source_hashes=source_hashes, **extra,
        )
        write(job_path, job)
        tick = time.time()
        with log_path.open("w") as handle:
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-B", str(HERE / "worker.py"), str(job_path)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            budget["active"] = dict(pid=process.pid, tag=tag, kind=kind, log=str(log_path), started_unix=tick)
            publish()
            try:
                code = process.wait(timeout=max(0.01, deadline - time.time() - 120))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        budget["events"].append(dict(pid=process.pid, tag=tag, kind=kind, returncode=code, seconds=time.time() - tick))
        budget["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} failed with exit {code}; see {log_path}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError(f"{tag} did not complete: {result['status']}")
        return result

    publish()
    try:
        construction = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(HERE / "check.py")],
            cwd=ROOT, capture_output=True, text=True, timeout=1200,
        )
        (run_root / "construction_stdout.log").write_text(construction.stdout)
        (run_root / "construction_stderr.log").write_text(construction.stderr)
        if construction.returncode:
            raise RuntimeError("Construction/identity gate failed")
        checks = read(HERE / "construction_checks.json")
        if checks["status"] != "passed" or not checks["identity_gate"]["bit_identical_logits_cpu"]:
            raise RuntimeError("Identity gate not passed")
        aggregate["construction_checks"] = checks
        aggregate["status"] = budget["status"] = "profiling"
        publish()

        profile = launch("profile", "profile", run_root / "profile", profile_updates=3)
        aggregate["profile"] = profile
        mean_update = float(profile["mean_update_seconds"])
        aggregate["profile_projection"] = dict(
            mean_update_seconds=mean_update,
            projected_training_hours_to_gate=GATE_STEP * mean_update / 3600,
            projected_training_hours_full=UPDATES * mean_update / 3600,
            peak_allocated_gb=(profile["peak_allocated_bytes"] or 0) / 1e9,
            remaining_hours=(deadline - time.time()) / 3600,
        )
        write(
            run_root / "fixed_config.json",
            {k: aggregate[k] for k in (
                "arm", "version", "base_version", "initialization_kind", "v2_changes",
                "learning_rates", "requested_exposure", "validation_targets", "gate_step",
                "validation_n_per_cell", "test_n_per_cell", "config", "profile_projection", "platform",
            )},
        )
        aggregate["status"] = budget["status"] = "training"
        publish()

        training_out = run_root / "training"
        checkpoint = None
        selected = None
        stopped_by_gate = False
        for target in VALIDATION_TARGETS:
            trained = launch(f"train_{target}", "train", training_out, target=target)
            checkpoint = trained["checkpoint"]
            aggregate["training"] = trained
            validation = launch(
                f"validation_{target}", "eval", run_root / f"validation_{target}",
                checkpoint=checkpoint, split="val", eval_seed=SPATIAL_VAL_SEED, n=VALIDATION_N,
            )
            aggregate["validation"].append(validation)
            selected = max(aggregate["validation"], key=lambda v: (tuple(v["rank"]), -v["step"]))
            aggregate["selected_step"] = selected["step"]
            aggregate["selected_checkpoint"] = selected["checkpoint"]
            if target == GATE_STEP:
                verdict = gate(validation)
                aggregate["gate"] = verdict
                write(run_root / "gate_receipt.json", dict(verdict=verdict, validation_summary=str(run_root / f"validation_{target}" / "summary.json"), unix=time.time()))
                publish()
                if not verdict["passed"]:
                    stopped_by_gate = True
                    break
            publish()

        aggregate["terminal_checkpoint"] = checkpoint
        if stopped_by_gate:
            aggregate["status"] = budget["status"] = "stopped_by_preregistered_gate"
            publish()
        else:
            aggregate["status"] = budget["status"] = "evaluating"
            publish()
            aggregate["selected_test"] = launch(
                "selected_test", "eval", run_root / "selected_test",
                checkpoint=selected["checkpoint"], split="test", eval_seed=SPATIAL_TEST_SEED, n=TEST_N,
            )
            aggregate["terminal_test"] = (
                aggregate["selected_test"]
                if selected["step"] == VALIDATION_TARGETS[-1]
                else launch(
                    "terminal_test", "eval", run_root / "terminal_test",
                    checkpoint=checkpoint, split="test", eval_seed=SPATIAL_TEST_SEED, n=TEST_N,
                )
            )
            aggregate["status"] = budget["status"] = "completed"
    except BaseException as error:
        aggregate.update(status="failed", error=repr(error))
        budget["status"] = "failed"
        raise
    finally:
        aggregate["wall_seconds"] = time.time() - started
        publish()
        write(
            run_root / "exit.json",
            dict(status=aggregate["status"], error=aggregate.get("error"), wall_seconds=aggregate["wall_seconds"], deadline_unix=deadline),
        )
        write(
            run_root / "artifact_index.json",
            {
                str(p.relative_to(run_root)).replace("\\", "/"): sha(p)
                for p in run_root.rglob("*")
                if p.is_file() and p.name != "artifact_index.json"
            },
        )


if __name__ == "__main__":
    run()
