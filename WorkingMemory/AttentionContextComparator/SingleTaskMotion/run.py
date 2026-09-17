"""Finite local supervisor for scratch AV-context motion acquisition."""
import datetime
import json
import os
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
from WorkingMemory.AttentionContextComparator.SingleTaskMotion.protocol import (
    RUNTIME_SECONDS,
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TASK,
    TEST_N,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    VALIDATION_TARGETS,
    recipe,
)


def latest_checkpoint(directory, target):
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
    if {path.name for path in files} != seen:
        raise RuntimeError("Checkpoint/index inventory mismatch")
    eligible = [row for row in rows if int(row["step"]) <= int(target)]
    return (
        str(directory / max(eligible, key=lambda row: int(row["step"]))["file"])
        if eligible
        else None
    )


def run():
    import torch

    cfg = recipe()
    run_root = HERE / "runs" / datetime.datetime.now().strftime(
        "scratch_motion_%Y%m%d_%H%M%S"
    )
    run_root.mkdir(parents=True)
    started = time.time()
    deadline = started + RUNTIME_SECONDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_paths = (
        "WorkingMemory/AttentionContextComparator/SingleTaskMotion/protocol.py",
        "WorkingMemory/AttentionContextComparator/SingleTaskMotion/worker.py",
        "WorkingMemory/AttentionContextComparator/SingleTaskMotion/check.py",
        "WorkingMemory/AttentionContextComparator/SingleTaskMotion/run.py",
        "WorkingMemory/AttentionContextComparator/model.py",
        "WorkingMemory/SpatialPriorityReadout/model.py",
        "WorkingMemory/PreUpdateAttention/model.py",
        "WorkingMemory/SpatialTaskBattery/stimuli.py",
        "WorkingMemory/UnbiasedAttention/protocol.py",
        "WorkingMemory/TrainingExposure/protocol.py",
        "PreAttentiveVision/evaluate_multitask.py",
    )
    source_hashes = {name: sha(ROOT / name) for name in source_paths}
    budget = dict(
        status="construction",
        started_unix=started,
        deadline_unix=deadline,
        runtime_seconds=RUNTIME_SECONDS,
        device=device,
        active=None,
        events=[],
        source_hashes=source_hashes,
    )
    aggregate = dict(
        status="construction",
        run_root=str(run_root),
        task=TASK,
        architecture_version="attention_context_comparator_scratch_v1",
        initialization_kind="full_model_and_optimizer_from_scratch",
        loaded_parent=False,
        loaded_model_tensors=0,
        inherited_optimizer_states=0,
        requested_exposure=dict(updates=UPDATES, episodes=TOTAL_EPISODES),
        validation_targets=list(VALIDATION_TARGETS),
        validation_n_per_delay=VALIDATION_N,
        test_n_per_delay=TEST_N,
        config=cfg,
        source_hashes=source_hashes,
        platform=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            platform=platform.platform(),
            device=device,
            gpu=(
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        ),
        validation=[],
        selection=(
            "Maximum [minimum delay balanced accuracy, mean delay macro OVR AUC], "
            "then earlier update."
        ),
        interpretation=(
            "Scratch single-task acquisition test. Full motion CE per update and "
            "absence of other task gradients differ from the five-task optimization "
            "trajectory; results do not by themselves identify a gradient-conflict mechanism."
        ),
    )
    write(HERE / "local_run.json", dict(run=str(run_root), deadline_unix=deadline))

    def publish():
        write(run_root / "budget.json", budget)
        write(run_root / "aggregate.json", aggregate)
        write(
            run_root / "live_status.json",
            dict(
                status=aggregate["status"],
                active=budget["active"],
                pinned_updates=aggregate.get("pinned_updates"),
                selected_step=aggregate.get("selected_step"),
                validation_steps=[
                    value["step"] for value in aggregate["validation"]
                ],
                updated_unix=time.time(),
            ),
        )

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            checkpoint = latest_checkpoint(out, extra["target"])
            if checkpoint:
                extra["checkpoint"] = checkpoint
        job_path = run_root / f"{tag}_job.json"
        result_path = run_root / f"{tag}_result.json"
        log_path = run_root / f"{tag}.log"
        job = dict(
            kind=kind,
            config=cfg,
            out=str(out),
            result=str(result_path),
            deadline=deadline - 120,
            device=device,
            source_hashes=source_hashes,
            **extra,
        )
        write(job_path, job)
        tick = time.time()
        with log_path.open("w") as handle:
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-B", str(HERE / "worker.py"), str(job_path)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            budget["active"] = dict(
                pid=process.pid,
                tag=tag,
                kind=kind,
                log=str(log_path),
                started_unix=tick,
            )
            publish()
            try:
                code = process.wait(
                    timeout=max(0.01, deadline - time.time() - 120)
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        budget["events"].append(
            dict(
                pid=process.pid,
                tag=tag,
                kind=kind,
                returncode=code,
                seconds=time.time() - tick,
            )
        )
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
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        (run_root / "construction_stdout.log").write_text(construction.stdout)
        (run_root / "construction_stderr.log").write_text(construction.stderr)
        if construction.returncode:
            raise RuntimeError("Construction checks failed")
        aggregate["construction_checks"] = read(HERE / "construction_checks.json")
        aggregate["status"] = budget["status"] = "profiling"
        publish()

        profile = launch("profile", "profile", run_root / "profile")
        aggregate["profile"] = profile
        mean_update = float(profile["mean_update_seconds"])
        validation_projection = (
            len(VALIDATION_TARGETS)
            * VALIDATION_N
            * sum(11 + delay for delay in (0, 4, 12, 24))
            / (8 * sum(11 + delay for delay in (0, 4, 12, 24)))
            * mean_update
        )
        test_projection = TEST_N / VALIDATION_N * validation_projection / len(
            VALIDATION_TARGETS
        )
        reserve = max(900.0, 1.5 * (validation_projection + test_projection) + 300)
        available = deadline - time.time() - reserve - 120
        conservative_update = max(0.01, 1.35 * mean_update)
        feasible = int(available / conservative_update)
        pinned_updates = min(UPDATES, feasible // 800 * 800)
        if pinned_updates < 800:
            raise RuntimeError("Local profile cannot support one useful validation interval")
        targets = [value for value in VALIDATION_TARGETS if value <= pinned_updates]
        aggregate.update(
            pinned_updates=pinned_updates,
            pinned_episodes=pinned_updates * cfg["batch_size"],
            pinned_targets=targets,
            exposure_reduced=(pinned_updates != UPDATES),
            profile_projection=dict(
                mean_update_seconds=mean_update,
                conservative_update_seconds=conservative_update,
                projected_training_seconds=pinned_updates * conservative_update,
                evaluation_reporting_reserve_seconds=reserve,
                remaining_seconds=deadline - time.time(),
            ),
        )
        write(
            run_root / "fixed_config.json",
            {
                key: aggregate[key]
                for key in (
                    "task",
                    "architecture_version",
                    "initialization_kind",
                    "requested_exposure",
                    "pinned_updates",
                    "pinned_episodes",
                    "pinned_targets",
                    "exposure_reduced",
                    "validation_n_per_delay",
                    "test_n_per_delay",
                    "config",
                    "profile_projection",
                    "platform",
                )
            },
        )
        aggregate["status"] = budget["status"] = "training"
        publish()

        training_out = run_root / "training"
        checkpoint = None
        selected = None
        for target in targets:
            trained = launch(
                f"train_{target}", "train", training_out, target=target
            )
            checkpoint = trained["checkpoint"]
            aggregate["training"] = trained
            validation = launch(
                f"validation_{target}",
                "eval",
                run_root / f"validation_{target}",
                checkpoint=checkpoint,
                split="val",
                eval_seed=SPATIAL_VAL_SEED,
                n=VALIDATION_N,
            )
            aggregate["validation"].append(validation)
            selected = max(
                aggregate["validation"],
                key=lambda value: (tuple(value["rank"]), -value["step"]),
            )
            aggregate["selected_step"] = selected["step"]
            aggregate["selected_checkpoint"] = selected["checkpoint"]
            publish()

        aggregate["terminal_checkpoint"] = checkpoint
        aggregate["status"] = budget["status"] = "evaluating"
        publish()
        aggregate["selected_test"] = launch(
            "selected_test",
            "eval",
            run_root / "selected_test",
            checkpoint=selected["checkpoint"],
            split="test",
            eval_seed=SPATIAL_TEST_SEED,
            n=TEST_N,
        )
        aggregate["terminal_test"] = (
            aggregate["selected_test"]
            if selected["step"] == pinned_updates
            else launch(
                "terminal_test",
                "eval",
                run_root / "terminal_test",
                checkpoint=checkpoint,
                split="test",
                eval_seed=SPATIAL_TEST_SEED,
                n=TEST_N,
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
            dict(
                status=aggregate["status"],
                error=aggregate.get("error"),
                wall_seconds=aggregate["wall_seconds"],
                deadline_unix=deadline,
            ),
        )
        write(
            run_root / "artifact_index.json",
            {
                str(path.relative_to(run_root)).replace("\\", "/"): sha(path)
                for path in run_root.rglob("*")
                if path.is_file() and path.name != "artifact_index.json"
            },
        )


if __name__ == "__main__":
    run()
