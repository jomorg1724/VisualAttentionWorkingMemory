"""User-authorized checkpoint-preserving extension from 4,000 to 12,000 updates."""
import json
import os
import re
import subprocess
import sys
import time
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write

TOTAL_TARGET = 12000
TOTAL_EPISODES = 96000
ALL_TARGETS = tuple(range(800, TOTAL_TARGET + 1, 800))
EXTENSION_TARGETS = tuple(range(4800, TOTAL_TARGET + 1, 800))
TOTAL_RUNTIME_SECONDS = 21600


def process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except (OSError, SystemError):
        return False
    return True


def latest_checkpoint(directory, target):
    directory = Path(directory)
    index = directory / "checkpoint_index.jsonl"
    files = set(directory.glob("checkpoint_*.pt")) if directory.exists() else set()
    if not index.exists():
        if files:
            raise RuntimeError("Unindexed checkpoint exists")
        return None, -1
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
    if not eligible:
        return None, -1
    selected = max(eligible, key=lambda row: int(row["step"]))
    return str(directory / selected["file"]), int(selected["step"])


def run():
    local = read(HERE / "local_run.json")
    run_root = Path(local["run"])
    original_budget = read(run_root / "budget.json")
    original_deadline = float(
        original_budget.get("original_deadline_unix", original_budget["deadline_unix"])
    )
    new_deadline = float(original_budget["started_unix"]) + TOTAL_RUNTIME_SECONDS
    if new_deadline < original_deadline:
        raise RuntimeError("Extension deadline must exceed original deadline")
    aggregate = read(run_root / "aggregate.json")
    cfg = aggregate["config"]
    source_hashes = aggregate["source_hashes"]
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned trajectory source changed " + name)

    authorization = dict(
        status="authorized_and_supervised",
        authorization="User explicitly extended the live trajectory post-launch.",
        protocol_change="Total exposure only: 4000 to 12000 optimizer updates.",
        original_total_updates=4000,
        original_total_episodes=32000,
        authorized_total_updates=TOTAL_TARGET,
        authorized_total_episodes=TOTAL_EPISODES,
        original_validation_targets=list(range(800, 4001, 800)),
        effective_validation_targets=list(ALL_TARGETS),
        extension_validation_targets=list(EXTENSION_TARGETS),
        preserve_validation_800=True,
        restart_or_reset=False,
        continuation=(
            "Resume exact indexed checkpoint with model, Adam, RNG, task-local "
            "stream and counters restored by the pinned worker."
        ),
        original_deadline_unix=original_deadline,
        extended_deadline_unix=new_deadline,
        original_runtime_seconds=original_budget["runtime_seconds"],
        extended_total_runtime_seconds=TOTAL_RUNTIME_SECONDS,
        deadline_extension_seconds=new_deadline - original_deadline,
        source_hashes=source_hashes,
        extension_supervisor_sha256=sha(Path(__file__)),
        recorded_unix=time.time(),
    )
    write(run_root / "protocol_extension.json", authorization)
    status = dict(
        status="waiting_for_original_supervisor",
        effective_target_updates=TOTAL_TARGET,
        effective_target_episodes=TOTAL_EPISODES,
        validation_targets=list(ALL_TARGETS),
        original_deadline_unix=original_deadline,
        extended_deadline_unix=new_deadline,
        active=None,
        events=[],
        updated_unix=time.time(),
    )

    def publish():
        status["updated_unix"] = time.time()
        write(run_root / "extension_status.json", status)

    publish()
    while time.time() < new_deadline - 120:
        current_budget = read(run_root / "budget.json")
        current_aggregate = read(run_root / "aggregate.json")
        active = current_budget.get("active")
        active_alive = process_alive(active.get("pid")) if active else False
        if current_aggregate.get("status") == "completed" and not active:
            status["handoff_reason"] = "original_4000_supervisor_completed"
            break
        if current_aggregate.get("status") == "failed" and not active_alive:
            status["handoff_reason"] = "recovering_original_supervisor_failure"
            break
        if active and not active_alive:
            _, checkpoint_step = latest_checkpoint(run_root / "training", TOTAL_TARGET)
            if checkpoint_step >= 0:
                status["handoff_reason"] = "recovering_interrupted_original_session"
                break
        time.sleep(10)
    else:
        raise TimeoutError("Extension handoff deadline expired")

    aggregate = read(run_root / "aggregate.json")
    budget = read(run_root / "budget.json")
    training_out = run_root / "training"
    checkpoint, checkpoint_step = latest_checkpoint(training_out, TOTAL_TARGET)
    if checkpoint_step < 800:
        raise RuntimeError("No validated checkpoint available for safe extension")
    preserved_validation = {
        int(value["step"]): value for value in aggregate.get("validation", [])
    }
    if 800 not in preserved_validation:
        preserved = run_root / "validation_800" / "summary.json"
        if not preserved.exists():
            raise RuntimeError("Validation 800 was not preserved")
        preserved_validation[800] = read(preserved)

    aggregate.setdefault(
        "pre_extension_protocol",
        dict(
            requested_exposure=aggregate.get("requested_exposure"),
            validation_targets=aggregate.get("validation_targets"),
            pinned_updates=aggregate.get("pinned_updates"),
            pinned_episodes=aggregate.get("pinned_episodes"),
            original_deadline_unix=original_deadline,
        ),
    )
    aggregate.setdefault(
        "pre_extension_evaluation",
        dict(
            selected_step=aggregate.get("selected_step"),
            selected_checkpoint=aggregate.get("selected_checkpoint"),
            selected_test=aggregate.get("selected_test"),
            terminal_test=aggregate.get("terminal_test"),
        ),
    )
    aggregate.update(
        status="extended_training",
        requested_exposure=dict(updates=TOTAL_TARGET, episodes=TOTAL_EPISODES),
        validation_targets=list(ALL_TARGETS),
        pinned_updates=TOTAL_TARGET,
        pinned_episodes=TOTAL_EPISODES,
        pinned_targets=list(ALL_TARGETS),
        exposure_reduced=False,
        post_launch_protocol_extension=authorization,
        validation=[preserved_validation[key] for key in sorted(preserved_validation)],
    )
    budget.update(
        status="extended_training",
        original_deadline_unix=original_deadline,
        deadline_unix=new_deadline,
        runtime_seconds=TOTAL_RUNTIME_SECONDS,
        active=None,
    )

    def publish_all():
        publish()
        write(run_root / "budget.json", budget)
        write(run_root / "aggregate.json", aggregate)
        write(
            run_root / "live_status.json",
            dict(
                status=aggregate["status"],
                active=budget["active"],
                effective_target_updates=TOTAL_TARGET,
                effective_target_episodes=TOTAL_EPISODES,
                selected_step=aggregate.get("selected_step"),
                validation_steps=[
                    value["step"] for value in aggregate["validation"]
                ],
                protocol_extension=str(run_root / "protocol_extension.json"),
                updated_unix=time.time(),
            ),
        )

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            resume, _ = latest_checkpoint(out, extra["target"])
            if resume:
                extra["checkpoint"] = resume
        job_path = run_root / f"{tag}_job.json"
        result_path = run_root / f"{tag}_result.json"
        log_path = run_root / f"{tag}.log"
        job = dict(
            kind=kind,
            config=cfg,
            out=str(out),
            result=str(result_path),
            deadline=new_deadline - 120,
            device=budget["device"],
            source_hashes=source_hashes,
            **extra,
        )
        if result_path.exists() and job_path.exists():
            result = read(result_path)
            if result.get("status") == "completed" and read(job_path) == job:
                return result
        write(job_path, job)
        tick = time.time()
        with log_path.open("a") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    str(HERE / "worker.py"),
                    str(job_path),
                ],
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
            status["active"] = budget["active"]
            publish_all()
            try:
                code = process.wait(
                    timeout=max(0.01, new_deadline - time.time() - 120)
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        event = dict(
            pid=process.pid,
            tag=tag,
            kind=kind,
            returncode=code,
            seconds=time.time() - tick,
        )
        budget["events"].append(event)
        status["events"].append(event)
        budget["active"] = None
        status["active"] = None
        publish_all()
        if code:
            raise RuntimeError(f"{tag} failed with exit {code}; see {log_path}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError(f"{tag} did not complete: {result['status']}")
        return result

    publish_all()
    try:
        for target in ALL_TARGETS:
            if target <= checkpoint_step and target in preserved_validation:
                continue
            trained = launch(
                f"extension_train_{target}",
                "train",
                training_out,
                target=target,
            )
            checkpoint = trained["checkpoint"]
            checkpoint_step = int(trained["step"])
            aggregate["training"] = trained
            validation_path = run_root / f"validation_{target}"
            validation = launch(
                f"extension_validation_{target}",
                "eval",
                validation_path,
                checkpoint=checkpoint,
                split="val",
                eval_seed=63973001,
                n=64,
            )
            preserved_validation[target] = validation
            aggregate["validation"] = [
                preserved_validation[key] for key in sorted(preserved_validation)
            ]
            selected = max(
                aggregate["validation"],
                key=lambda value: (tuple(value["rank"]), -value["step"]),
            )
            aggregate["selected_step"] = selected["step"]
            aggregate["selected_checkpoint"] = selected["checkpoint"]
            publish_all()

        aggregate["terminal_checkpoint"] = checkpoint
        aggregate["status"] = budget["status"] = status["status"] = (
            "extended_evaluating"
        )
        publish_all()
        selected = max(
            aggregate["validation"],
            key=lambda value: (tuple(value["rank"]), -value["step"]),
        )
        aggregate["selected_test"] = launch(
            "extension_selected_test",
            "eval",
            run_root / "extended_selected_test",
            checkpoint=selected["checkpoint"],
            split="test",
            eval_seed=64973001,
            n=256,
        )
        aggregate["terminal_test"] = (
            aggregate["selected_test"]
            if selected["step"] == TOTAL_TARGET
            else launch(
                "extension_terminal_test",
                "eval",
                run_root / "extended_terminal_test",
                checkpoint=checkpoint,
                split="test",
                eval_seed=64973001,
                n=256,
            )
        )
        aggregate["status"] = budget["status"] = status["status"] = "completed"
    except BaseException as error:
        aggregate.update(status="failed", extension_error=repr(error))
        budget["status"] = status["status"] = "failed"
        raise
    finally:
        aggregate["extended_wall_seconds"] = time.time() - authorization["recorded_unix"]
        publish_all()
        write(
            run_root / "extension_exit.json",
            dict(
                status=status["status"],
                error=aggregate.get("extension_error"),
                effective_target_updates=TOTAL_TARGET,
                extended_deadline_unix=new_deadline,
                wall_seconds=aggregate["extended_wall_seconds"],
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
