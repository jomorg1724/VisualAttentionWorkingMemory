"""Live orchestration handoff for the authorized AV-context 12k extension."""
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.model import ARM
from WorkingMemory.AttentionContextComparator.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    VALIDATION_N,
)

TARGETS = tuple(range(3200, 12001, 800))
ALL_VALIDATIONS = tuple(range(800, 12001, 800))


def newest_checkpoint(directory, target):
    rows = [
        json.loads(line)
        for line in (Path(directory) / "checkpoint_index.jsonl").read_text().splitlines()
    ]
    eligible = [row for row in rows if int(row["step"]) <= target]
    row = max(eligible, key=lambda value: int(value["step"]))
    path = Path(directory) / row["file"]
    if sha(path) != row["sha256"]:
        raise RuntimeError("Checkpoint index hash mismatch")
    return str(path)


def metric_endpoint(training_out):
    rows = list(csv.DictReader((Path(training_out) / "metrics.csv").open(newline="")))
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(1, steps[-1] + 1)):
        raise RuntimeError("Duplicated or skipped scheduler/metric updates")
    return steps[-1], len(rows)


def main():
    root = ROOT / "attention_context_comparator_results"
    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"
    receipt_path = root / "extension_12000_receipt.json"
    request = read(sys.argv[1])
    deadline = float(request["extended_deadline_unix"])
    old_supervisor = int(request["old_supervisor_pid"])
    inherited_worker = int(request["inherited_worker_pid"])
    ledger = read(budget_path)
    aggregate = read(aggregate_path)
    record = aggregate["run"][ARM]
    cfg = record["config"]
    training_out = root / ARM / "training"

    active = ledger.get("active") or {}
    if (
        active.get("tag") != "train_3200"
        or int(active.get("pid", -1)) != inherited_worker
        or not Path(f"/proc/{old_supervisor}").exists()
        or not Path(f"/proc/{inherited_worker}").exists()
    ):
        raise RuntimeError("Expected live train_3200 handoff state not found")
    endpoint, rows = metric_endpoint(training_out)
    if not 2400 < endpoint < 3200 or rows != endpoint:
        raise RuntimeError("Unexpected live metric endpoint")

    aggregate["version"] = "attention_context_comparator_scratch_resume_v1_extended_12000"
    aggregate["fixed_exposure"] = {"updates": 12000, "episodes": 480000}
    aggregate["validation_targets"] = list(ALL_VALIDATIONS)
    aggregate["protocol_extension"] = {
        "status": "armed",
        "kind": "post_hoc_user_authorized_target_extension",
        "authorized_unix": request["authorized_unix"],
        "old_total_updates": 4000,
        "new_total_updates": 12000,
        "preserved_completed_validation_steps": [800, 1600, 2400],
        "remaining_validation_steps": list(TARGETS),
        "trajectory": "unchanged model/optimizer/RNG/stream/counters; orchestration only",
        "rationale": "fairer scratch-versus-trained comparison",
        "deadline_unix": deadline,
    }
    ledger["deadline_unix"] = deadline
    ledger["events"].append(
        {
            "kind": "post_hoc_user_authorized_extension_handoff",
            "unix": time.time(),
            "old_supervisor_pid": old_supervisor,
            "new_supervisor_pid": os.getpid(),
            "inherited_worker_pid": inherited_worker,
            "metric_endpoint": endpoint,
            "deadline_unix": deadline,
        }
    )
    receipt = {
        "status": "armed",
        "armed_unix": time.time(),
        "old_supervisor_pid": old_supervisor,
        "supervisor_pid": os.getpid(),
        "inherited_worker_pid": inherited_worker,
        "live_step_at_handoff": endpoint,
        "old_total_updates": 4000,
        "new_total_updates": 12000,
        "validation_targets": list(ALL_VALIDATIONS),
        "deadline_unix": deadline,
        "trajectory_preservation": [
            "in-flight train_3200 worker",
            "model and optimizer",
            "RNG and task-local stream",
            "episode/frame/cell counters",
            "metrics, validation, and checkpoint history",
        ],
    }
    write(aggregate_path, aggregate)
    write(budget_path, ledger)
    write(receipt_path, receipt)

    os.kill(old_supervisor, signal.SIGTERM)
    time.sleep(1)
    if not Path(f"/proc/{inherited_worker}").exists():
        raise RuntimeError("Inherited training worker did not survive handoff")

    def publish():
        write(budget_path, ledger)
        write(aggregate_path, aggregate)

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            extra["checkpoint"] = newest_checkpoint(out, extra["target"])
        job_path = root / f"{tag}_job.json"
        result_path = root / f"{tag}_result.json"
        log_path = root / f"{tag}.log"
        job = {
            "kind": kind,
            "arm": ARM,
            "config": cfg,
            "out": str(out),
            "result": str(result_path),
            "deadline": deadline - 300,
            "source_hashes": ledger["source_hashes"],
            **extra,
        }
        write(job_path, job)
        tick = time.time()
        with log_path.open("a") as handle:
            process = subprocess.Popen(
                [sys.executable, "-B", str(HERE / "worker.py"), str(job_path)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            ledger["active"] = {
                "pid": process.pid,
                "kind": kind,
                "tag": tag,
                "log": str(log_path),
            }
            publish()
            try:
                code = process.wait(timeout=max(0.01, deadline - time.time() - 300))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        ledger["events"].append(
            {
                "tag": tag,
                "kind": kind,
                "pid": process.pid,
                "returncode": code,
                "seconds": time.time() - tick,
            }
        )
        ledger["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} exited {code}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError(f"{tag} incomplete")
        return result

    def accept_validation(validation):
        by_step = {value["step"]: value for value in record["validation"]}
        by_step[validation["step"]] = validation
        record["validation"] = [by_step[key] for key in sorted(by_step)]
        selected = max(
            record["validation"],
            key=lambda value: (tuple(value["rank"]), -value["step"]),
        )
        record["selected_checkpoint"] = selected["checkpoint"]
        record["selected_step"] = selected["step"]
        publish()

    started = time.time()
    try:
        inherited_result = root / "train_3200_result.json"
        while not inherited_result.exists():
            if time.time() >= deadline - 600:
                raise TimeoutError("Inherited worker exceeded finite allowance")
            time.sleep(2)
        train = read(inherited_result)
        if train.get("status") != "completed" or int(train.get("step", -1)) != 3200:
            raise RuntimeError("Inherited train_3200 did not complete exactly")
        endpoint, rows = metric_endpoint(training_out)
        if endpoint != 3200 or rows != 3200:
            raise RuntimeError("Scheduler continuity failed after inherited worker")
        ledger["events"].append(
            {
                "tag": "train_3200",
                "kind": "train",
                "pid": inherited_worker,
                "returncode": 0,
                "inherited_across_handoff": True,
                "step": endpoint,
            }
        )
        record["training"] = train
        checkpoint = train["checkpoint"]

        for target in TARGETS:
            if target != 3200:
                train = launch(f"train_{target}", "train", training_out, target=target)
                if train["step"] != target:
                    raise RuntimeError("Extended fixed exposure interrupted")
                endpoint, rows = metric_endpoint(training_out)
                if endpoint != target or rows != target:
                    raise RuntimeError("Scheduler continuity failed")
                checkpoint = train["checkpoint"]
                record["training"] = train
            validation = launch(
                f"validation_{target}",
                "eval",
                root / ARM / f"validation_{target}",
                checkpoint=checkpoint,
                split="val",
                eval_seed=SPATIAL_VAL_SEED,
                n=VALIDATION_N,
            )
            accept_validation(validation)

        record["terminal_checkpoint"] = checkpoint
        record["selected_test"] = launch(
            "selected_test",
            "eval",
            root / ARM / "selected_test",
            checkpoint=record["selected_checkpoint"],
            split="test",
            eval_seed=SPATIAL_TEST_SEED,
            n=TEST_N,
        )
        record["terminal_test"] = (
            record["selected_test"]
            if record["selected_step"] == TARGETS[-1]
            else launch(
                "terminal_test",
                "eval",
                root / ARM / "terminal_test",
                checkpoint=checkpoint,
                split="test",
                eval_seed=SPATIAL_TEST_SEED,
                n=TEST_N,
            )
        )
        record["status"] = aggregate["status"] = ledger["status"] = "completed"
    except BaseException as error:
        aggregate.update(status="failed", error=repr(error))
        ledger["status"] = "failed"
        raise
    finally:
        aggregate["extension_wall_seconds"] = time.time() - started
        publish()
        write(
            root / "exit.json",
            {
                "status": aggregate["status"],
                "error": aggregate.get("error"),
                "deadline_unix": deadline,
                "extension_wall_seconds": aggregate["extension_wall_seconds"],
            },
        )
        write(
            root / "artifact_index.json",
            {
                str(path.relative_to(root)).replace("\\", "/"): sha(path)
                for path in root.rglob("*")
                if path.is_file() and path.name != "artifact_index.json"
            },
        )


if __name__ == "__main__":
    main()
