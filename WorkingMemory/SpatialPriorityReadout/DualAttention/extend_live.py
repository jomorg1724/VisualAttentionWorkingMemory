"""Post-hoc, in-place extension of the live dual-attention scratch trajectory."""
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.SpatialPriorityReadout.DualAttention.model import ARM
from WorkingMemory.SpatialPriorityReadout.DualAttention.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    VALIDATION_N,
    assert_fixed,
    recipe,
)

TARGETS = tuple(range(3200, 12001, 800))


def verify_index(directory):
    directory = Path(directory)
    rows = [
        json.loads(line)
        for line in (directory / "checkpoint_index.jsonl").read_text().splitlines()
    ]
    files = {path.name for path in directory.glob("checkpoint_*.pt")}
    seen = set()
    for row in rows:
        path = directory / row.get("file", "")
        if (
            not re.fullmatch(r"checkpoint_[0-9]{6}\.pt", row.get("file", ""))
            or row["file"] in seen
            or not path.is_file()
            or int(path.stem.rsplit("_", 1)[1]) != int(row["step"])
            or sha(path) != row["sha256"]
        ):
            raise RuntimeError("Malformed checkpoint index")
        seen.add(row["file"])
    if files != seen:
        raise RuntimeError("Checkpoint/index inventory mismatch")
    return rows


def newest_checkpoint(directory, target):
    eligible = [
        row for row in verify_index(directory) if int(row["step"]) <= target
    ]
    row = max(eligible, key=lambda value: value["step"])
    return str(Path(directory) / row["file"])


def process_running(pid):
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except (OSError, FileNotFoundError):
        return False


def run(config_path):
    config = read(config_path)
    if (
        config["authorized_arms"] != [ARM]
        or config["old_target_step"] != 4000
        or config["target_step"] != 12000
        or tuple(config["validation_steps"]) != TARGETS
    ):
        raise RuntimeError("Unauthorized extension config")

    deadline = float(config["run_deadline_unix"])
    results = ROOT / "dual_attention_results"
    training_out = results / ARM / "training"
    ledger_path = results / "budget.json"
    aggregate_path = results / "aggregate.json"
    ledger = read(ledger_path)
    aggregate = read(aggregate_path)
    record = aggregate["run"][ARM]
    cfg = recipe()
    assert_fixed(cfg)
    if cfg["updates"] != config["old_target_step"]:
        raise RuntimeError("Original checkpoint config changed")
    if process_running(config["replaced_supervisor_pid"]):
        raise RuntimeError("Original supervisor still owns the run")
    active = ledger.get("active")
    if (
        not active
        or active["kind"] != "train"
        or active["tag"] != config["takeover_tag"]
        or active["pid"] != config["takeover_worker_pid"]
    ):
        raise RuntimeError("Live worker handoff mismatch")
    if [value["step"] for value in record["validation"]] != [800, 1600, 2400]:
        raise RuntimeError("Prior validation continuity mismatch")

    extension = dict(
        status="armed",
        authorization="post-hoc user-authorized extension",
        motivation="fair scratch-vs-trained comparison",
        old_target_step=4000,
        target_step=12000,
        validation_steps=list(TARGETS),
        inherited_validation_steps=[800, 1600, 2400],
        trajectory="in-place; model/optimizer/RNG/stream/counters preserved",
        takeover_tag=active["tag"],
        takeover_worker_pid=active["pid"],
        replaced_supervisor_pid=config["replaced_supervisor_pid"],
        armed_unix=time.time(),
        run_deadline_unix=deadline,
        cleanup_deadline_unix=config["cleanup_deadline_unix"],
        projection=config["projection"],
    )
    aggregate["posthoc_extension"] = extension
    aggregate["status"] = "training"
    record["status"] = "training"
    ledger["deadline_unix"] = deadline
    ledger["status"] = "training"
    if not any(event.get("kind") == "posthoc_extension" for event in ledger["events"]):
        ledger["events"].append(
            dict(
                kind="posthoc_extension",
                old_target_step=4000,
                target_step=12000,
                validation_steps=list(TARGETS),
                active_worker_preserved=active["pid"],
                unix=extension["armed_unix"],
            )
        )

    def publish():
        write(ledger_path, ledger)
        write(aggregate_path, aggregate)
        write(results / "extension_receipt.json", extension)

    def validate_training(target, result):
        if (
            result["status"] != "completed"
            or result["step"] != target
            or result["counts"]["episodes"] != target * 40
        ):
            raise RuntimeError(f"Training target {target} incomplete")
        rows = verify_index(training_out)
        if not any(
            row["step"] == target
            and str(training_out / row["file"]) == result["checkpoint"]
            for row in rows
        ):
            raise RuntimeError(f"Target checkpoint {target} missing from index")
        with (training_out / "metrics.csv").open(newline="") as handle:
            steps = [int(row["step"]) for row in csv.DictReader(handle)]
        if steps != list(range(1, target + 1)):
            raise RuntimeError(f"Metric continuity failed through {target}")

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            checkpoint = newest_checkpoint(out, extra["target"])
            checkpoint_step = int(Path(checkpoint).stem.rsplit("_", 1)[1])
            expected = extra["target"] - 800
            if checkpoint_step != expected:
                raise RuntimeError(
                    f"Expected checkpoint {expected}, found {checkpoint_step}"
                )
            extra["checkpoint"] = checkpoint
        job_path = results / f"{tag}_job.json"
        result_path = results / f"{tag}_result.json"
        log_path = results / f"{tag}.log"
        if any(path.exists() for path in (job_path, result_path, log_path)):
            raise RuntimeError("Refusing to overwrite existing job " + tag)
        job = dict(
            kind=kind,
            arm=ARM,
            config=cfg,
            out=str(out),
            result=str(result_path),
            deadline=deadline - 300,
            source_hashes=ledger["source_hashes"],
            **extra,
        )
        write(job_path, job)
        tick = time.time()
        with log_path.open("a") as handle:
            process = subprocess.Popen(
                [sys.executable, "-B", str(HERE / "worker.py"), str(job_path)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            ledger["active"] = dict(
                pid=process.pid, kind=kind, tag=tag, log=str(log_path)
            )
            publish()
            try:
                code = process.wait(timeout=max(0.01, deadline - time.time() - 300))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        ledger["events"].append(
            dict(
                tag=tag,
                kind=kind,
                pid=process.pid,
                returncode=code,
                seconds=time.time() - tick,
            )
        )
        ledger["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} exited {code}; see {log_path}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError("Bounded worker incomplete " + tag)
        return result

    started = time.time()
    publish()
    try:
        takeover_result = results / f"{active['tag']}_result.json"
        while process_running(active["pid"]):
            if time.time() >= deadline - 600:
                raise TimeoutError("Takeover worker exceeded extension allowance")
            time.sleep(5)
        if not takeover_result.is_file():
            raise RuntimeError("Takeover worker exited without result")
        train = read(takeover_result)
        validate_training(TARGETS[0], train)
        if not any(event.get("tag") == active["tag"] for event in ledger["events"]):
            ledger["events"].append(
                dict(
                    tag=active["tag"],
                    kind="train",
                    pid=active["pid"],
                    returncode=0,
                    seconds=train["worker_seconds"],
                    inherited_by_extension=True,
                )
            )
        ledger["active"] = None
        record["training"] = train
        publish()

        for target in TARGETS:
            if target != TARGETS[0]:
                train = launch(
                    f"train_{target}", "train", training_out, target=target
                )
                validate_training(target, train)
                record["training"] = train
            checkpoint = train["checkpoint"]
            validation = launch(
                f"validation_{target}",
                "eval",
                results / ARM / f"validation_{target}",
                checkpoint=checkpoint,
                split="val",
                eval_seed=SPATIAL_VAL_SEED,
                n=VALIDATION_N,
            )
            previous = {
                value["step"]: value
                for value in record["validation"]
                if value["step"] != target
            }
            previous[target] = validation
            record["validation"] = [previous[key] for key in sorted(previous)]
            selected = max(
                record["validation"],
                key=lambda value: (tuple(value["rank"]), -value["step"]),
            )
            record["selected_checkpoint"] = selected["checkpoint"]
            record["selected_step"] = selected["step"]
            publish()

        record["terminal_checkpoint"] = checkpoint
        record["selected_test"] = launch(
            "selected_test",
            "eval",
            results / ARM / "selected_test",
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
                results / ARM / "terminal_test",
                checkpoint=checkpoint,
                split="test",
                eval_seed=SPATIAL_TEST_SEED,
                n=TEST_N,
            )
        )
        extension["status"] = "completed"
        record["status"] = aggregate["status"] = ledger["status"] = "completed"
    except BaseException as error:
        extension.update(status="failed", error=repr(error))
        aggregate.update(status="failed", error=repr(error))
        ledger["status"] = "failed"
        raise
    finally:
        extension["extension_wall_seconds"] = time.time() - started
        aggregate["extension_wall_seconds"] = extension["extension_wall_seconds"]
        publish()
        write(
            results / "exit.json",
            dict(
                status=aggregate["status"],
                error=aggregate.get("error"),
                deadline_unix=deadline,
                extension_wall_seconds=extension["extension_wall_seconds"],
            ),
        )
        write(
            results / "artifact_index.json",
            {
                str(path.relative_to(results)).replace("\\", "/"): sha(path)
                for path in results.rglob("*")
                if path.is_file() and path.name != "artifact_index.json"
            },
        )


if __name__ == "__main__":
    run(sys.argv[1])
