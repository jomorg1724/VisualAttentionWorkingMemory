"""Recover the extended control trajectory from its exact durable step-3200 state."""
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.SpatialPriorityReadout.model import ARM, VERSION
from WorkingMemory.SpatialPriorityReadout.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    VALIDATION_N,
    assert_fixed,
    recipe,
)

RECOVERY_STEP = 3200
FINAL_TARGET = 12000
VALIDATION_TARGETS = tuple(range(3200, FINAL_TARGET + 1, 800))
TRAINING_TARGETS = tuple(range(4000, FINAL_TARGET + 1, 800))


def indexed_latest(directory, target):
    directory = Path(directory)
    rows = [
        json.loads(line)
        for line in (directory / "checkpoint_index.jsonl").read_text().splitlines()
    ]
    files = set(directory.glob("checkpoint_*.pt"))
    if len(rows) != len(files):
        raise RuntimeError("Checkpoint/index inventory mismatch")
    seen = set()
    for row in rows:
        path = directory / row["file"]
        if (
            not re.fullmatch(r"checkpoint_[0-9]{6}\.pt", row["file"])
            or row["file"] in seen
            or not path.is_file()
            or sha(path) != row["sha256"]
            or int(path.stem.rsplit("_", 1)[1]) != int(row["step"])
        ):
            raise RuntimeError("Malformed checkpoint index")
        seen.add(row["file"])
    eligible = [row for row in rows if int(row["step"]) <= target]
    if not eligible:
        raise RuntimeError("No eligible recovery checkpoint")
    return str(directory / max(eligible, key=lambda row: row["step"])["file"])


def run(manifest_path):
    import numpy
    import PIL
    import scipy
    import torch

    manifest = read(manifest_path)
    deadline = float(manifest["deadline_unix"])
    recovery = manifest["recovery"]
    if (
        recovery["checkpoint_step"] != RECOVERY_STEP
        or recovery["target_updates"] != FINAL_TARGET
        or recovery["validation_targets"] != list(VALIDATION_TARGETS)
        or recovery["reuse_cached_jobs"] is not False
    ):
        raise RuntimeError("Invalid recovery authorization")
    root = ROOT / "spatial_priority_recovery_results"
    root.mkdir(exist_ok=True)
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned source changed " + name)
    if sha(HERE / "recovery_sweep.py") != manifest["recovery_controller_sha256"]:
        raise RuntimeError("Recovery controller changed")

    source_checkpoint = ROOT / manifest["recovery_checkpoint"]
    if sha(source_checkpoint) != manifest["recovery_checkpoint_sha256"]:
        raise RuntimeError("Recovery checkpoint hash mismatch")
    payload = torch.load(source_checkpoint, map_location="cpu")
    cfg = recipe()
    assert_fixed(cfg)
    required = {
        "version", "arm", "config", "source_hashes", "lineage", "step", "model",
        "optimizer", "stream", "counts", "cell_counts", "rng",
    }
    if (
        set(payload) != required
        or payload["version"] != VERSION
        or payload["arm"] != ARM
        or payload["config"] != cfg
        or payload["source_hashes"] != manifest["source_hashes"]
        or payload["step"] != RECOVERY_STEP
        or payload["counts"]["episodes"] != 128000
    ):
        raise RuntimeError("Recovery checkpoint payload mismatch")

    source_metrics = ROOT / manifest["recovery_metrics"]
    metric_rows = list(csv.DictReader(source_metrics.open()))
    metric_steps = [int(row["step"]) for row in metric_rows]
    if (
        metric_steps != list(range(1601, RECOVERY_STEP + 1))
        or int(metric_rows[-1]["episodes"]) != 128000
    ):
        raise RuntimeError("Recovery metric sequence is not exact")

    versions = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        pillow=PIL.__version__,
        platform=platform.platform(),
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    )
    expected = manifest["expected_runtime"]
    if (versions["torch"], versions["numpy"], versions["scipy"], versions["pillow"]) != (
        expected["torch"],
        expected["numpy"],
        expected["scipy"],
        expected["pillow"],
    ):
        raise RuntimeError("Unpinned recovery runtime " + repr(versions))
    if "A100" in versions["gpu"].upper():
        raise RuntimeError("A100 is explicitly prohibited")

    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"
    training_out = root / ARM / "training"
    started = time.time()
    if not budget_path.exists():
        if list(root.iterdir()):
            raise RuntimeError("Unexpected files in versioned recovery result root")
        training_out.mkdir(parents=True)
        source_inventory = read(ROOT / manifest["recovery_checkpoint_inventory"])
        index_rows = []
        for item in source_inventory:
            source = ROOT / item["source"]
            if sha(source) != item["sha256"]:
                raise RuntimeError("Recovery inventory hash mismatch " + item["source"])
            target = training_out / item["file"]
            shutil.copy2(source, target)
            index_rows.append(
                dict(file=target.name, step=item["step"], sha256=item["sha256"])
            )
        (training_out / "checkpoint_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in index_rows)
        )
        shutil.copy2(source_metrics, training_out / "metrics.csv")
        shutil.copy2(ROOT / manifest["recovery_initialization"], training_out / "initialization.json")

        aggregate = read(ROOT / manifest["source_aggregate"])
        record = aggregate["run"][ARM]
        if [row["step"] for row in record["validation"]] != [800, 1600, 2400]:
            raise RuntimeError("Recovered validation history mismatch")
        if any(row["step"] == RECOVERY_STEP for row in record["validation"]):
            raise RuntimeError("Validation 3200 was already recorded")
        checkpoint_by_step = {
            row["step"]: str(training_out / row["file"]) for row in index_rows
        }
        for row in record["validation"]:
            row["checkpoint"] = checkpoint_by_step[row["step"]]
        restored_training = read(ROOT / manifest["recovery_train_result"])
        restored_training["checkpoint"] = checkpoint_by_step[RECOVERY_STEP]
        record["training"] = restored_training
        if not any(row["step"] == RECOVERY_STEP for row in record["resume_training"]):
            record["resume_training"].append(restored_training)
        record["status"] = "training"
        aggregate.pop("error", None)
        aggregate.pop("resume_wall_seconds", None)
        aggregate.update(
            status="training",
            platform=versions,
            deadline_unix=deadline,
            recovery=recovery,
        )
        ledger = dict(
            started_unix=started,
            deadline_unix=deadline,
            status="training",
            active=None,
            events=[
                dict(
                    kind="recover_exact_durable_checkpoint",
                    step=RECOVERY_STEP,
                    episodes=128000,
                    checkpoint_sha256=manifest["recovery_checkpoint_sha256"],
                    prior_failure="Cached result job mismatch",
                    unix=started,
                )
            ],
            source_hashes=manifest["source_hashes"],
        )
    else:
        ledger = read(budget_path)
        aggregate = read(aggregate_path)
        record = aggregate["run"][ARM]
        if ledger["status"] != "training" or aggregate["status"] != "training":
            raise RuntimeError("Cannot restart terminal recovery")
        ledger["active"] = None
        ledger["events"].append(dict(kind="recovery_supervisor_restart", unix=time.time()))

    def publish():
        write(budget_path, ledger)
        write(aggregate_path, aggregate)

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            extra["checkpoint"] = indexed_latest(out, extra["target"])
        job_path = root / f"{tag}_job.json"
        result_path = root / f"{tag}_result.json"
        log_path = root / f"{tag}.log"
        job = dict(
            kind=kind,
            arm=ARM,
            config=cfg,
            out=str(out),
            result=str(result_path),
            deadline=deadline - 600,
            source_hashes=manifest["source_hashes"],
            **extra,
        )
        if result_path.exists():
            result = read(result_path)
            if result.get("status") == "completed" and read(job_path) == job:
                return result
            raise RuntimeError("Versioned recovery result/job mismatch")
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
                code = process.wait(timeout=max(0.01, deadline - time.time() - 600))
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
            raise RuntimeError("Bounded recovery worker incomplete " + tag)
        return result

    publish()
    try:
        if time.time() >= deadline - 1200:
            raise TimeoutError("Insufficient finite recovery allowance")
        validation = launch(
            "validation_3200",
            "eval",
            root / ARM / "validation_3200",
            checkpoint=indexed_latest(training_out, RECOVERY_STEP),
            split="val",
            eval_seed=SPATIAL_VAL_SEED,
            n=VALIDATION_N,
        )
        previous = {value["step"]: value for value in record["validation"]}
        previous[RECOVERY_STEP] = validation
        record["validation"] = [previous[key] for key in sorted(previous)]
        selected = max(
            record["validation"],
            key=lambda value: (tuple(value["rank"]), -value["step"]),
        )
        record["selected_checkpoint"] = selected["checkpoint"]
        record["selected_step"] = selected["step"]
        publish()

        for target in TRAINING_TARGETS:
            train = launch(f"train_{target}", "train", training_out, target=target)
            if train["step"] != target:
                raise RuntimeError("Fixed recovery exposure interrupted")
            if not record["resume_training"] or record["resume_training"][-1]["step"] != target:
                record["resume_training"].append(train)
            record["training"] = train
            validation = launch(
                f"validation_{target}",
                "eval",
                root / ARM / f"validation_{target}",
                checkpoint=train["checkpoint"],
                split="val",
                eval_seed=SPATIAL_VAL_SEED,
                n=VALIDATION_N,
            )
            previous = {value["step"]: value for value in record["validation"]}
            previous[target] = validation
            record["validation"] = [previous[key] for key in sorted(previous)]
            selected = max(
                record["validation"],
                key=lambda value: (tuple(value["rank"]), -value["step"]),
            )
            record["selected_checkpoint"] = selected["checkpoint"]
            record["selected_step"] = selected["step"]
            publish()

        record["terminal_checkpoint"] = record["training"]["checkpoint"]
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
            if record["selected_step"] == FINAL_TARGET
            else launch(
                "terminal_test",
                "eval",
                root / ARM / "terminal_test",
                checkpoint=record["terminal_checkpoint"],
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
        aggregate["recovery_wall_seconds"] = time.time() - started
        publish()
        write(
            root / "exit.json",
            dict(
                status=aggregate["status"],
                error=aggregate.get("error"),
                deadline_unix=deadline,
                recovery_wall_seconds=aggregate["recovery_wall_seconds"],
            ),
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
    run(sys.argv[1])
