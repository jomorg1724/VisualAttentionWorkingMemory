"""Resume and extend the scratch spatial-priority run from its durable step-1600 state."""
import json
import os
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
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    assert_fixed,
    recipe,
)

RESUME_STEP = 1600
ORIGINAL_TARGET = 4000
FINAL_TARGET = 12000
REMAINING_TARGETS = tuple(range(2400, FINAL_TARGET + 1, 800))


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
        raise RuntimeError("No eligible resume checkpoint")
    return str(directory / max(eligible, key=lambda row: row["step"])["file"])


def run(manifest_path):
    import numpy
    import PIL
    import scipy
    import torch

    manifest = read(manifest_path)
    deadline = float(manifest["deadline_unix"])
    extension = manifest["posthoc_protocol_extension"]
    if (
        extension["original_target_updates"] != ORIGINAL_TARGET
        or extension["extended_target_updates"] != FINAL_TARGET
        or extension["validation_targets"] != list(REMAINING_TARGETS)
        or extension["preserve_trajectory"] is not True
    ):
        raise RuntimeError("Invalid post-hoc extension authorization")
    root = ROOT / "spatial_priority_results"
    root.mkdir(exist_ok=True)
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned source changed " + name)
    if sha(HERE / "resume_sweep.py") != manifest["resume_controller_sha256"]:
        raise RuntimeError("Resume controller changed")
    source_checkpoint = ROOT / manifest["resume_checkpoint"]
    if sha(source_checkpoint) != manifest["resume_checkpoint_sha256"]:
        raise RuntimeError("Resume checkpoint hash mismatch")
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
        or payload["step"] != RESUME_STEP
        or payload["counts"]["episodes"] != 64000
    ):
        raise RuntimeError("Resume checkpoint payload mismatch")
    versions = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        pillow=PIL.__version__,
        platform=platform.platform(),
        gpu=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True
        ).strip(),
    )
    expected = manifest["expected_runtime"]
    if (versions["torch"], versions["numpy"], versions["scipy"], versions["pillow"]) != (
        expected["torch"], expected["numpy"], expected["scipy"], expected["pillow"]
    ):
        raise RuntimeError("Unpinned cloud runtime " + repr(versions))
    if "A100" in versions["gpu"].upper():
        raise RuntimeError("A100 is explicitly prohibited")

    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"
    training_out = root / ARM / "training"
    started = time.time()
    if not budget_path.exists():
        if list(root.iterdir()):
            raise RuntimeError("Unexpected resume result files")
        training_out.mkdir(parents=True)
        staged = training_out / source_checkpoint.name
        shutil.copy2(source_checkpoint, staged)
        (training_out / "checkpoint_index.jsonl").write_text(
            json.dumps(dict(file=staged.name, step=RESUME_STEP, sha256=sha(staged))) + "\n"
        )
        source_aggregate = read(ROOT / manifest["source_aggregate"])
        record = source_aggregate["run"][ARM]
        if [row["step"] for row in record["validation"]] != [800, 1600]:
            raise RuntimeError("Historical validation state mismatch")
        aggregate = dict(source_aggregate)
        aggregate.pop("error", None)
        aggregate.pop("wall_seconds", None)
        aggregate.update(
            status="training",
            platform=versions,
            deadline_unix=deadline,
            posthoc_protocol_extension=extension,
            resume=dict(
                source_run=manifest["source_run"],
                checkpoint_sha256=manifest["resume_checkpoint_sha256"],
                durable_step=RESUME_STEP,
                durable_episodes=64000,
                discarded_updates=13,
                discarded_episodes=520,
                remaining_validation_targets=list(REMAINING_TARGETS),
            ),
        )
        record["status"] = "training"
        record["resume_training"] = []
        ledger = dict(
            started_unix=started,
            deadline_unix=deadline,
            status="training",
            active=None,
            events=[dict(kind="resume_from_durable_checkpoint", step=RESUME_STEP, unix=started)],
            source_hashes=manifest["source_hashes"],
        )
    else:
        ledger = read(budget_path)
        aggregate = read(aggregate_path)
        record = aggregate["run"][ARM]
        if ledger["status"] != "training" or aggregate["status"] != "training":
            raise RuntimeError("Cannot extend a terminal run")
        previous_deadline = ledger["deadline_unix"]
        ledger["deadline_unix"] = deadline
        ledger["active"] = None
        ledger["events"].append(
            dict(
                kind="user_authorized_posthoc_protocol_extension",
                unix=time.time(),
                original_target_updates=ORIGINAL_TARGET,
                extended_target_updates=FINAL_TARGET,
                previous_deadline_unix=previous_deadline,
                deadline_unix=deadline,
                reason=extension["reason"],
            )
        )
        aggregate["deadline_unix"] = deadline
        aggregate["posthoc_protocol_extension"] = extension
        aggregate.pop("error", None)
        aggregate["status"] = record["status"] = ledger["status"] = "training"
        aggregate["resume"]["remaining_validation_targets"] = list(REMAINING_TARGETS)

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
            kind=kind, arm=ARM, config=cfg, out=str(out), result=str(result_path),
            deadline=deadline - 300, source_hashes=manifest["source_hashes"], **extra
        )
        if result_path.exists():
            cached = read(result_path)
            if cached.get("status") == "completed":
                prior_job = read(job_path)
                prior_deadline = prior_job.pop("deadline")
                current_deadline = job.pop("deadline")
                equivalent = prior_job == job
                prior_job["deadline"] = prior_deadline
                job["deadline"] = current_deadline
                if not equivalent:
                    raise RuntimeError("Cached result job mismatch")
                return cached
        write(job_path, job)
        tick = time.time()
        with log_path.open("a") as handle:
            process = subprocess.Popen(
                [sys.executable, "-B", str(HERE / "worker.py"), str(job_path)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
            )
            ledger["active"] = dict(pid=process.pid, kind=kind, tag=tag, log=str(log_path))
            publish()
            try:
                code = process.wait(timeout=max(0.01, deadline - time.time() - 300))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        ledger["events"].append(dict(
            tag=tag, kind=kind, pid=process.pid, returncode=code,
            seconds=time.time() - tick
        ))
        ledger["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} exited {code}; see {log_path}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError("Bounded worker incomplete " + tag)
        return result

    publish()
    try:
        if time.time() >= deadline - 600:
            raise TimeoutError("Insufficient finite cloud allowance")
        for target in REMAINING_TARGETS:
            train = launch(f"train_{target}", "train", training_out, target=target)
            if train["step"] != target:
                raise RuntimeError("Fixed exposure interrupted")
            if not record["resume_training"] or record["resume_training"][-1]["step"] != target:
                record["resume_training"].append(train)
            record["training"] = train
            validation = launch(
                f"validation_{target}", "eval", root / ARM / f"validation_{target}",
                checkpoint=train["checkpoint"], split="val",
                eval_seed=SPATIAL_VAL_SEED, n=VALIDATION_N
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
            "selected_test", "eval", root / ARM / "selected_test",
            checkpoint=record["selected_checkpoint"], split="test",
            eval_seed=SPATIAL_TEST_SEED, n=TEST_N
        )
        record["terminal_test"] = (
            record["selected_test"] if record["selected_step"] == FINAL_TARGET else
            launch(
                "terminal_test", "eval", root / ARM / "terminal_test",
                checkpoint=record["terminal_checkpoint"], split="test",
                eval_seed=SPATIAL_TEST_SEED, n=TEST_N
            )
        )
        record["status"] = aggregate["status"] = ledger["status"] = "completed"
    except BaseException as error:
        aggregate.update(status="failed", error=repr(error))
        ledger["status"] = "failed"
        raise
    finally:
        aggregate["resume_wall_seconds"] = time.time() - started
        publish()
        write(root / "exit.json", dict(
            status=aggregate["status"], error=aggregate.get("error"),
            deadline_unix=deadline, resume_wall_seconds=aggregate["resume_wall_seconds"]
        ))
        write(root / "artifact_index.json", {
            str(path.relative_to(root)).replace("\\", "/"): sha(path)
            for path in root.rglob("*")
            if path.is_file() and path.name != "artifact_index.json"
        })


if __name__ == "__main__":
    run(sys.argv[1])
