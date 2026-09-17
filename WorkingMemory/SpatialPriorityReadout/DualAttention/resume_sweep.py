"""Exact step-800 continuation for the interrupted scratch dual-attention arm."""
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.SpatialPriorityReadout.DualAttention.model import ARM, VERSION
from WorkingMemory.SpatialPriorityReadout.DualAttention.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    assert_fixed,
    recipe,
)

RESUME_STEP = 800
TARGETS = (1600, 2400, 3200, 4000)


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
    rows = verify_index(directory)
    eligible = [row for row in rows if int(row["step"]) <= target]
    return str(Path(directory) / max(eligible, key=lambda row: row["step"])["file"])


def next_batch_digest(saved):
    import numpy as np

    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

    cfg = saved["config"]
    step = int(saved["step"])
    stream = SpatialBatteryStream(cfg["train_seed"], "train")
    stream.load_state_dict(saved["stream"])
    digest = hashlib.sha256()
    schedule = []
    for task_index, task in enumerate(cfg["task_classes"]):
        names = cfg["train_names"][task]
        generator = np.random.default_rng(
            cfg["scheduler_seed"] + 100003 * task_index + step // len(names)
        )
        chosen = names[int(generator.permutation(len(names))[step % len(names)])]
        images, labels, metadata = stream.batch(
            cfg["batch_size"],
            cfg["cells"][chosen]["task"],
            cfg["cells"][chosen]["condition"],
        )
        digest.update(chosen.encode())
        digest.update(images.numpy().tobytes())
        digest.update(labels.numpy().tobytes())
        digest.update(
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        )
        schedule.append(chosen)
    return digest.hexdigest(), schedule


def run(manifest_path):
    import numpy
    import PIL
    import scipy
    import torch

    manifest = read(manifest_path)
    deadline = float(manifest["deadline_unix"])
    source = ROOT / "resume_input" / "dual_attention_results"
    root = ROOT / "dual_attention_results"
    if root.exists():
        raise RuntimeError("Refusing to reuse resume output")
    if manifest["authorized_arms"] != [ARM] or manifest["resume_step"] != RESUME_STEP:
        raise RuntimeError("Unauthorized resume manifest")
    if sha(source / "artifact_index.json") != manifest["artifact_index_sha256"]:
        raise RuntimeError("Resume artifact index hash mismatch")
    artifact_index = read(source / "artifact_index.json")
    if len(artifact_index) != manifest["artifact_count"]:
        raise RuntimeError("Resume artifact count mismatch")
    failures = [
        name
        for name, digest in artifact_index.items()
        if not (source / Path(name)).is_file()
        or sha(source / Path(name)) != digest
    ]
    if failures:
        raise RuntimeError("Resume input artifact mismatch " + repr(failures))
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned source changed " + name)

    training_source = source / ARM / "training"
    rows = verify_index(training_source)
    checkpoint = training_source / "checkpoint_000800.pt"
    if (
        sha(checkpoint) != manifest["checkpoint_sha256"]
        or not any(
            row["step"] == RESUME_STEP
            and row["file"] == checkpoint.name
            and row["sha256"] == manifest["checkpoint_sha256"]
            for row in rows
        )
    ):
        raise RuntimeError("Pinned resume checkpoint/index mismatch")
    saved = torch.load(checkpoint, map_location="cpu")
    required = {
        "version",
        "arm",
        "config",
        "source_hashes",
        "lineage",
        "step",
        "model",
        "optimizer",
        "stream",
        "counts",
        "cell_counts",
        "rng",
    }
    cfg = recipe()
    assert_fixed(cfg)
    if (
        set(saved) != required
        or saved["version"] != VERSION
        or saved["arm"] != ARM
        or saved["config"] != cfg
        or saved["source_hashes"] != manifest["source_hashes"]
        or saved["step"] != RESUME_STEP
        or saved["counts"]["episodes"] != 32000
        or sum(saved["cell_counts"].values()) != 32000
        or set(saved["rng"]) != {"python", "numpy", "torch", "cuda"}
        or len(saved["optimizer"]["state"]) != manifest["optimizer_state_count"]
    ):
        raise RuntimeError("Resume checkpoint payload mismatch")
    first_digest, first_schedule = next_batch_digest(saved)
    second_digest, second_schedule = next_batch_digest(saved)
    if (
        first_digest != manifest["next_batch_sha256"]
        or second_digest != first_digest
        or second_schedule != first_schedule
        or first_schedule != manifest["next_batch_cells"]
    ):
        raise RuntimeError("Step-801 deterministic batch continuity mismatch")
    del saved

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
    if (
        versions["torch"],
        versions["numpy"],
        versions["scipy"],
        versions["pillow"],
    ) != (
        expected["torch"],
        expected["numpy"],
        expected["scipy"],
        expected["pillow"],
    ):
        raise RuntimeError("Unpinned cloud runtime " + repr(versions))
    if "A100" in versions["gpu"].upper():
        raise RuntimeError("A100 is explicitly prohibited")

    shutil.copytree(source, root)
    for name in (
        "artifact_index.json",
        "budget.json",
        "exit.json",
        "train_1600.log",
        "train_1600_job.json",
        "train_1600_result.json",
    ):
        (root / name).unlink(missing_ok=True)
    metrics = root / ARM / "training" / "metrics.csv"
    with metrics.open(newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
        columns = metric_rows[0].keys()
    if (
        len(metric_rows) != manifest["logged_rows"]
        or int(metric_rows[-1]["step"]) != manifest["logged_endpoint_step"]
    ):
        raise RuntimeError("Unexpected interrupted metric endpoint")
    durable = [row for row in metric_rows if int(row["step"]) <= RESUME_STEP]
    if len(durable) != RESUME_STEP or [int(row["step"]) for row in durable] != list(
        range(1, RESUME_STEP + 1)
    ):
        raise RuntimeError("Durable metric prefix is not continuous")
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(durable)

    old_aggregate = read(source / "aggregate.json")
    record = old_aggregate["run"][ARM]
    if [row["step"] for row in record["validation"]] != [RESUME_STEP]:
        raise RuntimeError("Validation-800 is missing or ambiguous")
    validation_800_hashes = {
        name: digest
        for name, digest in artifact_index.items()
        if f"{ARM}/validation_800/" in name
    }
    aggregate = old_aggregate
    aggregate.pop("error", None)
    aggregate.pop("wall_seconds", None)
    aggregate.update(
        status="training",
        platform=versions,
        resume=dict(
            source_checkpoint=str(checkpoint.relative_to(source)).replace("\\", "/"),
            source_checkpoint_sha256=manifest["checkpoint_sha256"],
            durable_step=RESUME_STEP,
            durable_episodes=32000,
            discarded_logged_only_updates=manifest["logged_rows"] - RESUME_STEP,
            discarded_logged_only_episodes=(
                manifest["logged_endpoint_episodes"] - 32000
            ),
            first_new_step=801,
            next_batch_sha256=first_digest,
            next_batch_cells=first_schedule,
            validation_800_preserved=validation_800_hashes,
            resumed_unix=time.time(),
        ),
    )
    record["status"] = "training"
    started = time.time()
    ledger = dict(
        started_unix=started,
        deadline_unix=deadline,
        status="training",
        active=None,
        events=[
            dict(
                kind="resume",
                step=RESUME_STEP,
                episodes=32000,
                checkpoint_sha256=manifest["checkpoint_sha256"],
                unix=started,
            )
        ],
        source_hashes=manifest["source_hashes"],
    )

    def publish():
        write(root / "budget.json", ledger)
        write(root / "aggregate.json", aggregate)

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            extra["checkpoint"] = newest_checkpoint(out, extra["target"])
        job_path = root / f"{tag}_job.json"
        result_path = root / f"{tag}_result.json"
        log_path = root / f"{tag}.log"
        job = dict(
            kind=kind,
            arm=ARM,
            config=cfg,
            out=str(out),
            result=str(result_path),
            deadline=deadline - 300,
            source_hashes=manifest["source_hashes"],
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

    publish()
    checkpoint = str(root / ARM / "training" / "checkpoint_000800.pt")
    try:
        if time.time() >= deadline - 600:
            raise TimeoutError("Insufficient finite cloud allowance")
        training_out = root / ARM / "training"
        for target in TARGETS:
            train = launch(f"train_{target}", "train", training_out, target=target)
            if train["step"] != target:
                raise RuntimeError("Fixed exposure interrupted")
            checkpoint = train["checkpoint"]
            record["training"] = train
            if target == TARGETS[0]:
                with metrics.open(newline="") as handle:
                    resumed_rows = [
                        row
                        for row in csv.DictReader(handle)
                        if int(row["step"]) > RESUME_STEP
                    ]
                if not resumed_rows or int(resumed_rows[0]["step"]) != 801:
                    raise RuntimeError("First resumed metric is not step 801")
            validation = launch(
                f"validation_{target}",
                "eval",
                root / ARM / f"validation_{target}",
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
        aggregate["resume_wall_seconds"] = time.time() - started
        publish()
        write(
            root / "exit.json",
            dict(
                status=aggregate["status"],
                error=aggregate.get("error"),
                deadline_unix=deadline,
                resume_wall_seconds=aggregate["resume_wall_seconds"],
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
