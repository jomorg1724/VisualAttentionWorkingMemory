"""Resume the interrupted scratch arm from its last durable checkpoint."""
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.model import (
    ARM,
    VERSION,
    initialize_scratch,
)
from WorkingMemory.AttentionContextComparator.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    assert_fixed,
    recipe,
)

RESUME_VERSION = "attention_context_comparator_scratch_resume_v1"
RESUME_STEP = 1024
REMAINING_TARGETS = (1600, 2400, 3200, 4000)


def newest_checkpoint(directory, target):
    directory = Path(directory)
    index = directory / "checkpoint_index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    files = set(directory.glob("checkpoint_*.pt"))
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
    eligible = [row for row in rows if row["step"] <= target]
    return str(directory / max(eligible, key=lambda row: row["step"])["file"])


def same(left, right):
    import numpy as np
    import torch

    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right)
        )
    return left == right


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def verify_resume(manifest, checkpoint, cfg):
    import numpy as np
    import torch

    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

    expected_keys = {
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
    saved = torch.load(checkpoint, map_location="cpu")
    if set(saved) != expected_keys:
        raise RuntimeError("Unexpected checkpoint payload keys")
    if (
        saved["version"] != VERSION
        or saved["arm"] != ARM
        or saved["config"] != cfg
        or saved["source_hashes"] != manifest["source_hashes"]
        or saved["step"] != RESUME_STEP
        or saved["counts"] != {"episodes": 40960, "frames": 860120}
        or sum(saved["cell_counts"].values()) != 40960
        or set(saved["rng"]) != {"python", "numpy", "torch", "cuda"}
        or saved["stream"].get("seed") != cfg["train_seed"]
        or saved["stream"].get("split") != "train"
        or len(saved["model"]) != 138
        or len(saved["optimizer"].get("param_groups", [])) != 4
        or len(saved["optimizer"].get("state", {})) != 133
    ):
        raise RuntimeError("Checkpoint payload continuity mismatch")

    restored = []
    for _ in range(2):
        random.seed(cfg["model_seed"])
        np.random.seed(cfg["model_seed"])
        torch.manual_seed(cfg["model_seed"])
        model, optimizer, lineage = initialize_scratch(ARM, cfg)
        if lineage != saved["lineage"]:
            raise RuntimeError("Lineage reconstruction mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        stream = SpatialBatteryStream(cfg["train_seed"], "train")
        stream.load_state_dict(saved["stream"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])

        batches = []
        for task_index, task in enumerate(cfg["task_classes"]):
            names = cfg["train_names"][task]
            generator = np.random.default_rng(
                cfg["scheduler_seed"]
                + 100003 * task_index
                + RESUME_STEP // len(names)
            )
            chosen = names[
                int(
                    generator.permutation(len(names))[
                        RESUME_STEP % len(names)
                    ]
                )
            ]
            images, labels, metadata = stream.batch(
                cfg["batch_size"],
                cfg["cells"][chosen]["task"],
                cfg["cells"][chosen]["condition"],
            )
            batches.append(
                (
                    chosen,
                    hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                    hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                    hashlib.sha256(
                        json.dumps(metadata, sort_keys=True).encode()
                    ).hexdigest(),
                )
            )
        restored.append(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                batches=batches,
                stream=stream.state_dict(),
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
            )
        )
    if not same(restored[0], restored[1]):
        raise RuntimeError("Duplicate restore did not produce exact next batches")
    if not same(restored[0]["model"], saved["model"]):
        raise RuntimeError("Model state did not restore exactly")
    if not same(restored[0]["optimizer"], saved["optimizer"]):
        raise RuntimeError("Optimizer state did not restore exactly")
    return dict(
        status="passed",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        version=saved["version"],
        arm=saved["arm"],
        config_exact=True,
        step=saved["step"],
        model_tensors=len(saved["model"]),
        optimizer_groups=len(saved["optimizer"]["param_groups"]),
        optimizer_states=len(saved["optimizer"]["state"]),
        stream_keys=sorted(saved["stream"]),
        counts=saved["counts"],
        cell_counts=saved["cell_counts"],
        rng_keys=sorted(saved["rng"]),
        exact_model_restore=True,
        exact_optimizer_restore=True,
        deterministic_next_batch_across_duplicate_restore=True,
        next_batch_digest=digest(restored[0]["batches"]),
        next_cells=[row[0] for row in restored[0]["batches"]],
    )


def bootstrap(root, manifest):
    prior = ROOT / manifest["prior_retrieved_root"]
    artifact_index = read(prior / "artifact_index.json")
    if sha(prior / "artifact_index.json") != manifest["prior_artifact_index_sha256"]:
        raise RuntimeError("Prior artifact index hash mismatch")
    failures = [
        name
        for name, expected in artifact_index.items()
        if not (prior / name).is_file() or sha(prior / name) != expected
    ]
    if failures:
        raise RuntimeError("Prior retrieved artifact verification failed " + repr(failures))
    source = prior / ARM
    destination = root / ARM
    shutil.copytree(source, destination)
    metrics = destination / "training/metrics.csv"
    rows = list(csv.DictReader(metrics.open(newline="")))
    columns = rows[0].keys()
    durable = [row for row in rows if int(row["step"]) <= RESUME_STEP]
    discarded = [row for row in rows if int(row["step"]) > RESUME_STEP]
    if (
        len(durable) != RESUME_STEP
        or len(discarded) != 97
        or int(discarded[-1]["step"]) != 1121
    ):
        raise RuntimeError("Unexpected logged-only metrics inventory")
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(durable)
    for name in ("validation_800_job.json", "validation_800_result.json", "validation_800.log"):
        shutil.copy2(prior / name, root / name)
    return dict(
        prior_files_verified=len(artifact_index),
        prior_checkpoint_files=sum(name.endswith(".pt") for name in artifact_index),
        durable_metric_rows=len(durable),
        discarded_logged_only_updates=len(discarded),
        discarded_logged_only_episodes=len(discarded) * 40,
        logged_endpoint_step=int(rows[-1]["step"]),
        durable_step=RESUME_STEP,
    )


def run(manifest_path):
    import numpy
    import PIL
    import scipy
    import torch

    manifest = read(manifest_path)
    root = ROOT / "attention_context_comparator_results"
    if root.exists():
        raise RuntimeError("Resume result root must start absent")
    root.mkdir()
    deadline = float(manifest["deadline_unix"])
    for name, expected in manifest["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Pinned source changed " + name)
    if sha(HERE / "resume_sweep.py") != manifest["resume_sweep_sha256"]:
        raise RuntimeError("Resume launcher source changed")
    cfg = recipe()
    assert_fixed(cfg)
    prior = ROOT / manifest["prior_retrieved_root"]
    prior_aggregate = read(prior / "aggregate.json")
    bootstrap_receipt = bootstrap(root, manifest)
    training_out = root / ARM / "training"
    checkpoint = Path(newest_checkpoint(training_out, RESUME_STEP))
    if sha(checkpoint) != manifest["resume_checkpoint_sha256"]:
        raise RuntimeError("Resume checkpoint SHA mismatch")
    continuity = verify_resume(manifest, checkpoint, cfg)
    write(root / "resume_continuity.json", continuity)

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
        raise RuntimeError("Unpinned cloud runtime " + repr(versions))
    if "A100" in versions["gpu"].upper():
        raise RuntimeError("A100 is explicitly prohibited")

    started = time.time()
    old_record = prior_aggregate["run"][ARM]
    record = dict(
        config=cfg,
        validation=old_record["validation"],
        selected_checkpoint=old_record["selected_checkpoint"],
        selected_step=old_record["selected_step"],
        resume_checkpoint=str(checkpoint),
        resume_step=RESUME_STEP,
    )
    ledger = dict(
        started_unix=started,
        deadline_unix=deadline,
        status="continuity_verified",
        active=None,
        events=[
            dict(
                kind="resume_bootstrap",
                unix=started,
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),
            )
        ],
        source_hashes=manifest["source_hashes"],
    )
    aggregate = dict(
        status="continuity_verified",
        platform=versions,
        version=RESUME_VERSION,
        checkpoint_payload_version=VERSION,
        initialization_kind="exact_durable_resume",
        loaded_model=True,
        loaded_optimizer_state=True,
        loaded_stream_rng_counters=True,
        source_hashes=manifest["source_hashes"],
        authorized_arms=[ARM],
        fixed_exposure=dict(updates=UPDATES, episodes=TOTAL_EPISODES),
        resume_from=dict(
            step=RESUME_STEP,
            episodes=40960,
            checkpoint_sha256=manifest["resume_checkpoint_sha256"],
            prior_run=manifest["prior_run"],
            prior_retrieval=manifest["prior_retrieval"],
            bootstrap=bootstrap_receipt,
        ),
        validation_targets=[800, *REMAINING_TARGETS],
        selection="maximum [minimum chance-normalized task BA, mean task AUC], then earlier step",
        continuity=continuity,
        run={ARM: record},
        architecture=manifest["architecture"],
    )
    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"

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
            ledger["active"] = dict(pid=process.pid, kind=kind, tag=tag, log=str(log_path))
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
    try:
        if time.time() >= deadline - 600:
            raise TimeoutError("Insufficient finite cloud allowance")
        aggregate["status"] = ledger["status"] = "training"
        publish()
        for target in REMAINING_TARGETS:
            train = launch(
                f"train_{target}",
                "train",
                training_out,
                target=target,
            )
            if train["step"] != target:
                raise RuntimeError("Fixed exposure interrupted")
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
            by_step = {value["step"]: value for value in record["validation"]}
            by_step[target] = validation
            record["validation"] = [by_step[key] for key in sorted(by_step)]
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
            if record["selected_step"] == REMAINING_TARGETS[-1]
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
        aggregate["wall_seconds"] = time.time() - started
        publish()
        write(
            root / "exit.json",
            dict(
                status=aggregate["status"],
                error=aggregate.get("error"),
                deadline_unix=deadline,
                wall_seconds=aggregate["wall_seconds"],
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
