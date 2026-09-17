"""Cloud supervisor for the scratch attention-context comparator arm."""
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.model import ARM, VERSION
from WorkingMemory.AttentionContextComparator.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    VALIDATION_TARGETS,
    assert_fixed,
    recipe,
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
    if {path.name for path in files} != seen:
        raise RuntimeError("Checkpoint/index inventory mismatch")
    eligible = [row for row in rows if row["step"] <= target]
    return (
        str(directory / max(eligible, key=lambda row: row["step"])["file"])
        if eligible
        else None
    )


def run(manifest_path):
    import numpy
    import PIL
    import scipy
    import torch

    manifest = read(manifest_path)
    root = ROOT / "attention_context_comparator_results"
    root.mkdir(exist_ok=True)
    deadline = float(manifest["deadline_unix"])
    if (
        manifest.get("initialization_kind") != "full_model_from_scratch"
        or "parent_sha256" in manifest
        or (ROOT / "portable/parent_checkpoint_008400.pt").exists()
    ):
        raise RuntimeError("Scratch run must not contain a parent checkpoint")
    fixture = (
        ROOT
        / "WorkingMemory/AttentionContextComparator/fixtures/"
        "control_checkpoint_000000.pt"
    )
    if (
        manifest.get("control_initialization_fixture_sha256")
        != "97b7e6ce513592fc094724854ea1d4d329c93fd1ae0d86aa20ed50e4fd653f76"
        or sha(fixture) != manifest["control_initialization_fixture_sha256"]
    ):
        raise RuntimeError("Pinned control checkpoint-0 fixture mismatch")
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned source changed " + name)
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
    cfg = recipe()
    assert_fixed(cfg)
    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"
    if budget_path.exists() != aggregate_path.exists():
        raise RuntimeError("Incomplete existing ledger")
    if budget_path.exists():
        ledger = read(budget_path)
        aggregate = read(aggregate_path)
        if (
            ledger["deadline_unix"] != deadline
            or ledger["source_hashes"] != manifest["source_hashes"]
            or aggregate["version"] != VERSION
            or aggregate["initialization_kind"] != "full_model_from_scratch"
            or aggregate["fixed_exposure"]
            != dict(updates=UPDATES, episodes=TOTAL_EPISODES)
            or aggregate["authorized_arms"] != [ARM]
        ):
            raise RuntimeError("Existing ledger differs from this run")
        if aggregate["status"] == "completed":
            return
        ledger["active"] = None
        ledger["events"].append(dict(kind="resume", unix=time.time()))
        record = aggregate["run"][ARM]
        started = ledger["started_unix"]
    else:
        if list(root.iterdir()):
            raise RuntimeError("Unexpected result files without ledger")
        started = time.time()
        ledger = dict(
            started_unix=started,
            deadline_unix=deadline,
            status="construction",
            active=None,
            events=[],
            source_hashes=manifest["source_hashes"],
        )
        record = dict(config=cfg, validation=[])
        aggregate = dict(
            status="construction",
            platform=versions,
            version=VERSION,
            initialization_kind="full_model_from_scratch",
            loaded_parent=False,
            loaded_optimizer_state=False,
            source_hashes=manifest["source_hashes"],
            authorized_arms=[ARM],
            fixed_exposure=dict(updates=UPDATES, episodes=TOTAL_EPISODES),
            validation_targets=list(VALIDATION_TARGETS),
            selection="maximum [minimum chance-normalized task BA, mean task AUC], then earlier step",
            run={ARM: record},
            architecture=manifest["architecture"],
        )

    def publish():
        write(budget_path, ledger)
        write(aggregate_path, aggregate)

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            extra["checkpoint"] = newest_checkpoint(out, extra["target"])
            if extra["checkpoint"] is None:
                extra.pop("checkpoint")
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
        if result_path.exists():
            cached = read(result_path)
            if cached.get("status") == "completed":
                if not job_path.exists() or read(job_path) != job:
                    raise RuntimeError("Cached result job mismatch")
                return cached
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
                code = process.wait(
                    timeout=max(0.01, deadline - time.time() - 300)
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
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
        construction = subprocess.run(
            [sys.executable, "-B", str(HERE / "check_model.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **__import__("os").environ,
                "REQUIRE_CONTROL_CHECKPOINT_EQUALITY": "1",
            },
        )
        (root / "construction_stdout.log").write_text(construction.stdout)
        (root / "construction_stderr.log").write_text(construction.stderr)
        if construction.returncode:
            raise RuntimeError("Remote construction checks failed")
        aggregate["construction_checks"] = read(HERE / "construction_checks.json")
        aggregate["status"] = ledger["status"] = "profiling"
        publish()
        profile = launch(
            "profile",
            "profile",
            root / "profile",
            profile_updates=3,
        )
        aggregate["profile"] = profile
        projected_training = 1.25 * UPDATES * profile["mean_update_seconds"]
        reserve = 2700
        aggregate["profile_projection"] = dict(
            measured_updates=3,
            mean_update_seconds=profile["mean_update_seconds"],
            projected_training_seconds=projected_training,
            evaluation_retrieval_reserve_seconds=reserve,
            remaining_seconds=deadline - time.time(),
            fixed_exposure=True,
        )
        publish()
        if projected_training + reserve > deadline - time.time():
            raise RuntimeError("Fixed 4000 updates do not fit finite deadline")
        aggregate["status"] = ledger["status"] = "training"
        publish()
        checkpoint = None
        training_out = root / ARM / "training"
        for target in VALIDATION_TARGETS:
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
            if record["selected_step"] == VALIDATION_TARGETS[-1]
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
