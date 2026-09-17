"""Cloud-side fixed-exposure supervisor; it provisions nothing by itself."""
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
from WorkingMemory.ProspectiveQuery.model import ARM, PARENT_SHA256
from WorkingMemory.ProspectiveQuery.protocol import (
    MATCHED_CONTROL_VALIDATION_TARGETS,
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    TOTAL_EPISODES,
    UPDATES,
    VALIDATION_N,
    VALIDATION_TARGETS,
    assert_same_as_biased_training,
    comparison_status,
    recipe,
)
from WorkingMemory.SpatialTaskBattery.stimuli import frame_count


def newest_indexed_checkpoint(directory, target):
    directory = Path(directory)
    if not directory.exists():
        return None
    index_path = directory / "checkpoint_index.jsonl"
    files = set(directory.glob("checkpoint_*.pt"))
    if not index_path.exists():
        if files:
            raise RuntimeError("Checkpoint files exist without a hash index")
        return None
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    indexed = set()
    steps = set()
    for row in rows:
        path = directory / row.get("file", "")
        if (
            not re.fullmatch(r"checkpoint_[0-9]{6}\.pt", row.get("file", ""))
            or row.get("step") in steps
            or row.get("file") in indexed
            or not path.is_file()
            or sha(path) != row.get("sha256")
            or int(path.stem.rsplit("_", 1)[1]) != int(row["step"])
        ):
            raise RuntimeError("Malformed, duplicate, missing, or corrupt checkpoint")
        indexed.add(row["file"])
        steps.add(row["step"])
    unexpected = {path.name for path in files} - indexed
    if unexpected:
        raise RuntimeError(
            "Unexpected unindexed checkpoint(s): " + ", ".join(sorted(unexpected))
        )
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
    root = ROOT / "prospective_query_results"
    root.mkdir(exist_ok=True)
    deadline = manifest["deadline_unix"]
    parent = ROOT / "portable/parent_checkpoint_008400.pt"
    if manifest["parent_sha256"] != PARENT_SHA256 or sha(parent) != PARENT_SHA256:
        raise RuntimeError("Not the exact original attention8400 parent")
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Source/package changed " + name)

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
    if (
        versions["torch"],
        versions["numpy"],
        versions["scipy"],
        versions["pillow"],
    ) != ("1.13.1+cu117", "1.23.1", "1.8.1", "9.1.1"):
        raise RuntimeError("Unpinned cloud runtime " + repr(versions))
    cfg = recipe()
    assert_same_as_biased_training(cfg)
    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"
    if budget_path.exists() != aggregate_path.exists():
        raise RuntimeError("Incomplete existing run ledger; refusing overwrite")
    if budget_path.exists():
        ledger = read(budget_path)
        aggregate = read(aggregate_path)
        if (
            ledger.get("deadline_unix") != deadline
            or ledger.get("source_hashes") != manifest["source_hashes"]
            or aggregate.get("parent_sha256") != PARENT_SHA256
            or aggregate.get("fixed_exposure")
            != dict(updates=UPDATES, episodes=TOTAL_EPISODES)
            or aggregate.get("authorized_arms") != [ARM]
        ):
            raise RuntimeError("Existing ledger is not this exact bounded run")
        if aggregate.get("status") == "completed":
            return aggregate
        started = ledger["started_unix"]
        record = aggregate["run"][ARM]
        if record.get("config") != cfg:
            raise RuntimeError("Existing run config differs from pinned recipe")
        ledger["active"] = None
        ledger["events"].append(
            dict(kind="supervisor_resume", resumed_unix=time.time())
        )
    else:
        unexpected = [path.name for path in root.iterdir()]
        if unexpected:
            raise RuntimeError(
                "Unexpected existing result files without ledger: "
                + ", ".join(sorted(unexpected))
            )
        started = time.time()
        ledger = dict(
            started_unix=started,
            deadline_unix=deadline,
            status="profiling",
            active=None,
            events=[],
            source_hashes=manifest["source_hashes"],
        )
        record = dict(config=cfg, validation=[])
        aggregate = dict(
            status="profiling",
            platform=versions,
            parent=str(parent),
            parent_sha256=PARENT_SHA256,
            run={ARM: record},
            authorized_arms=[ARM],
            fixed_exposure=dict(updates=UPDATES, episodes=TOTAL_EPISODES),
            validation_targets=list(VALIDATION_TARGETS),
            historical_control=dict(
                run="SpatialTaskBattery/BiasedTraining",
                matched_validation_steps=list(MATCHED_CONTROL_VALIDATION_TARGETS),
                terminal_12400_available=False,
                final_heldout_available=False,
                limitation=(
                    "Direct control comparison is limited to matched prior "
                    "validation points 9200/10000/10800/11600; the historical "
                    "biased control was user-stopped before 12400/final tests."
                ),
            ),
            selection=(
                "maximum [minimum chance-normalized task BA, mean task AUC], "
                "same as original-bias five-task run"
            ),
            interpretation=(
                "One authorized prospective-query arm only; no resumed/control "
                "training. Terminal and final held-out results are unmatched "
                "exploratory unless a matched control later exists."
            ),
        )
    if time.time() >= deadline - 300:
        raise TimeoutError("Insufficient remaining cloud time; budget cannot renew")

    def publish():
        write(budget_path, ledger)
        write(aggregate_path, aggregate)

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            extra.pop("checkpoint", None)
            checkpoint = newest_indexed_checkpoint(out, extra["target"])
            if checkpoint is not None:
                extra["checkpoint"] = checkpoint
        job_path = root / f"{tag}_job.json"
        result_path = root / f"{tag}_result.json"
        log_path = root / f"{tag}.log"
        job = dict(
            kind=kind,
            arm=ARM,
            config=cfg,
            out=str(out),
            result=str(result_path),
            deadline=deadline - 180,
            parent=str(parent),
            parent_sha256=PARENT_SHA256,
            source_hashes=manifest["source_hashes"],
            **extra,
        )
        if result_path.exists():
            cached = read(result_path)
            if cached.get("status") == "completed":
                if not job_path.exists() or read(job_path) != job:
                    raise RuntimeError("Completed result cache has a mismatched job")
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
                pid=process.pid, tag=tag, kind=kind, log=str(log_path)
            )
            publish()
            try:
                code = process.wait(timeout=max(0.01, deadline - time.time() - 180))
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
                returncode=code,
                wall_seconds=time.time() - tick,
            )
        )
        ledger["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} exited {code}; see {log_path}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError("Incomplete bounded worker " + tag)
        return result

    publish()
    try:
        profile_names = []
        for task, names in cfg["train_names"].items():
            ordered = sorted(
                names,
                key=lambda name: frame_count(task, cfg["cells"][name]["condition"]),
            )
            profile_names.extend(dict.fromkeys((ordered[0], ordered[-1])))
        profile = launch(
            "profile",
            "profile",
            root / "profile",
            profile_names=profile_names,
        )
        aggregate["profile"] = profile
        if "profile_projection" not in aggregate:
            measured = {row["name"]: row for row in profile["rows"]}
            task_costs = []
            for task, names in cfg["train_names"].items():
                endpoints = [
                    name
                    for name in profile_names
                    if cfg["cells"][name]["task"] == task
                ]
                low, high = endpoints[0], endpoints[-1]
                low_frames = frame_count(task, cfg["cells"][low]["condition"])
                high_frames = frame_count(task, cfg["cells"][high]["condition"])
                estimates = []
                for name in names:
                    fraction = (
                        0
                        if high_frames == low_frames
                        else (
                            frame_count(task, cfg["cells"][name]["condition"])
                            - low_frames
                        )
                        / (high_frames - low_frames)
                    )
                    estimates.append(
                        (1 - fraction) * measured[low]["seconds"]
                        + fraction * measured[high]["seconds"]
                    )
                task_costs.append(sum(estimates) / len(estimates))
            projected_training = 1.25 * UPDATES * sum(task_costs)
            reserve = 1800
            aggregate["profile_projection"] = dict(
                training_seconds=projected_training,
                evaluation_and_retrieval_reserve_seconds=reserve,
                remaining_seconds=deadline - time.time(),
                exposure_pinned=True,
            )
            publish()
            if projected_training + reserve > deadline - time.time():
                raise RuntimeError(
                    "Pinned 4000-update exposure does not fit; no automatic reduction"
                )

        aggregate["status"] = ledger["status"] = "training"
        publish()
        checkpoint = None
        best = None
        training_out = root / ARM / "training"
        for target in VALIDATION_TARGETS:
            train = launch(
                f"train_{target}",
                "train",
                training_out,
                checkpoint=checkpoint,
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
                comparison_status=comparison_status(target),
                comparison_limitation=aggregate["historical_control"]["limitation"],
            )
            by_step = {
                value["step"]: value
                for value in record.get("validation", [])
                if value.get("step") != target
            }
            by_step[target] = validation
            record["validation"] = [by_step[step] for step in sorted(by_step)]
            ranked = max(
                record["validation"],
                key=lambda value: (tuple(value["rank"]), -value["step"]),
            )
            best = ranked
            record["selected_checkpoint"] = ranked["checkpoint"]
            record["selected_step"] = ranked["step"]
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
            comparison_status=comparison_status(
                record["selected_step"], heldout=True
            ),
            comparison_limitation=aggregate["historical_control"]["limitation"],
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
                comparison_status=comparison_status(
                    VALIDATION_TARGETS[-1], heldout=True
                ),
                comparison_limitation=aggregate["historical_control"]["limitation"],
            )
        )
        record["comparison_limitation"] = aggregate["historical_control"]["limitation"]
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
                wall_seconds=aggregate["wall_seconds"],
                deadline_unix=deadline,
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
