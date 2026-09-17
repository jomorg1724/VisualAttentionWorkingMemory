"""Remote supervisor for the AV-context v2 overnight arm (long fixed-deadline run).

Same worker, model, protocol and construction gate as the local run. The
exposure target comes from the manifest; training stops gracefully at the
deadline reserve and the latest checkpoint is validated and tested. The
pre-registered 2,400 gate is evaluated and recorded but does not stop this
arm, by the user's overnight instruction.
"""
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
from WorkingMemory.AttentionContextComparator.V2.model import ARM, VERSION
from WorkingMemory.AttentionContextComparator.V2.protocol import (
    GATE_STEP,
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    VALIDATION_N,
    assert_fixed,
    gate,
    recipe,
)

RESULTS_NAME = "av_context_v2_results"
TRAIN_RESERVE_SECONDS = 2700  # final validation + tests + indexing before the deadline
EVAL_RESERVE_SECONDS = 240


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
    if {p.name for p in files} != seen:
        raise RuntimeError("Checkpoint/index inventory mismatch")
    eligible = [row for row in rows if int(row["step"]) <= int(target)]
    return str(directory / max(eligible, key=lambda r: int(r["step"]))["file"]) if eligible else None


def run(manifest_path):
    import numpy
    import PIL
    import scipy
    import torch

    manifest = read(manifest_path)
    root = ROOT / RESULTS_NAME
    root.mkdir(exist_ok=True)
    deadline = float(manifest["deadline_unix"])
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Pinned source changed " + name)
    versions = dict(
        python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
        scipy=scipy.__version__, pillow=PIL.__version__, platform=platform.platform(),
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
    )
    expected = manifest["expected_runtime"]
    if (versions["torch"], versions["numpy"], versions["scipy"], versions["pillow"]) != (expected["torch"], expected["numpy"], expected["scipy"], expected["pillow"]):
        raise RuntimeError("Unpinned cloud runtime " + repr(versions))
    if "A100" in versions["gpu"].upper():
        raise RuntimeError("A100 is explicitly prohibited")
    cfg = recipe()
    assert_fixed(cfg)
    cfg["updates"] = int(manifest["updates"])
    targets = list(manifest["validation_targets"])
    if targets[-1] != cfg["updates"] or any(b <= a for a, b in zip(targets, targets[1:])):
        raise RuntimeError("Malformed validation targets")
    source_hashes = manifest["files"]
    budget_path = root / "budget.json"
    aggregate_path = root / "aggregate.json"
    if budget_path.exists():
        raise RuntimeError("Existing ledger; this supervisor does not resume")
    started = time.time()
    ledger = dict(started_unix=started, deadline_unix=deadline, status="construction", active=None, events=[], source_hashes=source_hashes)
    record = dict(config=cfg, validation=[])
    aggregate = dict(
        status="construction", platform=versions, version=VERSION, arm=ARM,
        initialization_kind="full_model_from_scratch", loaded_parent=False, loaded_optimizer_state=False,
        source_hashes=source_hashes, v2_changes=cfg["v2_changes"],
        learning_rates=dict(new_lr=cfg["new_lr"], parent_lr=cfg["parent_lr"]),
        requested_exposure=dict(updates=cfg["updates"], episodes=cfg["updates"] * 40),
        validation_targets=targets, gate_step=GATE_STEP, gate_policy="informational only; overnight arm is not stopped by the gate",
        selection="maximum [minimum chance-normalized task BA, mean task AUC], then earlier step",
        run={ARM: record}, manifest=manifest,
    )

    def publish():
        write(budget_path, ledger)
        write(aggregate_path, aggregate)

    def launch(tag, kind, out, job_deadline, **extra):
        out = Path(out)
        if kind == "train":
            checkpoint = newest_checkpoint(out, extra["target"])
            if checkpoint:
                extra["checkpoint"] = checkpoint
        job_path = root / f"{tag}_job.json"
        result_path = root / f"{tag}_result.json"
        log_path = root / f"{tag}.log"
        job = dict(kind=kind, arm=ARM, config=cfg, out=str(out), result=str(result_path), deadline=job_deadline, device="cuda", source_hashes=source_hashes, **extra)
        write(job_path, job)
        tick = time.time()
        with log_path.open("a") as handle:
            process = subprocess.Popen([sys.executable, "-B", str(HERE / "worker.py"), str(job_path)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            ledger["active"] = dict(pid=process.pid, kind=kind, tag=tag, log=str(log_path))
            publish()
            try:
                code = process.wait(timeout=max(0.01, job_deadline + 60 - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        ledger["events"].append(dict(tag=tag, kind=kind, pid=process.pid, returncode=code, seconds=time.time() - tick))
        ledger["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} exited {code}; see {log_path}")
        result = read(result_path)
        if result["status"] not in ("completed", "budget_stopped"):
            raise RuntimeError("Bounded worker incomplete " + tag)
        return result

    publish()
    try:
        construction = subprocess.run([sys.executable, "-B", str(HERE / "check.py")], cwd=ROOT, capture_output=True, text=True, timeout=1200, env=dict(os.environ))
        (root / "construction_stdout.log").write_text(construction.stdout)
        (root / "construction_stderr.log").write_text(construction.stderr)
        if construction.returncode:
            raise RuntimeError("Remote construction/identity gate failed")
        checks = read(HERE / "construction_checks.json")
        if checks["status"] != "passed" or not checks["identity_gate"]["bit_identical_logits_cpu"]:
            raise RuntimeError("Identity gate not passed remotely")
        aggregate["construction_checks"] = checks
        aggregate["status"] = ledger["status"] = "profiling"
        publish()
        profile = launch("profile", "profile", root / "profile", deadline - EVAL_RESERVE_SECONDS, profile_updates=3)
        aggregate["profile"] = profile
        mean_update = float(profile["mean_update_seconds"])
        aggregate["profile_projection"] = dict(
            mean_update_seconds=mean_update,
            projected_training_hours_full=cfg["updates"] * mean_update / 3600,
            train_hours_available=(deadline - TRAIN_RESERVE_SECONDS - time.time()) / 3600,
            projected_reachable_updates=int((deadline - TRAIN_RESERVE_SECONDS - time.time()) / mean_update),
            peak_allocated_gb=(profile["peak_allocated_bytes"] or 0) / 1e9,
        )
        aggregate["status"] = ledger["status"] = "training"
        publish()
        training_out = root / ARM / "training"
        checkpoint = None
        selected = None
        for target in targets:
            if time.time() > deadline - TRAIN_RESERVE_SECONDS - 60:
                break
            trained = launch(f"train_{target}", "train", training_out, deadline - TRAIN_RESERVE_SECONDS, target=target)
            checkpoint = trained["checkpoint"]
            record["training"] = trained
            step = int(trained["step"])
            validation = launch(f"validation_{step}", "eval", root / ARM / f"validation_{step}", deadline - EVAL_RESERVE_SECONDS, checkpoint=checkpoint, split="val", eval_seed=SPATIAL_VAL_SEED, n=VALIDATION_N)
            previous = {v["step"]: v for v in record["validation"] if v["step"] != step}
            previous[step] = validation
            record["validation"] = [previous[k] for k in sorted(previous)]
            selected = max(record["validation"], key=lambda v: (tuple(v["rank"]), -v["step"]))
            record["selected_checkpoint"] = selected["checkpoint"]
            record["selected_step"] = selected["step"]
            if step == GATE_STEP:
                record["gate"] = gate(validation)
            publish()
            if trained["status"] == "budget_stopped":
                aggregate["budget_stopped_at_step"] = step
                break
        record["terminal_checkpoint"] = checkpoint
        record["terminal_step"] = int(record["training"]["step"])
        aggregate["status"] = ledger["status"] = "evaluating"
        publish()
        record["selected_test"] = launch("selected_test", "eval", root / ARM / "selected_test", deadline - EVAL_RESERVE_SECONDS, checkpoint=record["selected_checkpoint"], split="test", eval_seed=SPATIAL_TEST_SEED, n=TEST_N)
        record["terminal_test"] = (
            record["selected_test"] if record["selected_step"] == record["terminal_step"]
            else launch("terminal_test", "eval", root / ARM / "terminal_test", deadline - EVAL_RESERVE_SECONDS, checkpoint=checkpoint, split="test", eval_seed=SPATIAL_TEST_SEED, n=TEST_N)
        )
        record["status"] = aggregate["status"] = ledger["status"] = "completed"
    except BaseException as error:
        aggregate.update(status="failed", error=repr(error))
        ledger["status"] = "failed"
        raise
    finally:
        aggregate["wall_seconds"] = time.time() - started
        publish()
        write(root / "exit.json", dict(status=aggregate["status"], error=aggregate.get("error"), deadline_unix=deadline, wall_seconds=aggregate["wall_seconds"]))
        write(root / "artifact_index.json", {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in root.rglob("*") if p.is_file() and p.name != "artifact_index.json"})


if __name__ == "__main__":
    run(sys.argv[1])
