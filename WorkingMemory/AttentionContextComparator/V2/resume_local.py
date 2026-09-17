"""Resume an interrupted local v2 run in place from its newest indexed checkpoint.

Usage: python WorkingMemory/AttentionContextComparator/V2/resume_local.py <run_root>

Metric/diagnostic rows logged after the resume checkpoint are moved aside so the
trajectory stays exact (the worker restores model, optimizer, stream and RNG).
"""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import read, sha, write
from WorkingMemory.AttentionContextComparator.V2.model import ARM
from WorkingMemory.AttentionContextComparator.V2.protocol import (
    GATE_STEP,
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    TEST_N,
    VALIDATION_N,
    VALIDATION_TARGETS,
    gate,
)
from WorkingMemory.AttentionContextComparator.V2.run import SOURCE_PATHS, newest_checkpoint


def run(run_root):
    run_root = Path(run_root)
    aggregate = read(run_root / "aggregate.json")
    budget = read(run_root / "budget.json")
    cfg = aggregate["config"]
    deadline = float(budget["deadline_unix"])
    device = budget["device"]
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_PATHS}
    if source_hashes != budget["source_hashes"]:
        changed = [n for n in source_hashes if source_hashes[n] != budget["source_hashes"].get(n)]
        raise RuntimeError("Pinned sources changed since launch: " + repr(changed))
    training_out = run_root / "training"
    checkpoint = newest_checkpoint(training_out, VALIDATION_TARGETS[-1])
    resume_step = int(Path(checkpoint).stem.rsplit("_", 1)[1])
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    discarded = {}
    for name in ("metrics.csv", "live_diagnostics.jsonl"):
        path = training_out / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        if name == "metrics.csv":
            keep = [lines[0]] + [l for l in lines[1:] if int(l.split(",", 1)[0]) <= resume_step]
        else:
            keep = [l for l in lines if json.loads(l)["step"] <= resume_step]
        dropped = len(lines) - len(keep)
        if dropped:
            path.rename(training_out / f"{name}.before_resume_{stamp}")
            path.write_text("\n".join(keep) + "\n")
        discarded[name] = dropped
    receipt = dict(kind="in_place_resume", resumed_unix=time.time(), resume_checkpoint=checkpoint, resume_step=resume_step, discarded_uncheckpointed_rows=discarded, previous_active=budget.get("active"))
    write(run_root / f"resume_receipt_{stamp}.json", receipt)
    budget["events"].append(dict(kind="resume", unix=time.time(), step=resume_step))
    budget["active"] = None
    aggregate["resumes"] = aggregate.get("resumes", []) + [receipt]
    aggregate["status"] = budget["status"] = "training"

    def publish():
        write(run_root / "budget.json", budget)
        write(run_root / "aggregate.json", aggregate)
        write(run_root / "live_status.json", dict(status=aggregate["status"], active=budget["active"], selected_step=aggregate.get("selected_step"), validation_steps=[v["step"] for v in aggregate["validation"]], gate=aggregate.get("gate"), updated_unix=time.time(), deadline_unix=deadline))

    def launch(tag, kind, out, **extra):
        out = Path(out)
        if kind == "train":
            found = newest_checkpoint(out, extra["target"])
            if found:
                extra["checkpoint"] = found
        job_path = run_root / f"{tag}_job.json"
        result_path = run_root / f"{tag}_result.json"
        log_path = run_root / f"{tag}.log"
        job = dict(kind=kind, arm=ARM, config=cfg, out=str(out), result=str(result_path), deadline=deadline - 120, device=device, source_hashes=source_hashes, **extra)
        write(job_path, job)
        tick = time.time()
        with log_path.open("a") as handle:
            process = subprocess.Popen([sys.executable, "-X", "utf8", "-B", str(HERE / "worker.py"), str(job_path)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            budget["active"] = dict(pid=process.pid, tag=tag, kind=kind, log=str(log_path), started_unix=tick)
            publish()
            try:
                code = process.wait(timeout=max(0.01, deadline - time.time() - 120))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        budget["events"].append(dict(pid=process.pid, tag=tag, kind=kind, returncode=code, seconds=time.time() - tick))
        budget["active"] = None
        publish()
        if code:
            raise RuntimeError(f"{tag} failed with exit {code}; see {log_path}")
        result = read(result_path)
        if result["status"] != "completed":
            raise RuntimeError(f"{tag} did not complete: {result['status']}")
        return result

    publish()
    done = {v["step"] for v in aggregate["validation"]}
    try:
        checkpoint = aggregate.get("terminal_checkpoint")
        selected = None
        stopped_by_gate = False
        for target in VALIDATION_TARGETS:
            if target in done:
                continue
            trained = launch(f"train_{target}", "train", training_out, target=target)
            checkpoint = trained["checkpoint"]
            aggregate["training"] = trained
            validation = launch(f"validation_{target}", "eval", run_root / f"validation_{target}", checkpoint=checkpoint, split="val", eval_seed=SPATIAL_VAL_SEED, n=VALIDATION_N)
            aggregate["validation"].append(validation)
            selected = max(aggregate["validation"], key=lambda v: (tuple(v["rank"]), -v["step"]))
            aggregate["selected_step"] = selected["step"]
            aggregate["selected_checkpoint"] = selected["checkpoint"]
            if target == GATE_STEP:
                verdict = gate(validation)
                aggregate["gate"] = verdict
                write(run_root / "gate_receipt.json", dict(verdict=verdict, validation_summary=str(run_root / f"validation_{target}" / "summary.json"), unix=time.time()))
                publish()
                if not verdict["passed"]:
                    stopped_by_gate = True
                    break
            publish()
        aggregate["terminal_checkpoint"] = checkpoint
        if stopped_by_gate:
            aggregate["status"] = budget["status"] = "stopped_by_preregistered_gate"
        else:
            aggregate["status"] = budget["status"] = "evaluating"
            publish()
            selected = max(aggregate["validation"], key=lambda v: (tuple(v["rank"]), -v["step"]))
            aggregate["selected_test"] = launch("selected_test", "eval", run_root / "selected_test", checkpoint=selected["checkpoint"], split="test", eval_seed=SPATIAL_TEST_SEED, n=TEST_N)
            aggregate["terminal_test"] = aggregate["selected_test"] if selected["step"] == VALIDATION_TARGETS[-1] else launch("terminal_test", "eval", run_root / "terminal_test", checkpoint=checkpoint, split="test", eval_seed=SPATIAL_TEST_SEED, n=TEST_N)
            aggregate["status"] = budget["status"] = "completed"
    except BaseException as error:
        aggregate.update(status="failed", error=repr(error))
        budget["status"] = "failed"
        raise
    finally:
        aggregate["wall_seconds"] = time.time() - float(budget["started_unix"])
        publish()
        write(run_root / "exit.json", dict(status=aggregate["status"], error=aggregate.get("error"), wall_seconds=aggregate["wall_seconds"], deadline_unix=deadline))
        write(run_root / "artifact_index.json", {str(p.relative_to(run_root)).replace("\\", "/"): sha(p) for p in run_root.rglob("*") if p.is_file() and p.name != "artifact_index.json"})


if __name__ == "__main__":
    run(sys.argv[1])
