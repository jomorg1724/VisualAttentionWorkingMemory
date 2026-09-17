"""Hand off an in-flight training worker to the extended supervisor without replay."""
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/vawm_spatial_priority_scratch_resume_1600")
RESULTS = ROOT / "spatial_priority_results"
TRAINING = RESULTS / "spatial_priority_readout_scratch" / "training"
ACTIVE_PID = 3052
EXPECTED_TARGET = 3200


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


started = time.time()
while alive(ACTIVE_PID):
    time.sleep(5)

result = json.loads((RESULTS / "train_3200_result.json").read_text())
rows = list(csv.DictReader((TRAINING / "metrics.csv").open()))
steps = [int(row["step"]) for row in rows]
index = [
    json.loads(line)
    for line in (TRAINING / "checkpoint_index.jsonl").read_text().splitlines()
]
checkpoint = TRAINING / f"checkpoint_{EXPECTED_TARGET:06d}.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if (
    result.get("status") != "completed"
    or result.get("step") != EXPECTED_TARGET
    or steps != list(range(1601, EXPECTED_TARGET + 1))
    or index[-1]["step"] != EXPECTED_TARGET
    or index[-1]["sha256"] != digest
    or result["counts"]["episodes"] != EXPECTED_TARGET * 40
):
    raise RuntimeError("Active-worker continuity verification failed")

log = (ROOT / "remote_supervisor_extension.log").open("a")
process = subprocess.Popen(
    [
        "/workspace/spatial-priority-scratch-resume-venv/bin/python",
        "-B",
        "WorkingMemory/SpatialPriorityReadout/resume_sweep.py",
        "portable/resume_manifest.json",
    ],
    cwd=ROOT,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(ROOT / "remote_supervisor.pid").write_text(f"{process.pid}\n")
write(
    ROOT / "extension_handoff_receipt.json",
    {
        "status": "extended_supervisor_launched",
        "old_supervisor_pid": 1716,
        "completed_worker_pid": ACTIVE_PID,
        "continuity_step": EXPECTED_TARGET,
        "continuity_episodes": EXPECTED_TARGET * 40,
        "metric_first_step": steps[0],
        "metric_last_step": steps[-1],
        "metric_rows": len(steps),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": digest,
        "new_supervisor_pid": process.pid,
        "launched_unix": time.time(),
        "wait_seconds": time.time() - started,
    },
)
