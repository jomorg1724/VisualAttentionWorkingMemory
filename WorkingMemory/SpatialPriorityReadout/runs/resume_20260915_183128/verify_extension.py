"""Read-only verification of live trajectory continuity and extension state."""
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path("/workspace/vawm_spatial_priority_scratch_resume_1600")
RESULTS = ROOT / "spatial_priority_results"
TRAINING = RESULTS / "spatial_priority_readout_scratch" / "training"
rows = list(csv.DictReader((TRAINING / "metrics.csv").open()))
steps = [int(row["step"]) for row in rows]
assert steps == list(range(1601, steps[-1] + 1))
index = [
    json.loads(line)
    for line in (TRAINING / "checkpoint_index.jsonl").read_text().splitlines()
]
checkpoint_steps = [int(row["step"]) for row in index]
assert checkpoint_steps == sorted(set(checkpoint_steps))
last = index[-1]
path = TRAINING / last["file"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == last["sha256"]
checkpoint = torch.load(path, map_location="cpu")
assert checkpoint["step"] == last["step"]
assert checkpoint["counts"]["episodes"] == last["step"] * 40
ledger = json.loads((RESULTS / "budget.json").read_text())
manifest = json.loads((ROOT / "portable" / "resume_manifest.json").read_text())
assert ledger["deadline_unix"] == manifest["deadline_unix"] == 1789560600.0
assert manifest["posthoc_protocol_extension"]["extended_target_updates"] == 12000
for pid in (3052, 3668):
    os.kill(pid, 0)
print(
    json.dumps(
        {
            "metric_first": steps[0],
            "metric_last": steps[-1],
            "metric_rows": len(steps),
            "duplicates": len(steps) - len(set(steps)),
            "checkpoint_step": checkpoint["step"],
            "checkpoint_episodes": checkpoint["counts"]["episodes"],
            "checkpoint_sha256": last["sha256"],
            "next_update": steps[-1] + 1,
            "deadline_unix": ledger["deadline_unix"],
            "target_updates": 12000,
            "old_supervisor_1716_alive": Path("/proc/1716").exists(),
            "inflight_worker_pid": 3052,
            "handoff_pid": 3668,
            "ledger_active": ledger["active"],
            "validation_targets": manifest["posthoc_protocol_extension"][
                "validation_targets"
            ],
        },
        indent=2,
    )
)
