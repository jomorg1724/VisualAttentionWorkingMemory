"""Atomically mark the live ledger as extended while its inherited worker runs."""
import csv
import json
import os
import time
from pathlib import Path

ROOT = Path("/workspace/vawm_spatial_priority_scratch_resume_1600")
RESULTS = ROOT / "spatial_priority_results"
DEADLINE = 1789560600.0
TARGETS = list(range(2400, 12001, 800))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


ledger_path = RESULTS / "budget.json"
aggregate_path = RESULTS / "aggregate.json"
ledger = read(ledger_path)
aggregate = read(aggregate_path)
extension = read(ROOT / "protocol_extension_receipt.json")["extension"]
rows = list(
    csv.DictReader(
        (
            RESULTS
            / "spatial_priority_readout_scratch"
            / "training"
            / "metrics.csv"
        ).open()
    )
)
active = ledger.get("active")
if (
    ledger["status"] != "training"
    or aggregate["status"] != "training"
    or active is None
    or active["pid"] != 3052
    or active["tag"] != "train_3200"
):
    raise RuntimeError("Live ledger no longer matches the handoff")
os.kill(3052, 0)
observed_step = int(rows[-1]["step"])
event = {
    "kind": "posthoc_extension_armed_during_inflight_train",
    "unix": time.time(),
    "observed_step": observed_step,
    "inflight_worker_pid": 3052,
    "inflight_target": 3200,
    "handoff_pid": int((ROOT / "extension_handoff.pid").read_text()),
    "original_target_updates": 4000,
    "extended_target_updates": 12000,
    "deadline_unix": DEADLINE,
    "validation_targets": TARGETS,
    "reason": extension["reason"],
}
ledger["deadline_unix"] = DEADLINE
ledger["events"].append(event)
aggregate["deadline_unix"] = DEADLINE
aggregate["posthoc_protocol_extension"] = extension
aggregate["protocol_extension_status"] = event
aggregate["resume"]["remaining_validation_targets"] = TARGETS
write(ledger_path, ledger)
write(aggregate_path, aggregate)
write(
    ROOT / "extension_armed_receipt.json",
    {
        "status": "armed",
        "ledger_event": event,
        "old_supervisor_pid": 1716,
        "old_supervisor_stopped": not Path("/proc/1716").exists(),
        "worker_alive": Path("/proc/3052").exists(),
        "next_step_observed": observed_step + 1,
        "controller_sha256": extension_controller_sha
        if (extension_controller_sha := __import__("hashlib").sha256(
            (
                ROOT
                / "WorkingMemory"
                / "SpatialPriorityReadout"
                / "resume_sweep.py"
            ).read_bytes()
        ).hexdigest())
        else None,
        "manifest_deadline_unix": read(ROOT / "portable" / "resume_manifest.json")[
            "deadline_unix"
        ],
    },
)
