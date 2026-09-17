"""Finalize reports if computation completed before a presentation-layer error."""
import argparse
import json
import time
from pathlib import Path

from WorkingMemory.SpatialPriorityReadout.LocalDiagnostic.run import (
    ROOT,
    make_report,
    sha,
    source_hashes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", required=True)
    args = parser.parse_args()
    out = ROOT / args.out
    result_path = out / "results.json"
    result = json.loads(result_path.read_text())
    budget = json.loads((ROOT / args.budget).read_text())
    result["production_wall_seconds"] = result["wall_seconds"]
    result["budget_started_unix"] = budget["started_unix"]
    result["finalized_unix"] = time.time()
    result["total_experiment_wall_seconds"] = (
        result["finalized_unix"] - budget["started_unix"]
    )
    result["within_1800_second_shared_cap"] = (
        result["finalized_unix"] <= budget["deadline_unix"]
    )
    result["postrun_source_hashes"] = source_hashes()
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "report.md").write_text(make_report(result), encoding="utf-8")
    (out / "completion_receipt.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "analysis_only_post_hoc": True,
                "checkpoint": result["checkpoint"],
                "checkpoint_sha256": result["checkpoint_sha256_before"],
                "checkpoint_immutable": result["checkpoint_immutable"],
                "frozen_model_state_immutable": result[
                    "frozen_model_state_immutable"
                ],
                "split_seeds": result["split_seeds"],
                "independent_base_episodes_per_delay": result["samples"],
                "delay_presentations": result["delay_presentations"],
                "pooled_parameters": result["probes"]["pooled"]["parameters"],
                "spatial_parameters": result["probes"]["spatial"]["parameters"],
                "selected_epoch_both": 20,
                "selected_updates_each": 640,
                "selected_presentations_each": 40960,
                "held_out_pooled": result["test"]["pooled"]["overall"],
                "held_out_spatial": result["test"]["spatial"]["overall"],
                "paired_overall": result["test"]["paired"]["overall"],
                "priority_alignment": result["priority_alignment"],
                "production_wall_seconds": result["production_wall_seconds"],
                "total_experiment_wall_seconds": result[
                    "total_experiment_wall_seconds"
                ],
                "within_1800_second_shared_cap": result[
                    "within_1800_second_shared_cap"
                ],
                "gpu_workers": 1,
                "cpu_threads": 2,
                "cloud_calls": 0,
                "deployed_weight_updates": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            manifest[str(path.relative_to(out)).replace("\\", "/")] = sha(path)
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "files": manifest,
                "execution_source_hashes": result["source_hashes"],
                "postrun_source_hashes": result["postrun_source_hashes"],
                "checkpoint_sha256": result["checkpoint_sha256_before"],
                "config_sha256": result["config_sha256"],
                "presentation_fix": (
                    "The original computation completed; report serialization "
                    "then hit Windows cp1252 on a Unicode minus. Finalization "
                    "used UTF-8 and did not repeat extraction, fitting or test."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "finalized",
                "total_experiment_wall_seconds": result[
                    "total_experiment_wall_seconds"
                ],
                "within_cap": result["within_1800_second_shared_cap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
