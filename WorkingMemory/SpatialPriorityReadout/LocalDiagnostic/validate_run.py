"""Validate immutable outputs from the completed local diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = ROOT / args.run
    result = json.loads((run / "results.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        path = run / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Artifact digest mismatch: {relative}")
    checkpoint = ROOT / result["checkpoint"]
    if sha(checkpoint) != result["checkpoint_sha256_before"]:
        raise RuntimeError("Source checkpoint changed")
    assert result["checkpoint_immutable"]
    assert result["frozen_model_state_immutable"]
    assert result["within_1800_second_shared_cap"]
    assert result["cloud_calls"] == 0
    assert result["deployed_weight_updates"] == 0
    assert len(set(result["split_seeds"].values())) == 3
    assert result["samples"] == {"train": 512, "val": 128, "test": 256}
    assert all(
        result["extraction"][split]["paired_motion_movie_metadata_across_delays"]
        for split in ("train", "val", "test")
    )
    assert all(
        result["extraction"][split][
            "max_original_logit_reconstruction_error"
        ] < 2e-6
        for split in ("train", "val", "test")
    )
    maps = np.load(run / "priority_maps_test.npz")
    assert maps["priority"].shape == (1024, 13, 13)
    assert np.allclose(maps["priority"].sum((1, 2)), 1, atol=2e-6)
    assert len(set(maps["base_ids"].tolist())) == 256
    assert len(set(maps["trial_ids"].tolist())) == 1024
    for probe in ("pooled", "spatial"):
        assert result["probes"][probe]["selected"]["epoch"] in (5, 10, 20, 40)
        payload = torch.load(run / f"{probe}_selected.pt", map_location="cpu")
        assert payload["selected_epoch"] == result["probes"][probe]["selected"]["epoch"]
        assert payload["parameter_count"] == result["probes"][probe]["parameters"]
    receipt = {
        "status": "passed",
        "manifest_files_verified": len(manifest["files"]),
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_immutable": True,
        "frozen_state_immutable": True,
        "priority_maps_normalized": True,
        "test_presentations": int(len(maps["labels"])),
        "independent_test_base_episodes": int(len(set(maps["base_ids"].tolist()))),
        "split_seeds_distinct": True,
        "original_logit_reconstruction_max_error": max(
            result["extraction"][split][
                "max_original_logit_reconstruction_error"
            ]
            for split in ("train", "val", "test")
        ),
        "within_1800_second_shared_cap": True,
        "cloud_calls": 0,
        "deployed_weight_updates": 0,
    }
    (run / "validation_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
