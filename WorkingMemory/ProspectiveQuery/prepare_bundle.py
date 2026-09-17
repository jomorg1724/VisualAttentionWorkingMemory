"""Prepare a hash-pinned portable bundle without provisioning or launching cloud."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import sha
from WorkingMemory.ProspectiveQuery.model import PARENT_SHA256

PARENT = (
    ROOT
    / "WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/"
    "retrieved/remote_results/preupdate_attention/checkpoint_008400.pt"
)
REFERENCE_RECEIPT = (
    ROOT / "WorkingMemory/SpatialTaskBattery/BiasedTraining/launch_receipt.json"
)
EXPERIMENT_SOURCES = (
    "WorkingMemory/ProspectiveQuery/model.py",
    "WorkingMemory/ProspectiveQuery/protocol.py",
    "WorkingMemory/ProspectiveQuery/worker.py",
    "WorkingMemory/ProspectiveQuery/sweep.py",
)


def prepare(output, deadline_unix):
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing bundle")
    if sha(PARENT) != PARENT_SHA256:
        raise RuntimeError("Original attention8400 checkpoint hash mismatch")
    reference = json.loads(REFERENCE_RECEIPT.read_text())
    sources = set(reference["source_hashes"]) | set(EXPERIMENT_SOURCES)
    output.mkdir(parents=True)
    hashes = {}
    for relative in sorted(sources):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = sha(destination)
    portable = output / "portable"
    portable.mkdir()
    parent_copy = portable / "parent_checkpoint_008400.pt"
    shutil.copy2(PARENT, parent_copy)
    manifest = dict(
        experiment="prospective_query_v1",
        status="prepared_not_launched",
        deadline_unix=float(deadline_unix),
        parent_sha256=PARENT_SHA256,
        files=hashes,
        source_hashes=hashes,
        entrypoint="WorkingMemory/ProspectiveQuery/sweep.py",
        command=(
            "python -B WorkingMemory/ProspectiveQuery/sweep.py "
            "portable/manifest.json"
        ),
        fixed_exposure=dict(updates=4000, episodes=160000, batch_size=8),
        authorized_arms=["prospective_query"],
        historical_control=dict(
            matched_validation_steps=[9200, 10000, 10800, 11600],
            terminal_12400_available=False,
            final_heldout_available=False,
        ),
    )
    (portable / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    manifest = prepare(args.output.resolve(), args.deadline_unix)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
