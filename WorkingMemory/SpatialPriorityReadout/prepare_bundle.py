"""Create a minimal hash-pinned portable bundle; provisioning is separate."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import sha
from WorkingMemory.SpatialPriorityReadout.protocol import TOTAL_EPISODES, UPDATES

REFERENCE = ROOT / "WorkingMemory/SpatialTaskBattery/BiasedTraining/launch_receipt.json"
EXPERIMENT = (
    "WorkingMemory/SpatialPriorityReadout/model.py",
    "WorkingMemory/SpatialPriorityReadout/protocol.py",
    "WorkingMemory/SpatialPriorityReadout/check_model.py",
    "WorkingMemory/SpatialPriorityReadout/worker.py",
    "WorkingMemory/SpatialPriorityReadout/sweep.py",
)
EXCLUDE = (
    "WorkingMemory/SpatialTaskBattery/BiasedTraining/",
    "WorkingMemory/UnbiasedAttention/model.py",
    "WorkingMemory/UnbiasedAttention/sweep.py",
    "WorkingMemory/UnbiasedAttention/worker.py",
)


def prepare(output, deadline):
    if output.exists():
        raise FileExistsError("Refusing to overwrite bundle")
    reference = json.loads(REFERENCE.read_text())
    sources = {
        name
        for name in reference["source_hashes"]
        if not name.startswith(EXCLUDE)
    }
    sources.update(EXPERIMENT)
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
    manifest = dict(
        experiment="spatial_priority_readout_scratch_v2",
        status="prepared_not_launched",
        deadline_unix=float(deadline),
        initialization_kind="full_model_from_scratch",
        loaded_parent=False,
        loaded_optimizer_state=False,
        files=hashes,
        source_hashes=hashes,
        expected_runtime=dict(
            torch="1.13.1+cu117",
            numpy="1.23.1",
            scipy="1.8.1",
            pillow="9.1.1",
        ),
        fixed_exposure=dict(
            updates=UPDATES,
            episodes=TOTAL_EPISODES,
            batch_size=8,
            microbatches_per_update=5,
        ),
        authorized_arms=["spatial_priority_readout_scratch"],
        architecture=(
            "Entire encoder/opponent/attention/EI/comparator/readout model initialized "
            "from documented seeds. Concatenate final H/R/C as 192x13x13; "
            "1x1 192->96 and 3x3 96->64; task-specific spatial selection and "
            "class-evidence 1x1 maps; spatial softmax weighted evidence."
        ),
        entrypoint="WorkingMemory/SpatialPriorityReadout/sweep.py",
        command=(
            "python -B WorkingMemory/SpatialPriorityReadout/sweep.py "
            "portable/manifest.json"
        ),
    )
    (portable / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.output.resolve(), args.deadline_unix), indent=2)
    )


if __name__ == "__main__":
    main()
