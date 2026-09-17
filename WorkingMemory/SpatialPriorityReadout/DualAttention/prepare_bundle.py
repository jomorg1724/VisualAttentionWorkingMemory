"""Create a hash-pinned, scratch-only portable dual-attention bundle."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import sha
from WorkingMemory.SpatialPriorityReadout.DualAttention.protocol import TOTAL_EPISODES, UPDATES

REFERENCE = ROOT / "WorkingMemory/SpatialTaskBattery/BiasedTraining/launch_receipt.json"
CONTROL = ROOT / "WorkingMemory/AttentionContextComparator/fixtures/control_checkpoint_000000.pt"
CONTROL_SHA256 = "97b7e6ce513592fc094724854ea1d4d329c93fd1ae0d86aa20ed50e4fd653f76"
EXPERIMENT = (
    "WorkingMemory/SpatialPriorityReadout/model.py",
    "WorkingMemory/SpatialPriorityReadout/DualAttention/model.py",
    "WorkingMemory/SpatialPriorityReadout/DualAttention/protocol.py",
    "WorkingMemory/SpatialPriorityReadout/DualAttention/check_model.py",
    "WorkingMemory/SpatialPriorityReadout/DualAttention/worker.py",
    "WorkingMemory/SpatialPriorityReadout/DualAttention/sweep.py",
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
    if sha(CONTROL) != CONTROL_SHA256:
        raise RuntimeError("Control checkpoint-0 fixture mismatch")
    reference = json.loads(REFERENCE.read_text())
    sources = {
        name for name in reference["source_hashes"] if not name.startswith(EXCLUDE)
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
    control_copy = output / "WorkingMemory/AttentionContextComparator/fixtures/control_checkpoint_000000.pt"
    control_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONTROL, control_copy)
    portable = output / "portable"
    portable.mkdir()
    manifest = dict(
        experiment="dual_attention_spatial_priority_scratch_v1",
        status="prepared_not_launched",
        deadline_unix=float(deadline),
        initialization_kind="full_model_from_scratch",
        loaded_parent=False,
        loaded_optimizer_state=False,
        control_initialization_fixture_sha256=CONTROL_SHA256,
        control_initialization_fixture_use="construction equality check only",
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
        authorized_arms=["dual_attention_spatial_priority_scratch"],
        architecture=(
            "Scratch one-head width64 pre attention C->W_O_pre memory drive; "
            "unchanged spatial E/I; one-head width64 post attention R queries "
            "[C,H,R]; only terminal P enters convolutional spatial-priority decoder."
        ),
        entrypoint="WorkingMemory/SpatialPriorityReadout/DualAttention/sweep.py",
        command=(
            "python -B WorkingMemory/SpatialPriorityReadout/DualAttention/sweep.py "
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
    print(json.dumps(prepare(args.output.resolve(), args.deadline_unix), indent=2))


if __name__ == "__main__":
    main()
