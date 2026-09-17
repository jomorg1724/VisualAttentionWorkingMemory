"""Hash-pinned portable source bundle for the AV-context v2 cloud arm."""
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from PreAttentiveVision.train import sha

REFERENCE = ROOT / "WorkingMemory/SpatialTaskBattery/BiasedTraining/launch_receipt.json"
EXPERIMENT = (
    "WorkingMemory/SpatialPriorityReadout/model.py",
    "WorkingMemory/AttentionContextComparator/model.py",
    "WorkingMemory/AttentionContextComparator/protocol.py",
    "WorkingMemory/AttentionContextComparator/V2/__init__.py",
    "WorkingMemory/AttentionContextComparator/V2/model.py",
    "WorkingMemory/AttentionContextComparator/V2/protocol.py",
    "WorkingMemory/AttentionContextComparator/V2/check.py",
    "WorkingMemory/AttentionContextComparator/V2/worker.py",
    "WorkingMemory/AttentionContextComparator/V2/cloud_sweep.py",
)
EXCLUDE = (
    "WorkingMemory/SpatialTaskBattery/BiasedTraining/",
    "WorkingMemory/UnbiasedAttention/model.py",
    "WorkingMemory/UnbiasedAttention/sweep.py",
    "WorkingMemory/UnbiasedAttention/worker.py",
)


def prepare(output, deadline, updates, validation_targets):
    if output.exists():
        raise FileExistsError("Refusing to overwrite bundle")
    reference = json.loads(REFERENCE.read_text())
    sources = {name for name in reference["source_hashes"] if not name.startswith(EXCLUDE)}
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
    for package in ("WorkingMemory", "WorkingMemory/AttentionContextComparator", "PreAttentiveVision", "PreAttentiveVision/TemporalIntegration"):
        init = ROOT / package / "__init__.py"
        if init.is_file():
            target = output / package / "__init__.py"
            shutil.copy2(init, target)
            hashes[package + "/__init__.py"] = sha(target)
    manifest = dict(
        experiment="attention_context_comparator_v2_overnight",
        status="prepared_not_launched",
        deadline_unix=float(deadline),
        updates=int(updates),
        validation_targets=list(validation_targets),
        initialization_kind="full_model_from_scratch",
        loaded_parent=False,
        loaded_optimizer_state=False,
        files=hashes,
        source_hashes=hashes,
        expected_runtime=dict(torch="1.13.1+cu117", numpy="1.23.1", scipy="1.8.1", pillow="9.1.1"),
        authorized_arms=["attention_context_comparator_v2"],
        entrypoint="WorkingMemory/AttentionContextComparator/V2/cloud_sweep.py",
        command="python -B WorkingMemory/AttentionContextComparator/V2/cloud_sweep.py portable/manifest.json",
    )
    portable = output / "portable"
    portable.mkdir()
    (portable / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(output)).replace("\\", "/"))
    return manifest, archive, sha(archive)
