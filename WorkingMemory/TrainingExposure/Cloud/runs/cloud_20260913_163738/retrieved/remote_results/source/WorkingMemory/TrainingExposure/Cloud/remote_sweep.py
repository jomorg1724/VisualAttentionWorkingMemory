"""Cloud location adapter only; common supervisor owns protocol and training."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from WorkingMemory.TrainingExposure.sweep import run
if __name__=='__main__':
    manifest=json.loads((ROOT/'portable/manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Bundle changed: '+name)
    run('focused_50',ROOT/'remote_results',parent=ROOT/'portable/parent_checkpoint_008400.pt',external_manifest=manifest)
