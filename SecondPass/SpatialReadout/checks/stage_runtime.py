"""Build a local byte-verified runtime outside protected Desktop for launchd.

No macOS permissions are changed. Source checkpoints/datasets stay untouched.
This is a build/copy operation, not a training entry point.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=Path(__file__).resolve().parents[3]
    destination=Path(sys.argv[1]).resolve()
    if destination.exists(): raise FileExistsError('Never overwrite a staged runtime')
    destination.mkdir(parents=True)
    files={}
    for package in ('PreAttentiveVision','WorkingMemory','SecondPass'):
        for directory,dirs,names in os.walk(source/package):
            dirs[:]=[d for d in dirs if d not in ('runs','locked_source','__pycache__','data','.git','node_modules','.venv')]
            for name in names:
                path=Path(directory)/name
                if path.suffix=='.py' or path.relative_to(source).as_posix() in (
                    'SecondPass/TaskSuite/catalog.json','SecondPass/SpatialReadout/BRIEF.md','SecondPass/SpatialReadout/README.md'):
                    target=destination/path.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,target)
                    assert sha(target)==sha(path)
                    files[str(path.relative_to(source))]=sha(target)
    data=source/'PreAttentiveVision/data/bsds500'
    for path in data.rglob('*'):
        if path.is_file():
            target=destination/path.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
            assert sha(target)==sha(path)
            files[str(path.relative_to(source))]=sha(target)
    pointer=json.loads((source/'SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/latest_checkpoint.json').read_text())
    original=Path(pointer['path']); target=destination.parent/'parent_checkpoint_003393.pt'; shutil.copy2(original,target)
    assert sha(target)==pointer['sha256']; pointer['path']=str(target)
    (destination.parent/'parent_checkpoint.json').write_text(json.dumps(pointer,indent=2)+'\n')
    test_pointer=destination/'SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/latest_checkpoint.json'
    test_pointer.parent.mkdir(parents=True,exist_ok=True)
    test_pointer.write_text(json.dumps(pointer,indent=2)+'\n')
    receipt=dict(source_root=str(source),runtime_root=str(destination),files=files,
        source_checkpoint=pointer,byte_identical=True,reason='launchd Desktop read denied by TCC; local explicit runtime copy, no permission modifications')
    (destination.parent/'staging_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(runtime_root=str(destination),verified_files=len(files),parent=pointer),indent=2))


if __name__=='__main__': main()
