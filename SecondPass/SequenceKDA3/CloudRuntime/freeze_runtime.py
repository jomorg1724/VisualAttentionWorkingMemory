"""Offline allowlisted runtime freeze. Never includes test checkpoints or credentials."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
NAMES=('deploy.py','remote_owner.py','pod_guard.py','monitor.py','retrieve_once.py',
       'test_guard.py','test_runtime.py','test_deploy.py','test_mirror.py',
       'test_native_handshake.py','freeze_runtime.py','launch.sh','mirror_watch.py')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    # One-shot freeze, no silently updated immutable attempt.
    if (ROOT/'runtime_manifest.json').exists(): raise FileExistsError('Already frozen')
    bundle=json.loads((ROOT/'package/bundle_receipt.json').read_text())
    if sha(Path(bundle['archive']))!=bundle['sha256']: raise ValueError('Source archive receipt mismatch')
    control=Path('/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/control.py')
    manifest=dict(attempt='sequence_kda3_01',approved_archive_sha256=bundle['sha256'],
        authorization='Explicit NEW8h/$5; parent only rental; single A40<=0.49/h',
        source_hashes={n:sha(ROOT/n) for n in NAMES},
        inherited_source_hashes={str(control):sha(control)},
        checkpoints_included=False,credentials_included=False)
    (ROOT/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'runtime_sources.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for n in (*NAMES,'runtime_manifest.json'): tf.add(ROOT/n,arcname=n,recursive=False)
    receipt=dict(archive=str(archive),sha256=sha(archive),files=len(NAMES)+1,checkpoints_included=False)
    (ROOT/'runtime_bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for n in (*NAMES,'runtime_manifest.json','runtime_bundle_receipt.json','runtime_sources.tar.gz'):
        (ROOT/n).chmod(0o555 if n in ('deploy.py','launch.sh') else 0o444)
    bundle=json.loads((ROOT/'package/bundle_receipt.json').read_text())
    Path(bundle['archive']).chmod(0o444)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
