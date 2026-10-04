"""Confirm the reviewed repository equals the actual local runtime bytes."""
import hashlib
import json
from pathlib import Path

repo=Path(__file__).resolve().parents[3]
base=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01')
runtime=base/'repo'
preparation=json.loads((base/'run/preparation.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
for path,expected in preparation['source_hashes'].items():
    target=Path(path); source=repo/target.relative_to(runtime)
    assert sha(source)==sha(target)==expected,str(source)
    checks[str(source.relative_to(repo))]=expected
initial=json.loads((base/'staging_receipt.json').read_text())
data_checks={}
for relative,expected in initial['files'].items():
    if relative.startswith('PreAttentiveVision/data/'):
        assert sha(repo/relative)==sha(runtime/relative)==expected,relative
        data_checks[relative]=expected
result=dict(verified=True,repository=str(repo),runtime=str(runtime),frozen_source_files=checks,
    data_files_verified=len(data_checks),manifest_sha256=preparation['source_manifest_sha256'],
    parent=preparation['parent'],note='Current frozen-source parity receipt supersedes initial build hashes for pre-profile source edits; original files unchanged')
(repo/'SecondPass/SpatialReadout/checks/runtime_parity.json').write_text(json.dumps(result,indent=2)+'\n')
(base/'run/runtime_parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(verified=True,source_files=len(checks),data_files=len(data_checks)),indent=2))
