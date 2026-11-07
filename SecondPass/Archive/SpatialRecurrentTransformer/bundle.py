"""Minimal explicit-source deployment; no secrets, prior metrics or early checkpoint.

The known parent is digest-verified before deserialization. Archives are immutable
and every member is hash-read back. Run from the repository with --output PATH.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from . import worker as w

ROOT = w.ROOT
SOURCE = Path('/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/artifacts/terminal.pt')
SOURCE_SHA = 'e47031f3c66fefdb174384ad442688554966f3ce2cc86a9562e2ec99640ae72c'
BUDGET = SOURCE.parent/'budget.json'


def dependency_sources():
    sources = {str(p.relative_to(ROOT)):p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.md')}
    sources['SecondPass/TaskSuite/catalog.json'] = ROOT/'SecondPass/TaskSuite/catalog.json'
    todo=list(sources);seen=set()
    while todo:
        rel=todo.pop()
        if rel in seen or not rel.endswith('.py'):continue
        seen.add(rel);tree=ast.parse(sources[rel].read_text());package=list(Path(rel).with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name.split('.') for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                base+=node.module.split('.') if node.module else []
                modules=[base]+[base+x.name.split('.') for x in node.names if x.name!='*']
            for mod in modules:
                if not mod or mod[0] not in ('SecondPass','WorkingMemory','PreAttentiveVision'):continue
                options=[Path(*mod).with_suffix('.py'),Path(*mod)/'__init__.py']
                options += [Path(*mod[:i])/'__init__.py' for i in range(1,len(mod))]
                for path in options:
                    key=str(path)
                    if key not in sources and (ROOT/path).is_file():sources[key]=ROOT/path;todo.append(key)
    return sources


def build(output):
    if w.digest(SOURCE)!=SOURCE_SHA:raise ValueError('Production source SHA mismatch')
    source=w.cloud.trusted_load(SOURCE)
    oldroot=Path(source['state']['config']['runtime_root'])
    sources=dependency_sources()
    # Validate the unchanged scientific/native code against actual parent hashes.
    # Include their authorities too, avoiding recovery of any deleted source.
    for path,expected in source['state']['config']['source_hashes'].items():
        rel=str(Path(path).relative_to(oldroot));local=ROOT/rel
        if not local.is_file() or w.digest(local)!=expected:raise ValueError('Inherited source changed: '+rel)
        sources[rel]=local
    source_version=hashlib.sha256(json.dumps({k:w.digest(v) for k,v in sorted(sources.items())},sort_keys=True).encode()).hexdigest()
    dest=Path(output)/('deployment_'+source_version[:12]);dest.mkdir(parents=True,exist_ok=False)
    assets=dest/'assets';assets.mkdir()
    for rel,path in sources.items():
        target=dest/'repo'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    data=ROOT/'PreAttentiveVision/data/bsds500';manifest=json.loads((data/'manifest.json').read_text())
    if len(manifest['images'])!=500:raise ValueError('Require official complete BSDS500')
    for row in manifest['images']:
        if w.digest(data/row['file'])!=row['sha256']:raise ValueError('BSDS image hash mismatch')
    for rel in ['manifest.json']+[r['file'] for r in manifest['images']]:
        target=dest/'repo/PreAttentiveVision/data/bsds500'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(data/rel,target)
    shutil.copy2(SOURCE,assets/'terminal9360.pt')
    receipt=dict(schema=1,path='terminal9360.pt',sha256=SOURCE_SHA,bytes=SOURCE.stat().st_size,
        step=source['state']['step'],cumulative_step=source['state']['step']+source['state']['parent']['step'],
        scheduler_updates=source['scheduler']['updates'],fixture_only=False)
    w.atomic_json(assets/'source_pointer.json',receipt)
    w.source_payload(assets/'source_pointer.json')
    budget=json.loads(BUDGET.read_text());w.validate_budget(budget,budget)
    shutil.copy2(BUDGET,assets/'budget.json')
    (dest/'requirements.txt').write_text('torch==2.8.0\nnumpy>=2,<3\npillow>=10\nscipy>=1.11\npytest>=8\n')
    files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=w.digest(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    record=dict(protocol=w.VERSION,source_version=source_version,files=files,source_checkpoint=receipt,
        bsds500_images=500,secrets_included=False,old_metrics_included=False,budget_sha256=w.digest(BUDGET))
    w.atomic_json(dest/'deployment_manifest.json',record)
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(dest.rglob('*')):
            if p.is_file():tf.add(p,arcname=str(p.relative_to(dest)),recursive=False)
    with tarfile.open(archive,'r:gz') as tf:
        assert len(tf.getmembers())==len(files)+1
        for row in files:assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256'],row['path']
    result=dict(deployment=str(dest),archive=str(archive),sha256=w.digest(archive),bytes=archive.stat().st_size,
        source_version=source_version,verified_files=len(files)+1,source_checkpoint=receipt,budget=budget)
    w.atomic_json(Path(output)/'bundle_receipt.json',result)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)
