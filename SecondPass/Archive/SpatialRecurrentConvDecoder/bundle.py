"""Dependency-closed source + original BSDS500 only; no checkpoint/budget input."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from . import worker as w

ROOT=w.ROOT


def dependency_sources():
    sources={str(p.relative_to(ROOT)):p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.md')}
    sources['SecondPass/TaskSuite/catalog.json']=ROOT/'SecondPass/TaskSuite/catalog.json'
    todo=list(sources); seen=set()
    while todo:
        rel=todo.pop()
        if rel in seen or not rel.endswith('.py'): continue
        seen.add(rel); tree=ast.parse(sources[rel].read_text()); package=list(Path(rel).with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import): modules=[x.name.split('.') for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                base+=node.module.split('.') if node.module else []
                modules=[base]+[base+x.name.split('.') for x in node.names if x.name!='*']
            for mod in modules:
                if not mod or mod[0] not in ('SecondPass','WorkingMemory','PreAttentiveVision'): continue
                options=[Path(*mod).with_suffix('.py'),Path(*mod)/'__init__.py']
                options += [Path(*mod[:i])/'__init__.py' for i in range(1,len(mod))]
                for path in options:
                    key=str(path)
                    if key not in sources and (ROOT/path).is_file(): sources[key]=ROOT/path; todo.append(key)
    return sources


def build(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    sources=dependency_sources()
    source_version=hashlib.sha256(json.dumps({k:w.digest(v) for k,v in sorted(sources.items())},sort_keys=True).encode()).hexdigest()
    dest=output/('deployment_'+source_version[:12]); dest.mkdir(exist_ok=False)
    for rel,path in sources.items():
        target=dest/'repo'/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    data=ROOT/'PreAttentiveVision/data/bsds500'; manifest=json.loads((data/'manifest.json').read_text())
    counts=dict(Counter(r['split'] for r in manifest['images']))
    if len(manifest['images'])!=500 or counts!={'train':200,'val':100,'test':200}: raise ValueError('Require original BSDS500200/100/200')
    if len({r['base_id'] for r in manifest['images']})!=500: raise ValueError('Duplicate BSDS identity')
    for row in manifest['images']:
        if w.digest(data/row['file'])!=row['sha256']: raise ValueError('BSDS image hash mismatch: '+row['file'])
    for rel in ['manifest.json']+[r['file'] for r in manifest['images']]:
        target=dest/'repo/PreAttentiveVision/data/bsds500'/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(data/rel,target)
    (dest/'assets').mkdir()
    (dest/'requirements.txt').write_text('torch==2.8.0\nnumpy>=2,<3\npillow>=10\nscipy>=1.11\npytest>=8\n')
    files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=w.digest(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert not any(Path(r['path']).suffix in ('.pt','.pth','.ckpt','.safetensors') for r in files)
    record=dict(protocol=w.PROTOCOL,source_version=source_version,files=files,initialization=w.provenance(),
                bsds500_images=500,split_counts=counts,secrets_included=False,checkpoints_included=False,
                old_metrics_included=False,budget_included=False,budget_policy='Parent injects assets/budget.json at actual new pod creation')
    w.atomic_json(dest/'deployment_manifest.json',record)
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(dest.rglob('*')):
            if p.is_file(): tf.add(p,arcname=str(p.relative_to(dest)),recursive=False)
    with tarfile.open(archive,'r:gz') as tf:
        assert len(tf.getmembers())==len(files)+1
        assert not any(Path(m.name).suffix in ('.pt','.pth','.ckpt','.safetensors') for m in tf.getmembers())
        for row in files: assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256'],row['path']
        assert hashlib.sha256(tf.extractfile('deployment_manifest.json').read()).hexdigest()==w.digest(dest/'deployment_manifest.json')
    result=dict(deployment=str(dest),archive=str(archive),sha256=w.digest(archive),bytes=archive.stat().st_size,
                source_version=source_version,verified_files=len(files)+1,bsds500_images=500,split_counts=counts,
                checkpoints_included=False,budget_included=False)
    w.atomic_json(output/'bundle_receipt.json',result)
    print(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)
