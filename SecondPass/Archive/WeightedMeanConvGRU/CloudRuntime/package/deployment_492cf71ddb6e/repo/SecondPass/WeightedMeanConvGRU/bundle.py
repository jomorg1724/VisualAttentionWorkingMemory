"""Source-only deployment closure; no inherited checkpoints or credentials."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from . import worker as w

ROOT=w.ROOT

def dependency_sources():
    local=Path(__file__).parent
    sources={str(p.relative_to(ROOT)):p for p in local.iterdir() if p.suffix=='.py' or p.name in ('README.md','protocol.json')}
    # Deployment runtime is included without its package/status/artifact outputs.
    runtime=local/'CloudRuntime'
    if runtime.exists():
        for p in runtime.iterdir():
            if p.suffix=='.py' or p.name=='README.md': sources[str(p.relative_to(ROOT))]=p
    sources['SecondPass/TaskSuite/catalog.json']=ROOT/'SecondPass/TaskSuite/catalog.json'
    todo=list(sources); seen=set()
    while todo:
        rel=todo.pop()
        if rel in seen or not rel.endswith('.py'): continue
        seen.add(rel); package=list(Path(rel).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(sources[rel].read_text())):
            modules=[]
            if isinstance(node,ast.Import): modules=[n.name.split('.') for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                base+=node.module.split('.') if node.module else []
                modules=[base]+[base+n.name.split('.') for n in node.names if n.name!='*']
            for mod in modules:
                if not mod or mod[0] not in ('SecondPass','WorkingMemory','PreAttentiveVision'): continue
                paths=[Path(*mod).with_suffix('.py'),Path(*mod)/'__init__.py']+[Path(*mod[:i])/'__init__.py' for i in range(1,len(mod))]
                for path in paths:
                    key=str(path)
                    if key not in sources and (ROOT/path).is_file(): sources[key]=ROOT/path; todo.append(key)
    return sources

def build(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    sources=dependency_sources()
    version=hashlib.sha256(json.dumps({k:w.digest(v) for k,v in sorted(sources.items())},sort_keys=True).encode()).hexdigest()
    dest=output/('deployment_'+version[:12]); dest.mkdir(exist_ok=False)
    for rel,path in sources.items():
        target=dest/'repo'/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    (dest/'requirements.txt').write_text('torch==2.8.0\nnumpy>=2,<3\npillow>=10\nscipy>=1.11\npytest>=8\n')
    files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=w.digest(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    record=dict(protocol=w.PROTOCOL,source_version=version,files=files,initialization=w.provenance(),
        checkpoint_encoder=True,parameter_count=w.EXPECTED_PARAMETERS,parameter_tensors=w.EXPECTED_TENSORS,target_updates=w.TARGET,pool_size=1000,replay_epochs=10,
        checkpoints_included=False,secrets_included=False,budget_included=False,photos_included=False,
        data_policy='Versioned one-stimulus change task, no cue or second stimulus; exact29/37/45frames and original centers/dot dynamics retained; not the native cued benchmark',
        requirements_policy='PyTorch2.8 FP32 shared CNN and one standard nn.GRU; nonreentrant encoder checkpointing; no KDA/Triton dependency')
    w.atomic_json(dest/'deployment_manifest.json',record)
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(dest.rglob('*')):
            if p.is_file(): tf.add(p,arcname=str(p.relative_to(dest)),recursive=False)
    with tarfile.open(archive) as tf:
        assert len(tf.getmembers())==len(files)+1
        assert not any(Path(m.name).suffix in ('.pt','.pth','.ckpt','.safetensors','.key','.pem') for m in tf.getmembers())
        assert not any(part.startswith('.') for m in tf.getmembers() for part in Path(m.name).parts)
        for row in files: assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256']
    result=dict(deployment=str(dest),archive=str(archive),sha256=w.digest(archive),bytes=archive.stat().st_size,
        source_version=version,verified_files=len(files)+1,checkpoints_included=False,budget_included=False,photos_included=False)
    w.atomic_json(output/'bundle_receipt.json',result); print(json.dumps(result,indent=2)); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); build(p.parse_args().output)
