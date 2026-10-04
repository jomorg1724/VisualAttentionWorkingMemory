"""Build a minimal hash-verified deployment without repository secrets/history."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile
import torch

ROOT=Path(__file__).resolve().parents[3]
RUNTIME=Path('/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_01')
SOURCE=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt')
OLD=Path('/Users/jonathanmorgan/VAWMRuntime/spatial_comparison_01/run')
EXPECTED='1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    assert digest(SOURCE)==EXPECTED
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    dest=RUNTIME/'deployment';dest.mkdir(exist_ok=False)
    repo=dest/'repo';assets=dest/'assets';assets.mkdir()
    native=source['state']['config'];oldroot=Path(native['runtime_root']);sources={};fallback=[]
    for name,expected in native['source_hashes'].items():
        rel=Path(name).relative_to(oldroot);p=ROOT/rel
        if not p.is_file() or digest(p)!=expected:
            p=Path(name);fallback.append(str(rel))
        assert digest(p)==expected,(p,expected)
        sources[rel]=p
    for p in (ROOT/'SecondPass/SpatialComparisonReadout').glob('*'):
        if p.suffix in ('.py','.md'):sources[p.relative_to(ROOT)]=p
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):sources[p.relative_to(ROOT)]=p
    # Resolve only local Python imports, including relative imports and parents.
    todo=list(sources);seen=set()
    while todo:
        rel=todo.pop()
        if rel in seen or rel.suffix!='.py':continue
        seen.add(rel)
        tree=ast.parse(sources[rel].read_text())
        package=list(rel.with_suffix('').parts[:-1])
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
                    if path not in sources and (ROOT/path).is_file():sources[path]=ROOT/path;todo.append(path)
    for rel,p in sources.items():
        target=repo/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    data=ROOT/'PreAttentiveVision/data/bsds500';manifest=json.loads((data/'manifest.json').read_text())
    assert len(manifest['images'])==500
    for rel in ['manifest.json']+[x['file'] for x in manifest['images']]:
        target=repo/'PreAttentiveVision/data/bsds500'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(data/rel,target)
    for row in manifest['images']:assert digest(data/row['file'])==row['sha256']
    shutil.copy2(SOURCE,assets/'terminal6760.pt')
    shutil.copy2(OLD/'baseline_validation.json',assets/'baseline_validation.json')
    shutil.copy2(OLD/'profile_candidate/migration.pt',assets/'prior_candidate_migration.pt')
    rows=[json.loads(x) for x in (SOURCE.parent/'progress.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1691,6761))
    keys=sorted({(r['task'],r['cell']) for r in rows})
    costs=dict(prior_updates=len(rows),source_progress_sha256=digest(SOURCE.parent/'progress.jsonl'),
        cells=[dict(task=t,cell=c,seconds=statistics.mean(r['seconds'] for r in rows if (r['task'],r['cell'])==(t,c))) for t,c in keys])
    (assets/'historical_costs.json').write_text(json.dumps(costs,indent=2)+'\n')
    files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert all(not any(x in f['path'] for x in ('.env','ssh_key','.git/')) for f in files)
    record=dict(protocol='spatial_comparison_cloud_candidate_v1',files=files,source_sha256=EXPECTED,
                source_checkpoint_step=source['state']['step'],source_scheduler_updates=source['scheduler']['updates'],
                archived_native_fallbacks=fallback,actual_bsds500_images=len(manifest['images']),secrets_included=False)
    (dest/'deployment_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    output=RUNTIME/'deployment.tar.gz'
    with tarfile.open(output,'w:gz') as tf:
        for p in sorted(dest.rglob('*')):
            if p.is_file():tf.add(p,arcname=str(p.relative_to(dest)),recursive=False)
    with tarfile.open(output) as tf:
        assert len(tf.getmembers())==len(files)+1
        for row in files:assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256']
    print(json.dumps(dict(bundle=str(output),bytes=output.stat().st_size,sha256=digest(output),file_count=len(files)+1,
                          archived_native_fallbacks=fallback),indent=2))


if __name__=='__main__':build()
