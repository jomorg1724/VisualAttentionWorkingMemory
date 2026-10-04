"""Read-only source inventory; never imports a renderer or training module."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
DEMO = Path(__file__).resolve().parent
ENTRY = ('SecondPass/TaskSuite/suite.py',)


def source_closure():
    pending = list(ENTRY)
    found = {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        blob = path.read_bytes()
        found[name] = hashlib.sha256(blob).hexdigest()
        tree = ast.parse(blob)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [node.module + '.' + a.name for a in node.names]
            else:
                continue
            for mod in modules:
                candidate = mod.replace('.', '/') + '.py'
                if (ROOT / candidate).is_file() and candidate not in found:
                    pending.append(candidate)
    return found


def snapshot():
    catalog_path = ROOT / 'SecondPass/TaskSuite/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    cells = [{'task_id': t['id'], 'condition_id': c['id'], 'kwargs': c['kwargs']} for t in catalog['tasks'] for c in t['conditions']]
    if len(catalog['tasks']) != 13 or len(cells) != 35 or len({(c['task_id'], c['condition_id']) for c in cells}) != 35:
        raise ValueError('Unexpected catalog inventory')
    sources = source_closure()
    if any(any(part in p.lower() for part in ('train.py', 'model.py', 'runtime', 'checkpoint')) for p in sources):
        raise ValueError('Renderer closure unexpectedly includes training/model/runtime code')
    return {'catalog_version': catalog['version'], 'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(), 'cells': cells, 'source_sha256': sources, 'dataset_manifest_sha256': hashlib.sha256((ROOT / 'PreAttentiveVision/data/bsds500/manifest.json').read_bytes()).hexdigest(), 'allowed_write_subtree': str(DEMO), 'boundary': 'CPU-only explanatory rendering; no model, training, runtime, cloud, evaluation or checkpoint access.'}


if __name__ == '__main__':
    target = DEMO / 'artifacts/source_inventory_before.json'
    current = snapshot()
    if target.exists():
        if json.loads(target.read_text()) != current:
            raise SystemExit('STOP: source/catalog/dataset inventory changed during atlas work')
        print('Source/catalog/dataset inventory unchanged.')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(current, indent=2) + '\n')
        print(json.dumps(current, indent=2))
