"""Portable static atlas build; no numerical/model/renderer imports."""
from pathlib import Path
import json
import re
import zipfile
import hashlib
import shutil


def safe_asset(root, name):
    if not isinstance(name, str) or not name or '://' in name or Path(name).is_absolute():
        raise ValueError(f'Not a local relative asset: {name}')
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f'Missing or out-of-scope asset: {name}')
    return path


def write_data_script(path, data):
    text = json.dumps(data, ensure_ascii=True, separators=(',', ':')).replace('</', '<\\/')
    Path(path).write_text('window.ATLAS = ' + text + ';\n', encoding='utf-8')


def portable_zip(site, target):
    site = Path(site)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(site.rglob('*')):
            if path.is_file() and not any(part.startswith('.') for part in path.relative_to(site).parts):
                if path.is_symlink():
                    raise ValueError(f'Symlink forbidden in delivery: {path}')
                info = zipfile.ZipInfo('task-suite-atlas/' + path.relative_to(site).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, path.read_bytes())

DEMO = Path(__file__).resolve().parent
SECTIONS = ('appears', 'sequence', 'rules', 'ignore', 'implementation', 'neuroscience', 'network', 'boundary', 'metrics', 'adaptation')


def validate_content(catalog, tasks, sources):
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task content')
    if set(ids) != {t['id'] for t in catalog['tasks']}:
        raise ValueError('Content task inventory differs from catalog')
    refs = {s['id'] for s in sources}
    if len(refs) != len(sources):
        raise ValueError('duplicate source IDs')
    for task in tasks:
        for key in ('title', 'group', 'question', 'intro', 'properties', 'references'):
            if not task.get(key):
                raise ValueError(f'{task["id"]}: missing {key}')
        for section in SECTIONS:
            paragraphs = task.get('sections', {}).get(section)
            if not isinstance(paragraphs, list) or not paragraphs or not all(isinstance(p, str) and p.strip() for p in paragraphs):
                raise ValueError(f'{task["id"]}: missing/invalid {section}')
        for ref in task['references']:
            if ref not in refs:
                raise ValueError(f'{task["id"]}: unresolved reference {ref}')
        for paragraphs in task['sections'].values():
            for text in paragraphs:
                for ref in re.findall(r'(?<![\w])\[([a-zA-Z][a-zA-Z0-9_-]+)\]', text):
                    if ref not in refs:
                        raise ValueError(f'{task["id"]}: unresolved citation {ref}')
        for prop in task['properties']:
            for key in ('name', 'value', 'units', 'sampling', 'role', 'provenance', 'visibility'):
                if key not in prop or prop[key] is None:
                    raise ValueError(f'{task["id"]}: missing property {key}')
    for source in sources:
        for key in ('authors', 'title', 'year', 'url', 'type', 'verification', 'support_location', 'supports', 'does_not_support', 'task_ids'):
            if key in ('url', 'year') and source.get('repository_path'):
                continue
            if key not in source or source[key] is None:
                raise ValueError(f'{source["id"]}: missing source {key}')


def check_native_coverage(manifest, coverage):
    import importlib.util
    spec = importlib.util.spec_from_file_location('atlas_coverage', DEMO / 'validate_assets.py')
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validator.validate_inventory(manifest)
    actual = validator.validate_coverage(manifest['episodes'])
    if actual != coverage:
        raise ValueError('Recorded coverage differs from native-metadata-derived coverage')


def write_bibliography(path, sources):
    import html
    def text(value):
        if isinstance(value, list):
            return '<ul>' + ''.join('<li>' + text(v) + '</li>' for v in value) + '</ul>'
        if isinstance(value, dict):
            return '<dl>' + ''.join('<dt>' + html.escape(str(k)) + '</dt><dd>' + text(v) + '</dd>' for k,v in value.items()) + '</dl>'
        return html.escape(str(value or 'Not applicable'))
    articles = []
    for s in sources:
        link = ('<a href="' + html.escape(s['url'], quote=True) + '" target="_blank" rel="noopener">Open external source ↗</a>') if s.get('url') else '<p>Repository source: ' + text(s.get('repository_path')) + '</p>'
        sections = ''.join('<h3>' + title + '</h3><div>' + text(s.get(key)) + '</div>' for key,title in [('verification','What was verified'), ('support_location','Supporting section / figure'), ('supports','Claims supported'), ('does_not_support','Not inherited by this assay')])
        tasks = ' · '.join('<a href="index.html#task=' + html.escape(t,quote=True) + '">' + html.escape(t.replace('_',' ')) + '</a>' for t in s['task_ids'])
        articles.append('<article class="reference" id="ref-' + html.escape(s['id'],quote=True) + '"><p class="eyebrow">' + text(s['type']) + '</p><h2>' + text(s['title']) + '</h2><p>' + text(s['authors']) + ' · ' + text(s.get('year')) + '</p>' + link + '<p>Used for: ' + tasks + '</p>' + sections + '</article>')
    Path(path).write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sources & claim boundaries · Visual Task Atlas</title><link rel="icon" href="vendor/atlas-mark.svg"><link rel="stylesheet" href="vendor/fonts.css"><link rel="stylesheet" href="atlas.css"><style>main{max-width:1000px;padding-top:40px}.reference{padding:34px 0;border-bottom:1px solid var(--line)}.reference h2{font-size:29px;margin:12px 0}.reference h3{font:600 17px var(--sans);margin:20px 0 8px}li{margin-bottom:8px}.reference>p{margin:12px 0}</style><main><a href="index.html">← Back to the atlas</a><header class="page-intro"><h1>Sources & claim boundaries</h1><p>Complete reference ledger. Full-text, abstract-only, metadata-only and local-code evidence are distinguished below. A related paradigm motivates a project-specific assay; it does not validate every implementation choice.</p><a href="sources.json" download>Download source ledger JSON ↓</a></header>' + ''.join(articles) + '</main></html>', encoding='utf-8')


def build(root=DEMO):
    root = Path(root)
    load = lambda p: json.loads((root / p).read_text())
    manifest = load('artifacts/manifest.json')
    coverage = load('artifacts/coverage.json')
    if root.resolve() == DEMO.resolve():
        check_native_coverage(manifest, coverage)
    if coverage.get('missing') or coverage.get('missing_requirements'):
        raise ValueError('Incomplete native variant coverage')
    tasks = load('content/sensory_tasks.json') + load('content/spatial_tasks.json')
    sources = load('content/sensory_sources.json') + load('content/spatial_sources.json')
    # Identical shared records may be merged; conflicting IDs must fail.
    merged = {}
    for s in sources:
        if s['id'] in merged and merged[s['id']] != s:
            raise ValueError(f'Conflicting shared source ID {s["id"]}')
        merged[s['id']] = s
    sources = list(merged.values())
    catalog = manifest['catalog']
    validate_content(catalog, tasks, sources)
    tasks = sorted(tasks, key=lambda t: [r['id'] for r in catalog['tasks']].index(t['id']))
    expected = {(t['id'], c['id']): c['kwargs'] for t in catalog['tasks'] for c in t['conditions']}
    actual = {(c['task_id'], c['condition_id']): c['kwargs'] for c in manifest['cells']}
    if expected != actual or len(actual) != len(manifest['cells']):
        raise ValueError('Missing, duplicate or extra primary cells')
    episodes = manifest['episodes']
    by_id = {e['id']: e for e in episodes}
    if len(by_id) != len(episodes):
        raise ValueError('Duplicate episodes')
    for cell in manifest['cells']:
        if not cell['episode_ids'] or cell['showcase_id'] not in cell['episode_ids']:
            raise ValueError('Cell lacks an episode/showcase')
        for eid in cell['episode_ids']:
            e = by_id.get(eid)
            if not e or (e['task_id'], e['condition_id']) != (cell['task_id'], cell['condition_id']):
                raise ValueError(f'Invalid cell episode {eid}')
    dist = root / 'dist'
    for e in episodes:
        for asset in e['frames'] + [e[k] for k in ('gif', 'poster', 'frame_zip', 'metadata_path')]:
            safe_asset(dist, asset)
    for name in ('index.html', 'atlas.css', 'atlas.js'):
        shutil.copyfile(root / 'web' / name, dist / name)
    shutil.copytree(root / 'web/vendor', dist / 'vendor', dirs_exist_ok=True)
    for name, data in [('tasks', tasks), ('sources', sources)]:
        (root / 'content' / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    write_bibliography(dist / 'bibliography.html', sources)
    write_data_script(dist / 'data.js', {'manifest': manifest, 'tasks': tasks, 'sources': sources, 'coverage': coverage})
    shutil.copyfile(root / 'README.md', dist / 'README.md')
    for name in ('manifest.json', 'coverage.json'):
        shutil.copyfile(root / 'artifacts' / name, dist / name)
    for name in ('tasks.json', 'sources.json'):
        shutil.copyfile(root / 'content' / name, dist / name)
    files = {p.relative_to(dist).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dist.rglob('*')) if p.is_file() and p.name != 'build_receipt.json'}
    receipt = {'task_count': len(tasks), 'primary_cell_count': len(actual), 'episode_count': len(episodes), 'gif_count': len({e['gif'] for e in episodes}), 'native_frame_count': sum(len(e['frames']) for e in episodes), 'source_count': len(sources), 'file_count_excluding_receipt': len(files), 'file_bytes_excluding_receipt': sum((dist / p).stat().st_size for p in files), 'files_sha256': files, 'bundle_scope': 'Portable local explanatory atlas only; no model outputs, raw float development arrays, source photographs, credentials or runtime artifacts.', 'determinism': 'Content-hashed receipt without volatile timestamps; sorted ZIP members with fixed timestamps.'}
    (root / 'artifacts/build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.copyfile(root / 'artifacts/build_receipt.json', dist / 'build_receipt.json')
    portable_zip(dist, root / 'task-suite-atlas.zip')
    return {k:v for k,v in receipt.items() if k != 'files_sha256'}


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
