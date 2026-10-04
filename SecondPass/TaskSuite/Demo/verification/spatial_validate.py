"""Static content/provenance checks only. Never imports a native renderer."""
from pathlib import Path
import ast
import hashlib
import json
import re

DEMO = Path(__file__).resolve().parents[1]
REPO = DEMO.parents[2]
EXPECTED = {
    'orientation_ring', 'orientation_cued', 'motion_duration_cued',
    'krauzlis_cued_motion', 'spatial_binding', 'image_recognition',
}
SECTIONS = {'appears', 'sequence', 'rules', 'ignore', 'implementation',
            'neuroscience', 'network', 'boundary', 'metrics', 'adaptation'}
PROPERTY_FIELDS = {'name', 'value', 'units', 'sampling', 'role', 'provenance', 'visibility'}
SOURCE_FIELDS = {'id', 'authors', 'title', 'year', 'url', 'doi', 'type',
                 'verification', 'support_location', 'supports',
                 'does_not_support', 'task_ids'}


def check():
    tasks = json.loads((DEMO / 'content/spatial_tasks.json').read_text())
    sources = json.loads((DEMO / 'content/spatial_sources.json').read_text())
    catalog = json.loads((DEMO.parent / 'catalog.json').read_text())
    by_id = {s['id']: s for s in sources}
    assert len(by_id) == len(sources)
    assert len(tasks) == len(EXPECTED)
    assert {t['id'] for t in tasks} == EXPECTED
    catalog_tasks = {t['id']: t for t in catalog['tasks']}
    citations = set()
    counts = {}
    for t in tasks:
        assert t['group'] in {'selection', 'retention'}
        assert set(t['sections']) == SECTIONS
        assert len(t['properties']) >= 20
        assert t['question'] and t['intro'] and t['equations']
        for name, paragraphs in t['sections'].items():
            assert paragraphs and all(isinstance(p, str) and len(p) >= 150 for p in paragraphs), (t['id'], name)
        for p in t['properties']:
            assert set(p) == PROPERTY_FIELDS
            assert all(isinstance(p[k], str) and p[k] for k in PROPERTY_FIELDS)
        assert len(t['references']) == len(set(t['references']))
        assert set(t['references']) <= by_id.keys()
        text = '\n'.join([t['intro']] + [p for ps in t['sections'].values() for p in ps])
        refs = set(re.findall(r'\[(spatial-[a-z0-9-]+)\]', text))
        assert refs <= set(t['references']), (t['id'], refs - set(t['references']))
        citations.update(refs)
        empirical = [by_id[s] for s in t['references'] if by_id[s]['type'].startswith('primary empirical')]
        assert empirical
        for s in t['references']:
            assert t['id'] in by_id[s]['task_ids'], (t['id'], s)
        assert t['worked_example']['metadata_fields'] and t['worked_example']['binding']
        counts[t['id']] = {
            'properties': len(t['properties']),
            'paragraphs': sum(map(len, t['sections'].values())),
            'prose_words': len(text.split()),
            'conditions': [c['id'] for c in catalog_tasks[t['id']]['conditions']],
            'primary_anchors': [s['id'] for s in empirical],
        }
    verified_quotes = 0
    source_hashes = {}
    for s in sources:
        assert SOURCE_FIELDS <= s.keys()
        assert s['id'].startswith('spatial-')
        assert s['supports'] and s['does_not_support'] and s['verification']
        if s.get('evidence_file'):
            evidence = (DEMO / s['evidence_file']).read_text()
            normalized = ' '.join(evidence.split())
            for excerpt in s.get('verified_excerpts', []):
                assert ' '.join(excerpt.split()) in normalized, s['id']
                verified_quotes += 1
        if s.get('repository_path'):
            path = REPO / s['repository_path']
            source_hashes[s['repository_path']] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert source_hashes[s['repository_path']] == s['sha256'], ('Source changed since initial inspection', path)
            ast.parse(path.read_text())
            assert s['url'] is None  # local provenance, not a broken portable-file hyperlink
    assert citations == set(by_id), ('Unused or uncited source', set(by_id) - citations)
    # Derived values used in the editorial descriptions; no stimulus execution.
    assert [b + 17 for b in (12, 20, 28)] == [29, 37, 45]
    assert 15 * .01 * 2.5 == .375
    assert 57 / (57 + 29 + 14) == .57
    for n in (4, 12, 24):
        for h in (3, 4, 5):
            for j in range(n):
                assert (n + 4 + h - 1) - (j + 1) == n + h + 2 - j
    # Import closure is inspected as AST text, never imported or executed.
    spatial = ast.parse((REPO / 'WorkingMemory/SpatialTaskBattery/stimuli.py').read_text())
    imports = {(node.module, alias.name) for node in ast.walk(spatial)
               if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert ('WorkingMemory.stimuli', 'visual_cues') in imports
    assert ('WorkingMemory.stimuli', '_motion_schedule') in imports
    assert ('WorkingMemory.stimuli', 'duration_oracle') in imports
    assert ('PreAttentiveVision.neuroscience_stimuli', 'DIRECTION_VECTORS') in imports
    assert ('PreAttentiveVision.natural_stimuli', 'DATA_ROOT') in imports
    return {
        'status': 'pass', 'task_count': len(tasks),
        'primary_cells': sum(len(c['conditions']) for c in counts.values()),
        'property_rows': sum(c['properties'] for c in counts.values()),
        'paragraphs': sum(c['paragraphs'] for c in counts.values()),
        'source_count': len(sources), 'verified_literal_excerpts': verified_quotes,
        'task_details': counts, 'source_sha256': source_hashes,
        'execution_boundary': 'stdlib static checks only; no render, model, checkpoint, stream restoration, training, runtime or cloud access',
        'limitations': [
            'Griffin/Nobre and Luck/Vogel verified at abstract depth only; full-text restrictions disclosed.',
            'Worked-example fields are native-metadata binding instructions; live selected-trial explanation belongs to parent UI.',
            'Photo redistribution permission not established; local delivery only.',
            'Pillow fit source inspected from installed library, not stimulus execution.',
        ],
    }


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
