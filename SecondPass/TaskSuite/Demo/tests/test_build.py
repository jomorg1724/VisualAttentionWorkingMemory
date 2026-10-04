"""Build contracts: pure stdlib checks, no renderer/model imports."""
import importlib.util
from pathlib import Path
import json
import pytest

DEMO = Path(__file__).resolve().parents[1]


def builder():
    path = DEMO / 'build_site.py'
    assert path.is_file(), 'Portable builder has not been implemented'
    spec = importlib.util.spec_from_file_location('atlas_build', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def valid_content():
    catalog = {'tasks': [{'id': 'test_task'}]}
    sources = [{'id': 'primary', 'authors': 'Test author', 'title': 'Fixture only', 'year': 2000, 'url': 'https://example.org/test', 'type': 'primary', 'verification': 'test fixture', 'support_location': 'test', 'supports': 'test', 'does_not_support': 'test', 'task_ids': ['test_task']}]
    sections = {k: ['Connected explanation for a test fixture, not published content. [primary]'] for k in ('appears','sequence','rules','ignore','implementation','neuroscience','network','boundary','metrics','adaptation')}
    tasks = [{'id': 'test_task', 'title': 'Test task', 'group': 'sensory', 'question': 'Test?', 'intro': 'This is a test fixture.', 'sections': sections, 'equations': [], 'references': ['primary'], 'properties': [{'name': 'raster', 'value': '100', 'units': 'pixels', 'sampling': 'fixed', 'role': 'input', 'provenance': 'fixture.py:1', 'visibility': 'model-visible'}]}]
    return catalog, tasks, sources


def test_complete_content_and_exact_inventory():
    b = builder()
    catalog, tasks, sources = valid_content()
    assert b.validate_content(catalog, tasks, sources) is None
    with pytest.raises(ValueError, match='inventory'):
        b.validate_content(catalog, [], sources)
    with pytest.raises(ValueError, match='duplicate'):
        b.validate_content(catalog, tasks + tasks, sources)


def test_missing_sections_and_references_rejected():
    b = builder()
    catalog, tasks, sources = valid_content()
    tasks[0]['sections'].pop('neuroscience')
    with pytest.raises(ValueError, match='neuroscience'):
        b.validate_content(catalog, tasks, sources)
    catalog, tasks, sources = valid_content()
    with pytest.raises(ValueError, match='reference'):
        b.validate_content(catalog, tasks, [])
    tasks[0]['sections']['rules'] = ['Rule [unknown-citation].']
    with pytest.raises(ValueError, match='citation'):
        b.validate_content(catalog, tasks, sources)


def test_production_coverage_is_rederived_not_trusted():
    b = builder()
    manifest = json.loads((DEMO / 'artifacts/manifest.json').read_text())
    coverage = json.loads((DEMO / 'artifacts/coverage.json').read_text())
    assert b.check_native_coverage(manifest, coverage) is None
    coverage['required_count'] = 1
    with pytest.raises(ValueError, match='coverage'):
        b.check_native_coverage(manifest, coverage)


def test_code_index_not_citation_and_local_provenance_allowed():
    b = builder()
    catalog, tasks, sources = valid_content()
    tasks[0]['sections']['rules'] = ['Use rotations_degrees[target_location], not a citation. [primary]']
    sources[0].update(url=None, year=None, repository_path='WorkingMemory/stimuli.py', type='local executable source')
    assert b.validate_content(catalog, tasks, sources) is None


def test_assets_are_local_contained_and_exist(tmp_path):
    b = builder()
    p = tmp_path / 'assets/frame.png'
    p.parent.mkdir()
    p.write_bytes(b'fixture')
    assert b.safe_asset(tmp_path, 'assets/frame.png') == p
    for bad in ('../outside', '/etc/passwd', 'https://example.org/asset.png', 'assets/missing.png'):
        with pytest.raises(ValueError):
            b.safe_asset(tmp_path, bad)


def test_data_script_is_direct_file_compatible(tmp_path):
    b = builder()
    data = {'text': '</script>\u2028\u2029', 'tasks': []}
    path = tmp_path / 'data.js'
    b.write_data_script(path, data)
    script = path.read_text()
    assert script.startswith('window.ATLAS = ')
    assert '</script>' not in script
    assert json.loads(script.removeprefix('window.ATLAS = ').strip().removesuffix(';')) == data


def test_zip_reproducible_and_relocated(tmp_path):
    import zipfile
    b = builder()
    site = tmp_path / 'dist'
    site.mkdir()
    (site / 'index.html').write_text('<!doctype html><title>Fixture only</title>')
    (site / 'data.js').write_text('window.ATLAS = {};')
    archive = tmp_path / 'one.zip'
    second = tmp_path / 'two.zip'
    b.portable_zip(site, archive)
    b.portable_zip(site, second)
    assert archive.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ['task-suite-atlas/data.js', 'task-suite-atlas/index.html']
        z.extractall(tmp_path / 'relocated')
    assert (tmp_path / 'relocated/task-suite-atlas/index.html').read_bytes() == (site / 'index.html').read_bytes()


def test_build_copies_assets_and_merges_direct_data(tmp_path):
    b = builder()
    catalog, tasks, sources = valid_content()
    for folder in ('web/vendor','dist/assets','content','artifacts'):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for name, text in {'index.html':'<!doctype html><script src="data.js"></script><script src="atlas.js"></script>', 'atlas.css':'body {color: black}', 'atlas.js':'console.log(window.ATLAS.tasks.length)'}.items():
        (tmp_path / 'web' / name).write_text(text)
    (tmp_path / 'web/vendor/fonts.css').write_text('/* local fixture */')
    for kind, rows in [('sensory_tasks', tasks), ('spatial_tasks', []), ('sensory_sources', sources), ('spatial_sources', [])]:
        (tmp_path / 'content' / f'{kind}.json').write_text(json.dumps(rows))
    (tmp_path / 'README.md').write_text('Fixture only.')
    assets = ['assets/f.png','assets/e.gif','assets/frames.zip','assets/metadata.json']
    for a in assets:
        (tmp_path / 'dist' / a).write_text('fixture only')
    catalog['tasks'][0]['conditions'] = [{'id':'mixed', 'kwargs':{}}]
    manifest = {'catalog':catalog, 'cells':[{'task_id':'test_task','condition_id':'mixed','kwargs':{},'episode_ids':['e1'],'showcase_id':'e1'}], 'episodes':[{'id':'e1','task_id':'test_task','condition_id':'mixed','frames':[assets[0]],'gif':assets[1],'poster':assets[0],'frame_zip':assets[2],'metadata_path':assets[3]}]}
    (tmp_path / 'artifacts/manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'artifacts/coverage.json').write_text(json.dumps({'missing':[]}))
    receipt = b.build(tmp_path)
    assert receipt['task_count'] == 1 and receipt['primary_cell_count'] == 1 and receipt['gif_count'] == 1
    assert (tmp_path / 'dist/data.js').read_text().startswith('window.ATLAS = ')
    assert (tmp_path / 'dist/vendor/fonts.css').is_file()
    bibliography = (tmp_path / 'dist/bibliography.html').read_text()
    assert 'Fixture only' in bibliography and 'ref-primary' in bibliography
    assert (tmp_path / 'task-suite-atlas.zip').is_file()
    (tmp_path / 'dist/assets/e.gif').unlink()
    with pytest.raises(ValueError, match='asset'):
        b.build(tmp_path)
