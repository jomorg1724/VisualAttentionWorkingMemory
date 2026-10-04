"""Coverage-contract regressions. No models, training or experiment reads."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

DEMO = Path(__file__).resolve().parents[1]
CATALOG = json.loads((DEMO.parent / 'catalog.json').read_text())


def validator():
    name = 'SecondPass.TaskSuite.Demo.validate_assets'
    assert importlib.util.find_spec(name) is not None, 'Atlas coverage validator is missing'
    return __import__(name, fromlist=['*'])


def inventory_fixture():
    return {'catalog': copy.deepcopy(CATALOG), 'cells': [
        dict(task_id=t['id'], condition_id=c['id'], kwargs=c['kwargs'])
        for t in CATALOG['tasks'] for c in t['conditions']]}


def test_inventory_rejects_missing_duplicate_extra_and_invented_ring_delay():
    v = validator()
    valid = inventory_fixture()
    assert v.validate_inventory(valid) == {'tasks': 13, 'cells': 35}
    variants = []
    missing = copy.deepcopy(valid); missing['cells'].pop(); variants.append(missing)
    duplicate = copy.deepcopy(valid); duplicate['cells'].append(duplicate['cells'][0]); variants.append(duplicate)
    extra = copy.deepcopy(valid); extra['cells'].append(dict(task_id='invented', condition_id='mixed', kwargs={})); variants.append(extra)
    ring = copy.deepcopy(valid)
    next(c for c in ring['cells'] if c['task_id'] == 'orientation_ring')['condition_id'] = 'D24'
    variants.append(ring)
    kwargs = copy.deepcopy(valid); kwargs['cells'][0]['kwargs'] = {'delay': 3}; variants.append(kwargs)
    catalog = copy.deepcopy(valid); catalog['catalog']['version'] = 'changed'; variants.append(catalog)
    for bad in variants:
        with pytest.raises(ValueError):
            v.validate_inventory(bad)


def test_explicit_native_requirements_and_coverage_do_not_trust_claimed_tags():
    v = validator()
    assert hasattr(v, 'coverage_map'), 'Native variant coverage map is missing'
    req = v.required_variants()
    ids = [r['id'] for r in req]
    assert len(ids) == len(set(ids))
    contrast = [r for r in req if r['attribute'] == 'contrast_pair']
    assert {tuple(r['value']) for r in contrast} == {(i,p) for i in (.025,.06,.13) for p in (.08,.18,.30)}
    assert all(r.get('condition_id') in (None,'D0') for r in req if r['task_id'] == 'orientation_ring')
    assert all(r['value'] == 0 for r in req if r['attribute'] == 'label' and r['task_id'] == 'image_recognition' and r['condition_id'].startswith('N0_'))
    assert {r['value'] for r in req if r['attribute'] == 'membership_position'} == {'early','middle','late'}
    assert {r['value'] for r in req if r['attribute'] == 'orientation_case'} == {'aligned','unchanged','opposite'}
    assert {r['value'] for r in req if r['attribute'] == 'event_magnitude'} == {26,28}
    empty = v.coverage_map([])
    assert set(empty['missing']) == set(ids)
    forged = dict(id='forged', task_id='contrast', condition_id='mixed', label=0,
                  metadata={'contrast_increment':.025,'pedestal':.08}, covered_variants=ids, gif='assets/gifs/forged.gif')
    covered = v.coverage_map([forged])
    assert any(r['episode_ids'] == ['forged'] for r in covered['requirements'] if r['attribute']=='contrast_pair' and r['value']==[.025,.08])
    assert any(r['id'] in covered['missing'] for r in covered['requirements'] if r['attribute']=='contrast_pair' and r['value']==[.13,.30])
    with pytest.raises(ValueError, match='coverage'):
        v.validate_coverage([forged])


def test_complete_native_bank_and_deliberate_manifest_corruptions(tmp_path):
    v = validator()
    assert hasattr(v, 'validate_manifest'), 'Comprehensive manifest validator is missing'
    from SecondPass.TaskSuite.Demo.export_assets import export_bank
    receipt = export_bank(tmp_path)
    assert receipt['complete'] and receipt['coverage_missing'] == []
    manifest = json.loads((tmp_path/'artifacts/manifest.json').read_text())
    result = v.validate_manifest(manifest,tmp_path,native=True)
    assert result['tasks']==13 and result['cells']==35
    assert result['episodes']==result['gifs']==receipt['episodes']
    assert result['native_equal_episodes']==receipt['episodes']
    assert result['source_immutable'] and result['coverage_missing']==[]
    mutations=[]
    duplicate=copy.deepcopy(manifest); duplicate['episodes'].append(duplicate['episodes'][0]); mutations.append(duplicate)
    orphan=copy.deepcopy(manifest); orphan['cells'][0]['episode_ids'].append('unknown'); mutations.append(orphan)
    missing=copy.deepcopy(manifest)
    # Remove every example for one contrast cross-product while preserving
    # both classes and the primary cell. A superficial cell check would pass.
    remove={e['id'] for e in missing['episodes'] if e['task_id']=='contrast' and e['metadata']['contrast_increment']==.025 and e['metadata']['pedestal']==.08}
    missing['episodes']=[e for e in missing['episodes'] if e['id'] not in remove]
    for c in missing['cells']:
        c['episode_ids']=[i for i in c['episode_ids'] if i not in remove]
        if c['showcase_id'] in remove: c['showcase_id']=c['episode_ids'][0]
    mutations.append(missing)
    source=copy.deepcopy(manifest); source['provenance']['dataset_manifest_sha256']='bad'; mutations.append(source)
    for bad in mutations:
        with pytest.raises(ValueError): v.validate_manifest(bad,tmp_path)
