"""CPU-only contract checks; these do not train a model."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
EXPECTED_TASKS = {
    'motion_direction', 'orientation', 'contrast', 'spatial_frequency',
    'chromatic_increment', 'contour', 'natural_spectrum', 'orientation_ring',
    'orientation_cued', 'motion_duration_cued', 'krauzlis_cued_motion',
    'spatial_binding', 'image_recognition',
}


def test_catalog_covers_discussed_tasks_without_authorizing_training():
    path = ROOT / 'catalog.json'
    assert path.is_file(), 'The unified task catalog has not been assembled'
    catalog = json.loads(path.read_text())
    tasks = catalog['tasks']
    assert {t['id'] for t in tasks} == EXPECTED_TASKS
    assert len(tasks) == len(EXPECTED_TASKS)
    assert catalog['training_authorized'] is False
    assert catalog['future_initialization'] == 'all_weights_fresh'
    assert sum(len(t['conditions']) for t in tasks) == 35
    for task in tasks:
        assert len(task['labels']) == task['classes']
        assert task['renderer'] and task['question'] and task['report_by']
        assert len({c['id'] for c in task['conditions']}) == len(task['conditions'])
    recognition = next(t for t in tasks if t['id'] == 'image_recognition')
    assert all(c['selection_eligible'] == (c['kwargs']['load'] > 0)
               for c in recognition['conditions'])


def test_adapter_preserves_native_pixels_labels_and_conditions():
    import importlib.util
    assert importlib.util.find_spec('SecondPass.TaskSuite.suite') is not None, 'Missing suite adapter'
    import torch
    from SecondPass.TaskSuite.suite import SuiteStream, CATALOG, task_classes
    from PreAttentiveVision.neuroscience_stimuli import TaskStream
    from WorkingMemory.PlainBaseline.variants import VariantStream
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream, frame_count

    torch.set_num_threads(2)
    stream = SuiteStream('val')
    assert task_classes() == {t['id']: t['classes'] for t in CATALOG['tasks']}
    for spec in CATALOG['tasks']:
        if spec['requires_bsds500']:
            continue
        task = spec['id']
        for cell in spec['conditions']:
            native_seed = stream.stream_seed(task, cell['id'])
            if spec['group'] == 'sensory':
                native = TaskStream(native_seed, 'val')
                expected = native.batch(4, task)
                length = 2
            elif task == 'orientation_ring':
                native = VariantStream(native_seed, 'val')
                expected = native.batch(4, task, cell['kwargs'])
                length = 4
            else:
                native = SpatialBatteryStream(native_seed, 'val')
                expected = native.batch(4, task, cell['kwargs'])
                length = frame_count(task, cell['kwargs'])
            x, y, meta = stream.batch(4, task, cell['id'])
            assert x.shape == (4, length, 3, 100, 100)
            assert torch.equal(x, expected[0]) and torch.equal(y, expected[1])
            assert all(all(m[k] == v for k, v in old.items()) for m, old in zip(meta, expected[2]))
            assert all(m['suite_cell'] == cell['id'] and m['suite_task'] == task for m in meta)
            assert x.dtype == torch.float32 and y.dtype == torch.int64
            assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
            assert y.min() >= 0 and y.max() < spec['classes']


def test_suite_replay_is_exact_and_other_cells_do_not_advance_stream():
    import torch
    from SecondPass.TaskSuite.suite import SuiteStream
    stream = SuiteStream('train')
    assert hasattr(stream, 'state_dict'), 'Suite-level replay state is missing'
    stream.batch(4, 'orientation', 'mixed')
    saved = json.loads(json.dumps(stream.state_dict()))
    expected = stream.batch(4, 'orientation', 'mixed')
    stream.batch(4, 'spatial_binding', 'D24')
    stream.load_state_dict(saved)
    stream.batch(4, 'contrast', 'mixed')
    actual = stream.batch(4, 'orientation', 'mixed')
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1]) and expected[2] == actual[2]
    # A restored stream must discard any cells created after the snapshot.
    fresh = SuiteStream('train').batch(4, 'spatial_binding', 'D24')
    resumed = stream.batch(4, 'spatial_binding', 'D24')
    assert torch.equal(fresh[0], resumed[0]) and fresh[2] == resumed[2]


def test_invalid_state_and_requests_fail_without_changing_progress():
    import pytest
    from SecondPass.TaskSuite.suite import SuiteStream
    stream = SuiteStream('val')
    assert hasattr(stream, 'state_dict'), 'Suite-level replay state is missing'
    stream.batch(4, 'contrast', 'mixed')
    before = stream.state_dict()
    for args in [(0, 'contrast', 'mixed'), (True, 'contrast', 'mixed'),
                 (4, 'not_a_task', 'mixed'), (4, 'orientation_ring', 'D24')]:
        with pytest.raises(ValueError):
            stream.batch(*args)
    with pytest.raises(ValueError):
        SuiteStream('unknown')
    with pytest.raises(ValueError):
        stream.load_state_dict(SuiteStream('test').state_dict())
    bad = json.loads(json.dumps(before))
    bad['catalog']['version'] = 'incompatible'
    with pytest.raises(ValueError):
        stream.load_state_dict(bad)
    assert json.dumps(stream.state_dict(), sort_keys=True) == json.dumps(before, sort_keys=True)


def test_cpu_verifier_accounts_for_every_cell_and_never_trains(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('SecondPass.TaskSuite.verify') is not None, 'Missing CPU verification command'
    from SecondPass.TaskSuite.verify import verify
    result = verify(tmp_path, model_smoke=True)
    assert result['training_updates'] == 0 and result['device'] == 'cpu'
    assert result['expected_tasks'] == 13 and result['expected_cells'] == 35
    assert len(result['cells']) == 35
    assert len({(c['task'], c['cell']) for c in result['cells']}) == 35
    for cell in result['cells']:
        assert cell['status'] in ('passed', 'blocked_missing_dataset')
        if cell['status'] == 'passed':
            assert cell['exact_replay'] and cell['native_equivalence']
            assert cell['frames'] >= 2
    assert result['model']['parameters_unchanged']
    assert result['model']['all_parameters_trainable']
    assert result['model']['heads'] == sorted(EXPECTED_TASKS)
    assert json.loads((tmp_path / 'verification.json').read_text()) == result


def test_event_mixture_and_label_semantics_are_preserved():
    from collections import Counter
    from SecondPass.TaskSuite.suite import SuiteStream
    stream = SuiteStream('val')
    events = Counter()
    for _ in range(25):
        _, labels, metadata = stream.batch(4, 'krauzlis_cued_motion', 'B12')
        for label, row in zip(labels.tolist(), metadata):
            events[row['event_type']] += 1
            assert label == int(row['event_type'] == 'target')
    assert events == dict(target=57, foil=29, catch=14)
    for task in ('orientation_ring', 'orientation_cued', 'spatial_binding', 'motion_duration_cued'):
        _, labels, metadata = stream.batch(16, task, 'D0')
        for label, row in zip(labels.tolist(), metadata):
            target = row['target_location']
            if task.startswith('orientation'):
                assert label == int(row['rotations_degrees'][target] * row['cue_sign'] > 0)
            elif task == 'spatial_binding':
                assert len(row['swapped_locations']) == 2
                assert label == int(target in row['swapped_locations'])
                assert row['cue_frames'] == [3] and row['sample_frames'] == [1, 2]
            else:
                counts = row['duration_counts_by_patch'][target]
                assert sum(counts) == 8 and counts.count(max(counts)) == 1
                assert label == counts.index(max(counts))


def test_split_streams_are_distinct_and_photo_membership_is_real():
    import pytest
    import torch
    from PreAttentiveVision.natural_stimuli import DATA_ROOT
    from SecondPass.TaskSuite.suite import SuiteStream, CATALOG
    streams = [SuiteStream(split) for split in ('train', 'val', 'test')]
    seeds = [s.stream_seed(t['id'], c['id']) for s in streams
             for t in CATALOG['tasks'] for c in t['conditions']]
    assert len(set(seeds)) == len(seeds)
    if not (DATA_ROOT / 'manifest.json').is_file():
        pytest.skip('Official BSDS500 not prepared; verifier reports blocked cells')
    source_ids = []
    for stream in streams:
        ids = set()
        for task, cell in [('natural_spectrum', 'mixed'), ('image_recognition', 'N24_H5')]:
            before = stream.state_dict()
            x, y, meta = stream.batch(8, task, cell)
            stream.load_state_dict(before)
            xx, yy, mm = stream.batch(8, task, cell)
            assert torch.equal(x, xx) and torch.equal(y, yy) and meta == mm
            for row in meta:
                if task == 'natural_spectrum':
                    ids.add(row['base_id'])
                else:
                    ids.update(row['study_source_ids'])
                    ids.add(row['probe_source_id'])
        source_ids.append(ids)
    assert all(not a.intersection(b) for i, a in enumerate(source_ids) for b in source_ids[i+1:])


def test_photo_resume_rejects_changed_manifest_without_advancing_state(tmp_path, monkeypatch):
    import pytest
    from PreAttentiveVision.natural_stimuli import DATA_ROOT
    from SecondPass.TaskSuite import suite
    if not (DATA_ROOT / 'manifest.json').is_file():
        pytest.skip('Official BSDS500 not prepared')
    stream = suite.SuiteStream('val')
    stream.batch(4, 'image_recognition', 'N4_H3')
    saved = stream.state_dict()
    assert saved.get('dataset_manifest_sha256'), 'Recognition resume lacks dataset identity'
    # Alter only an isolated manifest copy, never the real dataset.
    path = tmp_path / 'manifest.json'
    path.write_bytes((DATA_ROOT / 'manifest.json').read_bytes() + b'\n')
    monkeypatch.setattr(suite, 'DATA_ROOT', tmp_path)
    with pytest.raises(ValueError, match='manifest'):
        stream.load_state_dict(saved)
    monkeypatch.setattr(suite, 'DATA_ROOT', DATA_ROOT)
    assert stream.state_dict() == saved
