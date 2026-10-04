"""Native-only exporter tests. Run sequentially with Demo-local basetemp."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import copy
import importlib.util
import json
from pathlib import Path
import pytest

DEMO = Path(__file__).resolve().parents[1]


def exporter():
    name = 'SecondPass.TaskSuite.Demo.export_assets'
    assert importlib.util.find_spec(name) is not None, 'Native atlas exporter is missing'
    return __import__(name, fromlist=['*'])


@pytest.mark.parametrize('task,cell', [('orientation','mixed'), ('motion_direction','mixed'),
    ('orientation_ring','D0'), ('orientation_cued','D24'), ('natural_spectrum','mixed'),
    ('image_recognition','N24_H5')])
def test_native_equality_and_isolated_json_resume(task, cell):
    e = exporter()
    import numpy as np
    from PreAttentiveVision.neuroscience_stimuli import TaskStream
    from WorkingMemory.PlainBaseline.variants import VariantStream
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    before = e.source_hashes()
    wrapper = e.DemoStream(task, cell)
    cls = TaskStream if e.TASKS[task]['group'] == 'sensory' else VariantStream if task == 'orientation_ring' else SpatialBatteryStream
    native = cls(wrapper.seed, 'train')
    for _ in range(2):
        raw, label, meta = wrapper.next()
        native_result = native.batch(1, task) if cls is TaskStream else native.batch(1, task, wrapper.kwargs)
        assert np.array_equal(raw, native_result[0][0].numpy())
        assert label == int(native_result[1][0]) and meta == native_result[2][0]
    state = json.loads(json.dumps(wrapper.state_dict()))
    expected = wrapper.next()
    restored = e.DemoStream(task, cell)
    restored.load_state_dict(state)
    actual = restored.next()
    assert np.array_equal(expected[0], actual[0]) and expected[1:] == actual[1:]
    bad = copy.deepcopy(state); bad['provenance']['dataset_manifest_sha256'] = 'changed'
    stable = restored.state_dict()
    with pytest.raises(ValueError): restored.load_state_dict(bad)
    assert restored.state_dict() == stable
    assert before == e.source_hashes()
    assert raw.dtype == np.float32 and raw.shape[1:] == (3,100,100)
    assert all('model' not in x.lower() and 'train' not in x.lower() for x in e.import_trace())


@pytest.mark.parametrize('task,cell', [('contrast','mixed'), ('chromatic_increment','mixed'),
    ('orientation_ring','D0'), ('krauzlis_cued_motion','B12'), ('image_recognition','N4_H3')])
def test_lossless_export_gif_decoded_timeline_and_determinism(tmp_path, task, cell):
    e = exporter()
    assert hasattr(e, 'export_episode'), 'Lossless frame/GIF exporter is missing'
    from SecondPass.TaskSuite.Demo import validate_assets as v
    import numpy as np
    from PIL import Image, ImageSequence
    import zipfile
    stream = e.DemoStream(task, cell)
    raw, label, meta = stream.next()
    episode = e.export_episode(tmp_path, stream, raw, label, meta, 'test native episode')
    dist = tmp_path / 'dist'
    assert episode['metadata'] == meta
    assert episode['label_meaning'] == e.TASKS[task]['labels'][label]
    assert np.array_equal(np.load(tmp_path / episode['raw_path']), raw)
    expected = np.rint(np.clip(raw,0,1)*255).astype(np.uint8).transpose(0,2,3,1)
    for i,path in enumerate(episode['frames']):
        assert np.array_equal(np.asarray(Image.open(dist/path)), expected[i])
    with Image.open(dist / episode['gif']) as gif:
        durations = [im.info['duration'] for im in ImageSequence.Iterator(gif)]
    assert sum(durations) == 300*len(raw)
    with zipfile.ZipFile(dist / episode['frame_zip']) as bundle:
        assert len([n for n in bundle.namelist() if n.endswith('.png')]) == len(raw)
        for path in episode['frames']:
            assert bundle.read(Path(path).name) == (dist/path).read_bytes()
    result = v.validate_episode(episode, tmp_path)
    assert result['png_exact'] and result['gif_timeline_exact'] and result['native_semantics']
    assert episode['conversion_error']['max_abs'] <= .5/255 + 1e-7
    replay = e.export_episode(tmp_path/'replay', stream, raw, label, meta, 'test native episode')
    assert episode == replay
    assert (dist/episode['gif']).read_bytes() == (tmp_path/'replay'/'dist'/replay['gif']).read_bytes()
    assert (dist/episode['frame_zip']).read_bytes() == (tmp_path/'replay'/'dist'/replay['frame_zip']).read_bytes()
    if task == 'image_recognition':
        probe = meta['probe_frames']
        assert all(np.array_equal(raw[probe[0]], raw[i]) for i in probe)
        assert [p['phase'] for p in episode['phases'] if p['index'] in probe] == ['probe']*len(probe)
    # Mutating any original PNG or semantic timing must be caught.
    path = dist / episode['frames'][0]
    content = path.read_bytes(); path.write_bytes(content+b'corruption')
    with pytest.raises(ValueError): v.validate_episode(episode, tmp_path)
    path.write_bytes(content)
    bad = copy.deepcopy(episode); bad['phases'][0]['cue_visible'] = not bad['phases'][0]['cue_visible']
    with pytest.raises(ValueError): v.validate_episode(bad, tmp_path)


def test_bounded_bank_resume_byte_identity_and_partial_asset_rejection(tmp_path):
    e = exporter()
    assert hasattr(e, 'export_bank'), 'Bounded/resumable exporter is missing'
    from SecondPass.TaskSuite.Demo import validate_assets as v
    cells = [('contrast','mixed'), ('image_recognition','N4_H3')]
    first = e.export_bank(tmp_path/'resumed', cells=cells, max_new_draws=2)
    assert not first['complete']
    resumed = e.export_bank(tmp_path/'resumed', cells=cells)
    clean = e.export_bank(tmp_path/'clean', cells=cells)
    assert resumed == clean and resumed['complete']
    assert (tmp_path/'resumed/artifacts/manifest.json').read_bytes() == (tmp_path/'clean/artifacts/manifest.json').read_bytes()
    manifest = json.loads((tmp_path/'resumed/artifacts/manifest.json').read_text())
    assert len({x['id'] for x in manifest['episodes']}) == len(manifest['episodes'])
    assert manifest['provenance']['namespace'] == e.NAMESPACE
    for ep in manifest['episodes']:
        assert ep['seed'] < 2**53  # metadata also survives JSON -> JavaScript exactly
        v.validate_episode(ep,tmp_path/'resumed')
    for cell in manifest['cells']:
        assert cell['episode_ids'] and cell['showcase_id'] in cell['episode_ids']
    contacts = sorted((tmp_path/'resumed/verification').glob('export-contact*.png'))
    assert contacts and all(p.stat().st_size>0 for p in contacts)
    # Completed resume does zero additional draws; it must still inspect bytes.
    assert e.export_bank(tmp_path/'resumed', cells=cells) == resumed
    frame = tmp_path/'resumed/dist'/manifest['episodes'][0]['frames'][0]
    frame.unlink()
    with pytest.raises(ValueError,match='missing asset'):
        e.export_bank(tmp_path/'resumed', cells=cells)
    with pytest.raises(ValueError,match='coverage'):
        e.export_bank(tmp_path/'limited', cells=[('contrast','mixed')], max_attempts=1)


def test_validator_rejects_forged_photometric_measurements_and_timeline(tmp_path):
    e = exporter()
    from SecondPass.TaskSuite.Demo import validate_assets as v
    from PIL import Image, ImageSequence
    import hashlib
    stream=e.DemoStream('image_recognition','N0_H3')
    raw,label,meta=stream.next()
    episode=e.export_episode(tmp_path,stream,raw,label,meta,'test')
    metadata_path=tmp_path/'dist'/episode['metadata_path']
    bad=copy.deepcopy(episode); bad['gif_quantization_error']['mean_abs']=999
    e.atomic_json(metadata_path,bad)
    with pytest.raises(ValueError,match='quantization'):
        v.validate_episode(bad,tmp_path)
    bad=copy.deepcopy(episode); bad['conversion_error']['max_abs']=999
    e.atomic_json(metadata_path,bad)
    with pytest.raises(ValueError,match='conversion'):
        v.validate_episode(bad,tmp_path)
    # Keep the file hash self-consistent: decoded timing still has to catch
    # a syntactically valid GIF whose first hold was extended by one frame.
    path=tmp_path/'dist'/episode['gif']
    with Image.open(path) as gif:
        frames=[f.copy() for f in ImageSequence.Iterator(gif)]
        durations=[f.info['duration'] for f in frames]
    durations[0]+=300
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False,disposal=1)
    bad=copy.deepcopy(episode); bad['asset_sha256'][episode['gif']]=hashlib.sha256(path.read_bytes()).hexdigest()
    e.atomic_json(metadata_path,bad)
    with pytest.raises(ValueError,match='timeline'):
        v.validate_episode(bad,tmp_path)
