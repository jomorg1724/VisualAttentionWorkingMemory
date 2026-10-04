import json
from pathlib import Path
import time
import pytest
from . import worker as w


def test_measured_allocation_and_no_restart(tmp_path):
    b=dict(cap_started=100,hard_deadline=28900,deadline=28300,wall_cap_seconds=28800,retrieval_reserve_seconds=600,max_usd=5)
    d=tmp_path; (d/'profile').mkdir()
    profile=dict(complete=True,rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=.1) for _ in range(2) for c in w.CELLS],evaluation_cells=[dict(task=w.TASK,cell=c,n=20,seconds=.1) for c in w.CELLS])
    (d/'profile/profile.json').write_text(json.dumps(profile))
    p=w.measured_plan(d,b,now=200)
    assert p['max_steps']==10000 and sum(r['updates'] for r in p['planned_exposure'].values())==10000
    assert sum(r['episodes'] for r in p['planned_exposure'].values())==320000
    with pytest.raises(RuntimeError): w.measured_plan(d,b,now=28000)
    with pytest.raises(ValueError): w.validate_budget(b,dict(b,hard_deadline=30000),200)
    b=dict(cap_started=time.time()-1,hard_deadline=time.time()-1+28800,deadline=0,wall_cap_seconds=28800,retrieval_reserve_seconds=600)
    b['hard_deadline']=b['cap_started']+28800;b['deadline']=b['hard_deadline']-600
    (d/'budget.json').write_text(json.dumps(b));(d/'config.json').write_text(json.dumps(dict(b,source_hashes={})))
    (d/'initial.pt').write_bytes(b'existing-run-sentinel')
    with pytest.raises(RuntimeError,match='Fresh-only'): w.run(d)


def test_native_streams_and_proportions():
    from .stimuli import SpatialBatteryStream
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream as Native
    assert Native._krauzlis.__module__!='.'.join(SpatialBatteryStream._krauzlis.__module__.split('.'))
    for cell in w.CELLS:
        s=w.FreshStream('train'); native=s._make_stream(w.TASK,cell); counts=dict(target=0,foil=0,catch=0)
        for _ in range(100):
            y,t=native._case(w.TASK); counts[native._event_kind]+=1; native.counts[w.TASK]+=1
            assert y==int(native._event_kind=='target') and t in (0,1)
        assert counts==dict(target=57,foil=29,catch=14)


def test_bundle_is_explicit_checkpoint_free_closure(tmp_path):
    from . import bundle
    import tarfile,hashlib
    result=bundle.build(tmp_path)
    with tarfile.open(result['archive']) as tf:
        manifest=json.load(tf.extractfile('deployment_manifest.json'))
        assert manifest['checkpoints_included'] is False and manifest['photos_included'] is False
        assert not any(Path(x.name).suffix in ('.pt','.pth','.ckpt','.safetensors') for x in tf.getmembers())
        for r in manifest['files']: assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
