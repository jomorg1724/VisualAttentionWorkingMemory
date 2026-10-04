import importlib.util
from pathlib import Path
import numpy as np

def test_input_only_equal_capacity_temporal_access():
    p=Path(__file__).with_name("upstream.py")
    assert p.exists(), "missing operational implementation"
    spec=importlib.util.spec_from_file_location("upstream",p)
    u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    rng=np.random.default_rng(12)
    data={"memory__"+k:rng.normal(size=(4,10)) for k in ("baseline","post","final")}
    proj=rng.normal(size=(10,3))
    a=u.access(data,"memory","final",proj)
    assert np.array_equal(a,u.access({"memory__final":data["memory__final"]},"memory","final",proj))
    poisoned={k:(v if k.endswith("final") else np.full_like(v,np.nan)) for k,v in data.items()}
    assert np.array_equal(a,u.access(poisoned,"memory","final",proj))
    assert a.shape==u.access(data,"memory","phases",proj).shape
    assert np.isfinite(a).all()

def test_motion_comparison_wrap_and_flow_window_contract():
    import upstream as u
    assert hasattr(u,'pixel_summary'), 'missing pixel input-only comparison'
    # Saved flow vectors are image-derived, not true movement schedules.
    a=np.deg2rad([179.,-179.])
    assert np.allclose(u.circular_change(np.array([[a[0],a[0]]]),np.array([[a[1],a[1]]])),2)
    frames=np.full((29,3,100,100),.5,np.float32)
    r=u.pixel_summary(frames)
    assert r['angles'].shape==(2,2,2)
    assert r['counts'].shape==(20,2)
    assert r['flow_sums'].shape==(20,2,2)
    assert np.isfinite(r['angles']).all()
