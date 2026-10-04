import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream as Native


def test_renderer_isolated_paired():
    path=Path(__file__).with_name('stimuli.py')
    assert path.exists(), 'isolated ninety-degree renderer missing'
    from .stimuli import SpatialBatteryStream as Variant
    for baseline in (12,20,28):
        for event in ('target','foil','catch'):
            for seed in (81,82,83,84):
                a,b=Native(seed),Variant(seed)
                a._event_kind=b._event_kind=event
                args=(int(event=='target'),seed%2,dict(baseline_transitions=baseline))
                x,m=a._krauzlis(np.random.default_rng(seed),*args)
                y,n=b._krauzlis(np.random.default_rng(seed),*args)
                assert len(x)==len(y)==17+baseline
                assert np.array_equal(x[:8+baseline],y[:8+baseline])
                assert np.array_equal(x[-1],y[-1])
                for key in m:
                    if key not in ('event_magnitude_degrees','signed_change_degrees','postevent_means_degrees'):
                        assert m[key]==n[key],key
                assert n['event_magnitude_degrees']==90
                changed=n['changed_patch']
                if changed is None:
                    assert np.array_equal(x,y)
                else:
                    assert n['signed_change_degrees']==90*n['event_sign']
                    before=np.deg2rad(n['baseline_means_degrees'][changed]); after=np.deg2rad(n['postevent_means_degrees'][changed])
                    u=np.array([np.cos(before),np.sin(before)])
                    v=np.array([np.cos(after),np.sin(after)])
                    np.testing.assert_allclose(v,n['event_sign']*np.array([-u[1],u[0]]),atol=1e-14)
    # The only numerical intervention retains the native magnitude RNG draw.
    native=Path(__file__).resolve().parents[4]/'WorkingMemory/SpatialTaskBattery/stimuli.py'
    assert path.read_text()==native.read_text().replace('magnitude=int(rng.choice([26,28]));event=', 'magnitude=int(rng.choice([26,28]));magnitude=90;event=')
