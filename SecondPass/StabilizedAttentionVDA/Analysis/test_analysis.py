import numpy as np
import torch
from .design import movie,masks,bias,PRIMARY_ARMS,CHECKPOINT
from .experiment import renderer_checks
from ..model import ExpandedSuite

def test_exact_native_and_change_timing():
    torch.set_num_threads(2)
    assert len(renderer_checks())==3
    x,m=movie(4,12,18,'uncued',1,cue=3)
    assert m['cue']==3 and m['change_index']==0 and m['label']==1 and m['counterfactual_cue_violation']
    a=next(a for a in PRIMARY_ARMS if a['name']=='cued_-4_change')
    b=bias(m,a,masks(4))
    assert np.all(b[:5]==0) and np.any(b[5:]<0) and np.all(b[:,masks(4)[0][0]]==0)

def test_zero_and_global_control_and_frozen_attention():
    torch.set_num_threads(2)
    s=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=ExpandedSuite().eval().requires_grad_(False);model.load_state_dict(s['model'])
    x,m=movie(4,12,18,'uncued',1,cue=3)
    with torch.inference_mode():
        z,d=model(x[None],'vda4',True)
        zero,dd=model(x[None],'vda4',True,torch.zeros(7,100))
        global_z,gg=model(x[None],'vda4',True,torch.full((7,100),8.))
        assert torch.equal(z,zero) and torch.equal(d['attention'],dd['attention'])
        torch.testing.assert_close(z,global_z,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(d['attention'],gg['attention'],atol=2e-5,rtol=2e-5)
        arm=next(a for a in PRIMARY_ARMS if a['name']=='cued_-4_change')
        _,ii=model(x[None],'vda4',True,torch.from_numpy(bias(m,arm,masks(4))))
        assert torch.equal(d['attention'][:,:5],ii['attention'][:,:5])
        site=torch.tensor(masks(4)[0][3]).reshape(10,10)
        assert (ii['attention'][0,5]*site).sum() < (d['attention'][0,5]*site).sum()
