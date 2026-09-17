import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from common import *
import torch
from diagnostic import forward,sensory_fields,stages
from WorkingMemory.SpatialComparison.stimuli import SpatialStream
torch.set_num_threads(1);torch.set_num_interop_threads(1)
result={};stream=SpatialStream(46973001,'test');x,y,_=stream.batch(2,'orientation_single',dict(delay=0,spacing='mixed'))
for arm,model in models().items():
    with torch.no_grad():ordinary=model(x,'orientation');fields=sensory_fields(model,x);wrapped=forward(model,fields,'orientation');noop=forward(model,fields,'orientation',exclude_frames=stages('orientation_single',0)['blanks'])
    assert torch.equal(ordinary,wrapped['logits']) and torch.equal(ordinary,noop['logits'])
    result[arm]=dict(baseline_exact=True,empty_phase_noop_exact=True,branch_reconstruction_max_abs=wrapped['branch_reconstruction_max_abs'])
    if arm=='attention':
        changed=forward(model,fields,'orientation',exclude_frames=[1,2]);assert changed['attention_stats'][:,1:3,:,0].abs().max()==0
        assert (changed['parts'][:,0]-wrapped['parts'][:,0]).abs().max()==0
        probe=forward(model,fields,'orientation',exclude_frames=[4]);assert torch.equal(probe['parts'][:,:1],wrapped['parts'][:,:1]) and torch.equal(probe['parts'][:,2],wrapped['parts'][:,2])
        result[arm].update(excluded_source_mass_exact_zero=True,unchanged_sensory_branch_exact=True,probe_only_comparator_exact=True)
write(HERE/'checks.json',dict(status='passed',models=result,checkpoint_sha256={a:sha(p) for a,p in paths().items()}));print(json.dumps(result))
