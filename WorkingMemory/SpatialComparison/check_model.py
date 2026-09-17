import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,time,torch
from pathlib import Path
from WorkingMemory.SpatialComparison.model import migrate
from WorkingMemory.SpatialComparison.stimuli import SpatialStream
from WorkingMemory.SpatialComparison.sweep import PARENT
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.time();parent=torch.load(PARENT,map_location='cpu');cfg=dict(model_seed=31973001,new_lr=.0003,parent_lr=.00003,weight_decay=.0001,adam_eps=1e-10);rows={};common=[]
    for arm in ('dense_comparator','spatial_ei'):
        model,opt,lineage=migrate(parent,arm,cfg);model.train();x,y,meta=SpatialStream(99).batch(2,'orientation_binding',dict(delay=4));out,diagnostic=model(x,'orientation_binding',True);torch.nn.functional.cross_entropy(out,y).backward()
        e=float(sum(p.grad.square().sum() for p in model.encoder.parameters() if p.grad is not None).sqrt());early=float(diagnostic['states'][0][0].grad.norm());field=float(diagnostic['first_field'].grad.norm());assert e>0 and early>0 and field>0 and torch.isfinite(out).all()
        w=model.memory.recurrent_weight();n=205 if arm=='dense_comparator' else 51
        assert (w[:,:n]>=0).all() and (w[:,n:]<=0).all()
        assert all(torch.equal(model.state_dict()[name],parent['model'][name]) for name in lineage['retained_tensors'])
        assert all(opt.state[p]['exp_avg'].shape==p.shape for p in opt.state)
        if arm=='spatial_ei':assert diagnostic['states'][-1][0].shape==(2,64,13,13)
        common.append(model.comparison_output.weight.detach().clone());rows[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),encoder_grad=e,early_rate_grad=early,first_field_grad=field,raw_recurrent_grad=float(model.memory.raw_recurrent.grad.norm()),retained_tensors=len(lineage['retained_tensors']),adam_states=len(lineage['adam_states_by_name']),retained_weights_exact=True,presynaptic_signs_valid=True)
    assert torch.equal(*common)
    Path(__file__).with_name('model_checks.json').write_text(json.dumps(dict(status='passed',cpu_only=True,seconds=time.time()-start,common_output_initial_weights_exact=True,arms=rows),indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
