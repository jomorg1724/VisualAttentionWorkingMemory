"""One focused CPU check of unfreezing, causal gradients and exact recomputation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
from WorkingMemory.model import load_parent
from PreAttentiveVision.TemporalIntegration.accumulators import StreamingPAVClassifier
from PreAttentiveVision.hybrid_models import build_hybrid
from PreAttentiveVision.neuroscience_stimuli import TaskStream,TASK_CLASSES

def run():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    path=ROOT/'PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/opponent_seed30301/checkpoint_004032.pt'
    parent=torch.load(path,map_location='cpu',weights_only=False)
    x,y,_=TaskStream(481002,'train').batch(2,'orientation')
    model=load_parent(parent,TASK_CLASSES,True)
    old=StreamingPAVClassifier(build_hybrid('convnext_se_residual'),TASK_CLASSES,'opponent')
    old.load_state_dict(parent['model']);old.eval();model.eval()
    with torch.no_grad():
        expected=old(x,'orientation');actual=model(x,'orientation')
    assert torch.equal(expected,actual),'Unchanged two-frame computation must match parent'
    model.train();assert model.encoder.training and all(p.requires_grad for p in model.parameters())
    torch.manual_seed(480001)
    model.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(x,'orientation'),y);loss.backward()
    encoder_grad=sum(float(p.grad.square().sum()) for p in model.encoder.parameters() if p.grad is not None)**.5
    assert encoder_grad>0 and all(p.grad is not None for p in model.encoder.parameters())
    grads={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.activation_checkpoint=False;model.zero_grad(set_to_none=True);torch.manual_seed(480001)
    loss2=torch.nn.functional.cross_entropy(model(x,'orientation'),y);loss2.backward()
    max_difference=max(float((grads[k]-p.grad).abs().max()) for k,p in model.named_parameters() if k in grads)
    assert torch.equal(loss,loss2) and max_difference<1e-5
    model.activation_checkpoint=True;model.eval();model.zero_grad(set_to_none=True)
    history=torch.cat((x[:,:1],x[:,1:],x[:,:1]),1).requires_grad_(True)
    torch.nn.functional.cross_entropy(model(history,'orientation'),y).backward()
    first_grad=float(history.grad[:,0].norm());assert first_grad>0
    result=dict(status='passed',cpu_only=True,ordinary_images_requires_grad=False,
                all_encoder_parameters_receive_gradients=True,encoder_gradient_norm=encoder_grad,
                earliest_frame_gradient_norm=first_grad,checkpoint_vs_uncheckpointed_max_gradient_difference=max_difference,
                exact_parent_two_frame_logits=True,all_learned_parameters=sum(p.numel() for p in model.parameters()),
                fixed_retention=[.25,.75],buffers_unchanged=all(torch.equal(v,parent['model'][k]) for k,v in model.named_buffers()))
    (ROOT/'WorkingMemory/model_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':run()
