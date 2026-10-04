"""CPU constructor audit only: no model forwards, datasets, or checkpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import sys, json, hashlib, subprocess
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from SecondPass.SpatialReadout.model import SpatialReadout
catalog = json.loads((ROOT/'SecondPass/TaskSuite/catalog.json').read_text())
classes = {t['id']:t['classes'] for t in catalog['tasks']}
torch.manual_seed(98192763)
model = SpatialReadout(classes)
count = lambda m: sum(p.numel() for p in m.parameters())
groups = {name:count(m) for name,m in model.named_children()}
blocks=[]
n=100
for i,block in enumerate(model.blocks):
    conv,norm,_=block
    n=(n+2*conv.padding[0]-conv.dilation[0]*(conv.kernel_size[0]-1)-1)//conv.stride[0]+1
    blocks.append(dict(index=i+1,in_channels=conv.in_channels,out_channels=conv.out_channels,kernel=conv.kernel_size[0],stride=conv.stride[0],padding=conv.padding[0],side=n,parameters=count(block),norm_groups=norm.num_groups,norm_epsilon=norm.eps))
files=['SecondPass/SpatialReadout/model.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py','SecondPass/SpatialReadout/FreshRun/worker.py','SecondPass/SpatialReadout/ContinuationRun/worker.py','SecondPass/SpatialReadout/ContinuationRun/README.md','SecondPass/SpatialReadout/README.md','SecondPass/SpatialReadout/BRIEF.md','LabJournal/spatial-readout-convgru.md','SecondPass/TaskSuite/catalog.json','SecondPass/TaskSuite/suite.py','SecondPass/JointTraining/worker.py','SecondPass/JointTraining/core.py','SecondPass/SpatialComparisonReadout/CloudRun/worker.py','ANALYSIS_SOP.md']
sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
result=dict(method='Fresh CPU constructor and analytical convolution shape propagation; no forward/backward, no checkpoints, no data',torch_version=torch.__version__,threads=torch.get_num_threads(),inter_op_threads=torch.get_num_interop_threads(),total_parameters=count(model),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),groups=groups,blocks=blocks,projection_counts=[count(x) for x in model.proj],kda_counts=[count(x) for x in model.acc],kda_input_count=count(model.acc[0].inputs),kda_output_count=count(model.acc[0].output),convgru_gate_count=count(model.spatial_gru.gates),convgru_candidate_count=count(model.spatial_gru.candidate),heads={t:dict(classes=classes[t],parameters=count(m)) for t,m in model.heads.items()},tasks=len(classes),conditions=sum(len(t['conditions']) for t in catalog['tasks']),labels={t['id']:t['labels'] for t in catalog['tasks']},kda_state_scalars=[s*s*2*8*16 for s in (25,13,7)],convgru_state_scalars=64*7*7,parameter_tensors={n:dict(shape=list(p.shape),numel=p.numel()) for n,p in model.named_parameters()},source_sha256=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),all_cpu=all(p.device.type=='cpu' for p in model.parameters()),no_removed_modules=all(not hasattr(model,n) for n in ('feat','gru','norm')),gate_bias_zero=bool((model.spatial_gru.gates.bias==0).all()),candidate_bias_zero=bool((model.spatial_gru.candidate.bias==0).all()))
assert result['total_parameters']==1383028
assert result['tasks']==13 and result['conditions']==35
assert sum(groups.values())==result['total_parameters']
(OUT/'count_verification.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'source_manifest.json').write_text(json.dumps(dict(git_head=result['git_head'],note='Dirty worktree; file hashes are authoritative for this local source snapshot, not deployment byte parity.',sha256=sources),indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('parameter_tensors','source_sha256')},indent=2))
