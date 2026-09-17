"""One CPU migration and inherited family-prefix check; no GPU work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys,json,hashlib,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from WorkingMemory.TrainingExposure.model import migrate
from WorkingMemory.TrainingExposure.protocol import recipe,assess
from WorkingMemory.TrainingExposure.sweep import parent_path,write
from WorkingMemory.SpatialComparison.stimuli import SpatialStream
torch.set_num_threads(1);torch.set_num_interop_threads(1);began=time.time();cp=torch.load(parent_path(),map_location='cpu')
assert recipe('control_10')==cp['config']
arms={a:migrate(cp,a,recipe(a)) for a in ('control_10','focused_50')};streams={};exposures={}
for arm,(model,opt,lineage) in arms.items():
    assert all(torch.equal(v,model.state_dict()[name]) for name,v in cp['model'].items())
    current=opt.state_dict();assert current['param_groups']==cp['optimizer']['param_groups']
    for key,values in cp['optimizer']['state'].items():
        for name,v in values.items():
            actual=current['state'][key][name];assert torch.equal(v,actual) if torch.is_tensor(v) else v==actual
    cfg=recipe(arm);stream=SpatialStream(cfg['train_seed'],'train');stream.load_state_dict(cp['stream']);hashes={k:[] for k in ('orientation_single','orientation_binding','motion_direction')};count={name:0 for name in cfg['cells']}
    order=np.random.default_rng(cfg['scheduler_seed']+8400//80).permutation(80)
    for pos in order:
        name=cfg['cycle'][int(pos)];cell=cfg['cells'][name];x,y,m=stream.batch(8,cell['family'],cell['condition']);count[name]+=8
        stable=torch.cat((x[:,:10],x[:,-1:]),1) if cell['family']=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1)
        hashes[cell['family']].append(hashlib.sha256(stable.numpy().tobytes()+y.numpy().tobytes()).hexdigest())
    streams[arm]=hashes;exposures[arm]=count
matched={}
for family in streams['control_10']:
    a=streams['control_10'][family];b=streams['focused_50'][family];n=min(len(a),len(b));assert a[:n]==b[:n];matched[family]=n*8
write(HERE/'checks.json',dict(status='passed',seconds=time.time()-began,parent_step=8400,model_parameters_exact=True,all_adam_tensors_and_groups_exact=True,old_LR_policy_exact=True,control_recipe_exact_parent=True,matched_family_prefix_episodes_first80updates=matched,actual_firstcycle_cell_exposure=exposures,per_delay_prefix_not_claimed=True,new_parameters=0))
print(json.dumps(json.loads((HERE/'checks.json').read_text()),indent=2))
