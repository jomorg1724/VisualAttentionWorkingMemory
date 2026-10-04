"""Post-execution CPU-only replay and intervention-substrate verification."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .run import load_model,model_hash
from .core import observe
OUT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    budget=json.loads((OUT/'budget.json').read_text());assert time.time()<budget['deadline_unix']
    model,_=load_model();before=model_hash(model)
    a=np.load(OUT/'data/orientation_cued_causal_stimuli.npz');x=torch.from_numpy(a['images'][:2]);meta=json.loads((OUT/'data/orientation_cued_causal_metadata.json').read_text())[:2]
    saved=[r for r in map(json.loads,(OUT/'data/trials.jsonl').read_text().splitlines()) if r['group']=='causal' and r['task']=='orientation_cued' and r['condition']=='sham'][:2]
    states=[];emissions=[];downstream=[]
    h=model.acc[0].register_forward_hook(lambda module,args,y:(states.append(y[1].clone()),emissions.append(y[0].clone())) and None)
    g=model.spatial_gru.register_forward_hook(lambda module,args,y:downstream.append(y.clone()))
    with torch.inference_mode():
        direct,_=observe(model,x,'orientation_cued',meta)
        first_states=states.copy();first_emissions=emissions.copy();first_downstream=downstream.copy();states.clear();emissions.clear();downstream.clear()
        perturbed,_=observe(model,x,'orientation_cued',meta,intervention=dict(kind='inhibit',site='cued',epoch='encoding',dose=1.))
    h.remove();g.remove()
    state_error=max(float((a-b).abs().max()) for a,b in zip(states,first_states));emission_error=float((emissions[2]-first_emissions[2]).abs().max());downstream_error=float((downstream[-1]-first_downstream[-1]).abs().max());replay_error=float(np.max(np.abs(direct.numpy()-np.array([r['logits'] for r in saved]))))
    assert state_error==0 and emission_error>0 and downstream_error>0 and replay_error<1e-4
    assert model_hash(model)==before and not any(p.grad is not None or p.requires_grad for p in model.parameters())
    result=dict(cpu_threads=1,cpu_model_presentations=4,first_scale_state_max_difference=state_error,encoding_emission_max_difference=emission_error,final_downstream_state_max_difference=downstream_error,saved_mps_vs_cpu_logit_max_difference=replay_error,model_parameters_immutable=True,no_mainmodel_gradients=True,executed_unix=time.time(),within_original_cap=time.time()<budget['deadline_unix'])
    (OUT/'model_replay_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
