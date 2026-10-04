"""Independent saved-artifact verification, CPU one thread, no optimizer updates."""
from control import *
import json,time,collections
OUT=Path(__file__).resolve().parent
budget=json.loads((OUT/'budget.json').read_text())
assert time.time()<budget['deadline_unix']-8,'No verification compute after cap'
torch.set_num_threads(1); torch.set_num_interop_threads(1)
result=json.loads((OUT/'result.json').read_text()); manifest=json.loads((OUT/'scene_manifest.json').read_text())
seen={}; counts=collections.Counter()
for row in manifest:
    assert row['label']==int(row['event_type']=='target')
    assert row['event_magnitude_degrees']==90
    assert row['frame_count']==row['baseline_transitions']+17
    key=row['physical_scene_sha256']; split=row['experiment_split']
    assert key not in seen or seen[key]==split
    seen[key]=split; counts[split]+=1
assert len(seen)==len(manifest)
receipt={'split_counts':dict(counts),'unique_physical_scenes':len(seen),'metadata_labels_and_native_lengths_checked':True,'pairwise_physical_disjointness':True,'elapsed':time.time()-budget['start_unix']}
if result['generalization']['executed']:
    verify={}
    for role in ('selected','terminal'):
        rows=json.loads((OUT/f'test_{role}_predictions.json').read_text())
        y=np.array([r['label'] for r in rows]); p=np.array([r['probability'] for r in rows]); positive=y==1; negative=~positive
        auc=float(np.mean([np.mean((score>p[negative])+.5*(score==p[negative])) for score in p[positive]]))
        ba=float((np.mean(p[positive]>=.5)+np.mean(p[negative]<.5))/2)
        saved=result['generalization'][role+'_test']
        assert abs(auc-saved['auc'])<1e-12 and abs(ba-saved['balanced_accuracy'])<1e-12 and len(rows)==600
        state=torch.load(OUT/f'{role}.pt',map_location='cpu',weights_only=False)
        adamsteps=[float(s['step']) for s in state['optimizer']['state'].values()]
        assert all(s==state['step'] for s in adamsteps)
        model=Model(); model.load_state_dict(state['model']); model.eval()
        x,y0,_=NativeStream(940001,'test').batch(2,12)
        hidden=[]
        hook=model.readout[1].register_forward_hook(lambda _m,_i,o: hidden.append(o.detach()))
        with torch.no_grad(): parity=(model(x).softmax(-1)[:,1].numpy()-p[:2])
        hook.remove()
        F.cross_entropy(model(x),y0).backward()
        groupgrad={}
        for name,param in model.named_parameters():
            group=name.split('.')[0]; groupgrad[group]=groupgrad.get(group,0.)+float(param.grad.square().sum())
        groupgrad={k:math.sqrt(v) for k,v in groupgrad.items()}
        assert abs(parity).max()<1e-4,parity
        verify[role]=dict(n=len(rows),balanced_accuracy=ba,auc=auc,step=state['step'],Adam_state_tensors=len(adamsteps),Adam_steps_unique=sorted(set(adamsteps)),CPU_saved_inference_max_error=float(abs(parity).max()),dense_preGELU_mean=float(hidden[0].mean()),dense_preGELU_fraction_below_minus5=float((hidden[0]<-5).float().mean()),dense_GELU_fraction_exact_zero=float((F.gelu(hidden[0])==0).float().mean()),first2_gradient_group_norms=groupgrad)
    receipt['endpoints']=verify
    assert counts['train']==result['generalization']['train_episodes']
else:
    state=torch.load(OUT/'corrected_overfit_only.pt',map_location='cpu',weights_only=False)
    adamsteps=[float(s['step']) for s in state['optimizer']['state'].values()]
    assert all(s==state['step'] for s in adamsteps)
    receipt['corrected_overfit_optimizer']=dict(step=state['step'],Adam_state_tensors=len(adamsteps),Adam_steps_unique=sorted(set(adamsteps)))
receipt['elapsed']=time.time()-budget['start_unix']; receipt['within_original_budget']=time.time()<budget['deadline_unix']
(OUT/'saved_artifact_verification.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt,indent=2))
