"""Read-only CPU diagnosis of a failed tiny-set checkpoint; same budget."""
from control import *
import json,time
OUT=Path(__file__).resolve().parent
budget=json.loads((OUT/'budget.json').read_text())
assert time.time()<budget['deadline_unix']-15,'Expired nonrenewable deadline'
torch.set_num_threads(1); torch.set_num_interop_threads(1)
checkpoint=torch.load(OUT/'overfit_only.pt',map_location='cpu',weights_only=False)
stream=NativeStream(910001,'train'); quota={'target':16,'foil':8,'catch':8}; accepted=[]; draws=0
while sum(quota.values()):
    b=(12,20,28)[draws%3]; x,y,meta=stream.batch(1,b); draws+=1
    event=meta[0]['event_type']
    if quota[event]: quota[event]-=1; accepted.append((x,y,meta))
result={}
for role in ('initial','failed_fit'):
    torch.manual_seed(72000); model=Model()
    if role=='failed_fit': model.load_state_dict(checkpoint['model'])
    activations={}; hooks=[]
    for name,module in (('motion',model.motion),('patch',model.patch),('readout_pre_GELU',model.readout[1]),('readout_post_GELU',model.readout[2])):
        def capture(_m,_i,o,name=name): activations.setdefault(name,[]).append(o.detach().numpy())
        hooks.append(module.register_forward_hook(capture))
    losses=[]
    for x,y,_ in accepted[:8]:
        loss=F.cross_entropy(model(x),y); (loss/8).backward(); losses.append(loss.item())
    stats={}
    for name,arrs in activations.items():
        flat=np.concatenate([a.ravel() for a in arrs]); stats[name]=dict(mean=float(flat.mean()),std=float(flat.std()),min=float(flat.min()),max=float(flat.max()),fraction_below_minus5=float((flat<-5).mean()),fraction_exact_zero=float((flat==0).mean()))
    gradients={}
    for name,p in model.named_parameters():
        group=name.split('.')[0]; gradients[group]=gradients.get(group,0.)+float(p.grad.square().sum())
    gradients={k:math.sqrt(v) for k,v in gradients.items()}
    result[role]=dict(loss_first8=float(np.mean(losses)),activation_statistics=stats,gradient_group_norms=gradients)
    for hook in hooks: hook.remove()
result['elapsed_seconds']=time.time()-budget['start_unix']; result['hypothesis']='Excessive early AdamW step drives dense GELU head negative, suppressing sensory credit; compare readout preactivations and group gradient norms rather than assuming disconnected gradients.'
(OUT/'failed_fit_localization.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
