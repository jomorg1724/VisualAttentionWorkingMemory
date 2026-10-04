"""CPU-only executable audit of the separately extracted deployed sources.
Run each arm in a separate interpreter so imports cannot cross-contaminate.
No checkpoint reads, production edits, accelerator work, or trained-weight saves.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import sys,time,json,hashlib,importlib,inspect,signal,copy
from pathlib import Path
arm=sys.argv[1]
out=Path(__file__).resolve().parent
root=Path('/Users/jonathanmorgan/VAWMRuntime')/('cloud_krauzlis90_dense01' if arm=='dense' else 'cloud_krauzlis90_fresh02')/'clean_extract/repo'
sys.path.insert(0,str(root))
import numpy as np
import torch
from torch.nn import functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('arm 85s wall cap')));signal.alarm(85)
w=importlib.import_module('SecondPass.SpatialReadout.SpatialConsolidation.'+('Krauzlis90Dense' if arm=='dense' else 'Krauzlis90Fresh')+'.worker')
T=w.TASK;cell=w.CELLS[0];result={'arm':arm,'root':str(root),'torch':torch.__version__,'threads':torch.get_num_threads(),'tests':{}}
def record(name,**kw):
 result['tests'][name]=kw; (out/(arm+'_results.json')).write_text(json.dumps(result,indent=2));print(name,json.dumps(kw),flush=True)
def h(x): return hashlib.sha256(x.numpy().tobytes()).hexdigest()
# Exact renderer intervention with fixed nuisance RNG: target vs catch and foil.
S=w.SpatialBatteryStream
causal=[]
for target in (0,1):
 for baseline in (12,20,28):
  renders={}
  for event in ('target','foil','catch'):
   s=S(712);s._event_kind=event
   x,m=s._krauzlis(np.random.default_rng(831),int(event=='target'),target,{'baseline_transitions':baseline})
   renders[event]=(np.stack(x),m)
  catch=renders['catch'][0];onset=baseline+8
  for event in ('target','foil'):
   x,m=renders[event];patch=target if event=='target' else 1-target
   assert np.array_equal(x[:onset],catch[:onset]) and np.array_equal(x[-1],catch[-1])
   diff=np.any(x!=catch,axis=1);ys,xs=np.where(diff[onset])
   assert len(xs)>0 and np.any((xs<50) if patch==0 else (xs>50))
   # Shared RNG couples reset draws across patches; nuisance pixels can change
   # in the other patch even on onset. Direction means still change only here.
   delta=np.array(m['postevent_means_degrees'])-np.array(m['baseline_means_degrees'])
   assert abs(abs(delta[patch])-90)<1e-9 and abs(delta[1-patch])<1e-9
   assert m['changed_patch']==patch
   causal.append({'target':target,'baseline':baseline,'event':event,'first_changed_frame':int(np.where(diff.any((1,2)))[0][0]),'changed_pixels_first':len(xs),'counterfactual_other_patch_pixels_first':int(np.sum(xs>=50) if patch==0 else np.sum(xs<50))})
record('causal_renderer',passed=True,cases=causal,no_pre_event_or_report_label_difference=True)
# Label permutation, boundary batching, native equality and split identity.
events={k:0 for k in ('target','foil','catch')};s=w.FreshStream('train');ids=[]
for i in range(100):
 x,y,meta=s.batch(1,T,cell);m=meta[0];events[m['event_type']]+=1
 assert int(y[0])==m['label']==int(m['changed_patch']==m['target_location'])
 assert x.dtype==torch.float32 and y.dtype==torch.int64 and 0<=x.min()<=x.max()<=1
 ids.append(m['suite_trial_id'])
assert events=={'target':57,'foil':29,'catch':14} and len(set(ids))==100
s1=w.FreshStream('train');s2=w.FreshStream('train');x,y,meta=s1.batch(2,T,cell)
xa,ya,ma=s2.batch(1,T,cell);xb,yb,mb=s2.batch(1,T,cell)
assert torch.equal(x,torch.cat([xa,xb])) and torch.equal(y,torch.cat([ya,yb])) and meta==ma+mb
native=S(s1.stream_seed(T,cell),'train');xx,yy,mm=native.batch(2,T,{'baseline_transitions':12});assert torch.equal(x,xx) and torch.equal(y,yy)
splits={p:w.FreshStream(p) for p in ('train','val','test')};seeds=[s.stream_seed(T,c) for s in splits.values() for c in w.CELLS];assert len(set(seeds))==9
hashes={p:h(s.batch(1,T,cell)[0]) for p,s in splits.items()};assert len(set(hashes.values()))==3
record('labels_streams',passed=True,events=events,split_cell_seeds=seeds,split_raster_hashes=hashes,batch_chunk_order_equal=True,adapter_native_exact=True)
# Rebind neural fixture to one episode of each class, not a majority-only batch.
s=w.FreshStream('train');chosen={}
for _ in range(100):
 a,b,c=s.batch(1,T,cell)
 chosen.setdefault(int(b[0]),(a,b,c))
 if len(chosen)==2:break
x=torch.cat([chosen[k][0] for k in (0,1)]);y=torch.cat([chosen[k][1] for k in (0,1)]);meta=sum([chosen[k][2] for k in (0,1)],[])
# Same physical event, opposite cue: only cue pixels differ; labels must flip.
cue_pairs=[]
for patch in (0,1):
 pairs=[]
 for event,target,label in [('target',patch,1),('foil',1-patch,0)]:
  s=S(712);s._event_kind=event
  a,m=s._krauzlis(np.random.default_rng(831),label,target,{'baseline_transitions':12});pairs.append(np.stack(a))
 assert np.array_equal(pairs[0][2:],pairs[1][2:]) and not np.array_equal(pairs[0][:2],pairs[1][:2])
 cue_pairs.append(patch)
record('cue_label_counterfactual',passed=True,changed_patches=cue_pairs,opposite_labels_identical_motion_only_cue_differs=True)
# Exercise real constructor/optimizer, but suppress no safeguards and load no weights.
cfg={'device':'cpu','effective_batch':2,'microbatch':1,'cap_started':time.time(),'deadline':time.time()+80,'disposable_profile':True}
session=w.Session(out/(arm+'_scratch'),cfg);model=session.model;opt=session.optimizer
params=dict(model.named_parameters());assert {id(p) for p in model.parameters()}=={id(p) for g in opt.param_groups for p in g['params']}
# Correct temporal stack and center verified with deliberately distinct frame values.
test=torch.stack([torch.full((1,3,100,100),v) for v in (.1,.3,.8,.6)],1)
z=model.frames(test);expected=[]
for t in range(4):expected.append(torch.cat([torch.zeros_like(test[:,0]) if j<0 else test[:,j]-.5 for j in (t-2,t-1,t)],1))
assert torch.equal(z,torch.stack(expected,1))
record('constructor_optimizer_frames',passed=True,parameter_tensors=len(params),parameter_count=sum(p.numel() for p in params.values()),optimizer_group_count=len(opt.param_groups),ordered_centered_stack=True)
# Full native 29-frame input, not a shortened surrogate. Cross-entropy gradients.
model.train();x=x.detach().requires_grad_();logits=model(x,T);loss=F.cross_entropy(logits,y);loss.backward()
full={n:p.grad.detach().clone() for n,p in params.items() if p.grad is not None};inputnorm=x.grad.flatten(2).norm(dim=2).detach().tolist()
assert all(torch.isfinite(g).all() for g in full.values()) and inputnorm[0][0]>0
active=[n for n in params if not n.startswith('heads.') or n.startswith('heads.'+T+'.')];assert set(full)==set(active)
record('native_bptt',passed=True,shape=list(x.shape),loss=float(loss.detach()),labels=y.tolist(),logits=logits.detach().tolist(),per_frame_input_grad_norm=inputnorm,all_active_parameter_gradients_present=True,nonzero_gradient_tensors=sum(bool(g.abs().max()>0) for g in full.values()))
# Run the EXACT deployed update (capturing step so same weights compare gradients).
class Fixture:
 def __init__(self):self.i=0
 def batch(self,n,*_):
  i=self.i;self.i+=n;return x.detach()[i:i+n],y[i:i+n],meta[i:i+n]
step=opt.step;opt.step=lambda:None
row=w.fresh.cloud.update(model,opt,Fixture(),T,cell,2,1,'cpu');opt.step=step
errors={n:float((params[n].grad-g).abs().max()) for n,g in full.items()}
maxerr=max(errors.values());assert maxerr<2e-5
assert abs(row['loss']-float(loss.detach()))<2e-6
record('exact_worker_microbatch',passed=True,max_gradient_absolute_difference=maxerr,loss_difference=abs(row['loss']-float(loss.detach())),unweighted_mean_ce=True)
# Native finite difference on highest-magnitude entries, covering sensory and head.
fd=[]
# Use float64 finite differences to resolve small balanced-batch recurrent grads.
model.double();xd=x.detach().double()
fd_names=[('encoder.0.weight' if arm=='dense' else 'blocks.0.0.weight'),('dense_gru.gates.weight' if arm=='dense' else 'spatial_gru.gates.weight'),'heads.'+T+'.weight']
fd_grads=dict(zip(fd_names,torch.autograd.grad(F.cross_entropy(model(xd,T),y),[params[n] for n in fd_names])))
for prefix in fd_names:
 p=params[prefix];g=fd_grads[prefix];idx=int(g.abs().argmax());original=float(p.detach().flatten()[idx]);eps=1e-5
 vals=[]
 for sign in (1,-1):
  with torch.no_grad():
   p.flatten()[idx]=original+sign*eps;vals.append(float(F.cross_entropy(model(xd,T),y)))
 with torch.no_grad():p.flatten()[idx]=original
 analytic=float(g.flatten()[idx]);numeric=(vals[0]-vals[1])/(2*eps);relative=abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-6)
 assert relative<.15 or abs(analytic-numeric)<1e-4
 fd.append(dict(parameter=prefix,index=idx,analytic=analytic,numeric=numeric,relative_error=relative))
model.float()
record('finite_difference',passed=True,precision='float64',entries=fd)
# The real Adam step must change every active tensor and not inactive heads.
before={n:p.detach().clone() for n,p in params.items()};step();changed=[n for n,p in params.items() if not torch.equal(before[n],p)]
assert set(changed)==set(active)
with torch.no_grad():after=float(F.cross_entropy(model(x.detach(),T),y))
record('adam_step',passed=True,changed_active_tensors=len(changed),before_ce=float(loss.detach()),after_ce=after,ce_decreased=after<float(loss.detach()))
# A first-step increase is not a reversed gradient: test the same Adam direction
# at a tenth step without any extra optimizer updates or retained training.
post={n:p.detach().clone() for n,p in params.items()}
with torch.no_grad():
 for n,p in params.items():p.copy_(before[n]+.1*(post[n]-before[n]))
 tenth=float(F.cross_entropy(model(x.detach(),T),y))
 for n,p in params.items():p.copy_(post[n])
record('adam_direction_scale',tenth_step_ce=tenth,original_ce=float(loss.detach()),full_step_ce=after,tenth_step_descends=tenth<float(loss.detach()))
# Batch permutation equivariance, stream resume, no cross-call hidden carry.
with torch.no_grad():
 original_logits=model(x.detach(),T);permuted=model(x.detach().flip(0),T).flip(0)
 unrelated=model(torch.full_like(x.detach(),.5),T);replayed=model(x.detach(),T)
assert torch.allclose(original_logits,permuted,atol=2e-6) and torch.equal(original_logits,replayed)
assert abs(float(F.cross_entropy(original_logits,y))-float(F.cross_entropy(permuted.flip(0),y.flip(0))))<1e-6
s=w.FreshStream('train');s.batch(1,T,cell);state=s.state_dict();a,b,c=s.batch(2,T,cell);s.load_state_dict(state);d,e,f=s.batch(2,T,cell)
assert torch.equal(a,d) and torch.equal(b,e) and c==f
record('permutation_reset_resume',passed=True,logit_max_difference=float((original_logits-permuted).abs().max()),state_reset_exact=True,stream_resume_exact=True)
# Evaluate class-index scoring via exact scorer with perfect one-hot outputs.
labels=[0,1,0,1];probs=[[.99,.01],[.01,.99],[.99,.01],[.01,.99]];metadata=[{'event_type':e} for e in ('foil','target','catch','target')]
score=w.score_cell(T,cell,labels,probs,metadata);assert score['balanced_accuracy']==1 and score['auc']==1
record('class_index_scoring',passed=True,balanced_accuracy=score['balanced_accuracy'],auc=score['auc'])
# Imported-file provenance: executable path and hashes, NOT presumed historical equality.
files={}
for name,module in list(sys.modules.items()):
 p=getattr(module,'__file__',None)
 if p and Path(p).is_relative_to(root) and Path(p).is_file():files[str(Path(p).relative_to(root))]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
group_fn=w.fresh.cloud.update.__globals__['parameter_group']
misclassified=[n for n in active if group_fn(n,T)=='inactive_heads']
record('diagnostic_group_names',active_parameters_reported_as_inactive=misclassified,affects_optimizer=False)
manifest=json.loads((root.parent/'deployment_manifest.json').read_text())
manifest_map={r['path']:r['sha256'] for r in manifest['files']}
missing=[p for p in files if 'repo/'+p not in manifest_map]
mismatched=[p for p,v in files.items() if 'repo/'+p in manifest_map and manifest_map['repo/'+p]!=v]
assert not missing and not mismatched
result['manifest_verification']={'imported_files':len(files),'missing':missing,'mismatched':mismatched,'source_version':manifest['source_version']}
result.update(imported_source_sha256=files,wall_seconds=time.monotonic()-start,complete=True)
(out/(arm+'_results.json')).write_text(json.dumps(result,indent=2));signal.alarm(0)
print('COMPLETE',arm,result['wall_seconds'],flush=True)
