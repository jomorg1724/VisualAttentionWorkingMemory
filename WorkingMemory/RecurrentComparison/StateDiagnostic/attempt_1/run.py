"""Frozen EI-state accessibility probes; disjoint canonical schedule templates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time,json,hashlib,itertools,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import rankdata
from WorkingMemory.stimuli import SequenceStream,setting,TASK_CLASSES,duration_oracle
from WorkingMemory.RecurrentComparison.model import RecurrentOpponent
OUT=Path(__file__).resolve().parent;SEED=16951001;START=time.time()
CP=ROOT/'WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/remote_retrieval/remote_results/ei_adaptive/checkpoint_005000.pt'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checktime():
    if time.time()-START>1120:raise TimeoutError('Reserve80seconds for receipts; no extension')
def schedules():
    rng=np.random.default_rng(SEED);bank=[]
    for tail in itertools.product(range(4),repeat=7):
        s=(0,)+tail;c=np.bincount(s,minlength=4)
        if (c==c.max()).sum()==1:bank.append(s)
    rng.shuffle(bank);n=len(bank);chunks={'train':bank[:int(.7*n)],'val':bank[int(.7*n):int(.85*n)],'test':bank[int(.85*n):]}
    result={}
    for split,want in [('train',384),('val',96),('test',128)]:
        available=list(chunks[split]);rng.shuffle(available);used=set();pairs=[]
        for a in available:
            if a in used:continue
            candidates=[b for b in available if b not in used and b!=a and b[-2:]==a[-2:] and duration_oracle(b)[0]!=duration_oracle(a)[0]
                and not np.array_equal(np.bincount(a[:4],minlength=4),np.bincount(b[:4],minlength=4))]
            if not candidates:continue
            b=candidates[int(rng.integers(len(candidates)))];used.update((a,b));pairid=len(pairs)
            pairs.append((a,b))
            if len(pairs)==want:break
        assert len(pairs)==want
        records=[]
        for group,(a,b) in enumerate(pairs):
            for rotation in range(4):
                records.append(dict(template_group=group,rotation=rotation,render_seed=SEED+{'train':10000,'val':20000,'test':30000}[split]+len(records),
                    schedules=[[(v+rotation)%4 for v in s] for s in (a,b)]))
        result[split]=records
    # Every individual template and all its cyclic rotations belong to exactly one split.
    sets={k:{tuple((x-s[0])%4 for x in s) for r in rs for s in r['schedules']} for k,rs in result.items()}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    return result
def extract(model,records,split):
    chunks={k:[] for k in ('prefix_r','prefix_ra','final_r','final_ra','sensory','sensory_r','counts','y','existing','template_group','pair_group')};metadata=[]
    for offset in range(0,len(records),4):
        xs=[];ys=[];targets=[]
        for j,record in enumerate(records[offset:offset+4]):
            stream=SequenceStream(record['render_seed'],split='test' if split=='test' else ('val' if split=='val' else 'train'));state=copy.deepcopy(stream.state_dict());movies=[]
            for s in record['schedules']:
                stream.load_state_dict(state);x,y,m=stream.batch(1,'motion_direction',setting(length=8,delay=0,directions=s))
                movies.append(x[0]);ys.append(int(y[0]));targets.append(np.bincount(s[:4],minlength=4)[:3]-1.)
                metadata.append(dict(**record,directions=s,label=int(y[0]),pair_group=offset+j,metadata=m[0],movie_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest()))
            assert torch.equal(movies[0][-3:],movies[1][-3:]) and torch.equal(movies[0][0],movies[1][0])
            xs.extend(movies);chunks['template_group'].extend([record['template_group']]*2);chunks['pair_group'].extend([offset+j]*2)
        x=torch.stack(xs).cuda();state=None;start=time.time()
        with torch.inference_mode():
            for t in range(11):
                (field,h,z),state,_=model.step(x[:,t],state)
                if t==5:
                    chunks['prefix_r'].append(state[1][0].cpu().numpy());chunks['prefix_ra'].append(torch.cat(state[1],1).cpu().numpy())
            r,a=state[1];sensory=model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
            chunks['final_r'].append(r.cpu().numpy());chunks['final_ra'].append(torch.cat((r,a),1).cpu().numpy());chunks['sensory'].append(sensory.cpu().numpy());chunks['sensory_r'].append(torch.cat((sensory,r),1).cpu().numpy())
            logits=model.classify(sensory+model.memory_output(r),'motion_direction')
            chunks['existing'].append(logits.cpu().numpy())
            if split=='train' and offset==0:
                direct=model(x,'motion_direction');assert torch.equal(logits,direct)
                save('profile.json',dict(batch_movies=len(x),seconds=time.time()-start,includes_single_forward_equivalence_check=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        chunks['y'].extend(ys);chunks['counts'].extend(targets)
        if offset%128==0:print(json.dumps(dict(stage='extract',split=split,movies=2*(offset+4),elapsed=time.time()-START)),flush=True)
        checktime()
    result={k:(np.concatenate(v) if k not in ('counts','y','template_group','pair_group') else np.asarray(v)) for k,v in chunks.items()}
    np.savez_compressed(OUT/(split+'_features.npz'),**result);save(split+'_metadata.json',metadata)
    return result
def standard(train,val,key):
    mean=train[key].mean(0);scale=train[key].std(0);scale=np.where(scale>1e-6,scale,1.)
    return ((train[key]-mean)/scale).astype(np.float64),((val[key]-mean)/scale).astype(np.float64),mean,scale
def ridge(train,val,key,target,tag):
    x,v,mean,scale=standard(train,val,key);y=train[target] if target=='counts' else np.eye(4)[train['y']];vy=val[target] if target=='counts' else np.eye(4)[val['y']]
    ym=y.mean(0);eig,u=eigh(x.T@x/len(x));rhs=u.T@(x.T@(y-ym)/len(x));traces=[];best=None
    for lam in (1e-4,1e-2,1.):
        w=u@(rhs/(np.maximum(eig,0)[:,None]+lam));pred=v@w+ym;loss=float(np.mean((pred-vy)**2));traces.append(dict(lam=lam,val_mse=loss))
        if best is None or loss<best[0]:best=(loss,w,lam)
    _,w,lam=best;np.savez(OUT/(tag+'.npz'),mean=mean,scale=scale,w=w,b=ym)
    return dict(kind='ridge',key=key,target=target,lambda_mean_mse=lam,trace=traces,mean=mean,scale=scale,w=w,b=ym)
def mlp(train,val,key,target,tag):
    x,v,mean,scale=standard(train,val,key);x=torch.tensor(x,dtype=torch.float32);v=torch.tensor(v,dtype=torch.float32)
    y=torch.tensor(train['counts'] if target=='counts' else train['y']);vy=torch.tensor(val['counts'] if target=='counts' else val['y'])
    if target=='counts':y=y.float();vy=vy.float()
    torch.random.default_generator.manual_seed(SEED+900);net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.SiLU(),torch.nn.Linear(64,3 if target=='counts' else 4))
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.MSELoss() if target=='counts' else torch.nn.CrossEntropyLoss();best=None;trace=[]
    for epoch in range(1,151):
        net.train();order=torch.randperm(len(x))
        for ix in order.split(256):
            optimizer.zero_grad();loss=criterion(net(x[ix]),y[ix]);loss.backward();optimizer.step()
        if epoch%10==0:
            net.eval()
            with torch.no_grad():value=float(criterion(net(v),vy))
            trace.append(dict(epoch=epoch,val_loss=value))
            if best is None or value<best[0]:best=(value,epoch,copy.deepcopy(net.state_dict()))
        checktime()
    net.load_state_dict(best[2]);net.eval();torch.save(dict(model=best[2],mean=mean,scale=scale,epoch=best[1]),OUT/(tag+'.pt'))
    return dict(kind='mlp64',key=key,target=target,selected_epoch=best[1],trace=trace,mean=mean,scale=scale,net=net)
def predict(fit,data):
    x=(data[fit['key']]-fit['mean'])/fit['scale']
    if fit['kind']=='ridge':return x@fit['w']+fit['b']
    with torch.no_grad():return fit['net'](torch.tensor(x,dtype=torch.float32)).numpy()
def auc(y,scores):
    values=[]
    for k in range(4):
        p=y==k;n=p.sum();r=rankdata(scores[:,k]);values.append((r[p].sum()-n*(n+1)/2)/(n*(len(y)-n)))
    return float(np.mean(values))
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.random.default_generator.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=schedules();save('schedules.json',records);digest=sha(CP)
    assert digest=='0a1d498ca678337aac42232ad1c9e048951b64ec0c0128e68b9cf0522b0b2520'
    config=dict(version='frozen_state_probe_v1',seed=SEED,budget_seconds=1200,movies={'train':3072,'val':768,'test':1024},canonical_pair_templates={'train':384,'val':96,'test':128},
        checkpoint=dict(path=str(CP),sha256=digest),count_target='first4 transition counts for directions0,1,2 minus1; fourth count determined by sum4',prefix_image_index=5,final_image_index=10,
        ridge_lambdas_mean_mse=[1e-4,1e-2,1.],mlp=dict(hidden=64,activation='SiLU',epochs=150,batch=256,lr=.001,weight_decay=.0001,val_every=10,selection='minimum validation MSE for counts / CE for duration'),
        source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'WorkingMemory/stimuli.py',ROOT/'WorkingMemory/RecurrentComparison/model.py',ROOT/'WorkingMemory/model.py']})
    save('config.json',config)
    cp=torch.load(CP,map_location='cpu');model=RecurrentOpponent(TASK_CLASSES,'ei_adaptive',activation_checkpoint=False);model.load_state_dict(cp['model'],strict=True);del cp
    model.requires_grad_(False);model.eval();model.cuda()
    train=extract(model,records['train'],'train');val=extract(model,records['val'],'val');fits={}
    for key in ('prefix_r','prefix_ra','final_r','final_ra','sensory','sensory_r'):
        fits['counts_ridge_'+key]=ridge(train,val,key,'counts','counts_ridge_'+key);checktime()
    for key in ('prefix_ra','final_ra'):
        fits['counts_mlp_'+key]=mlp(train,val,key,'counts','counts_mlp_'+key)
    fits['duration_ridge']=ridge(train,val,'sensory_r','y','duration_ridge')
    fits['duration_mlp']=mlp(train,val,'sensory_r','y','duration_mlp')
    selections={name:{k:v for k,v in fit.items() if k not in ('mean','scale','w','b','net')} for name,fit in fits.items()};save('fit_selection.json',selections)
    # Locked heldout feature collection and scoring begin only after all fits/selection are frozen.
    test=extract(model,records['test'],'test');del model;torch.cuda.empty_cache();predictions={name:predict(fit,test) for name,fit in fits.items()};predictions['existing']=test['existing'];np.savez(OUT/'heldout_predictions.npz',**predictions)
    countresults={};duration={};y=test['y'];target=test['counts'];group=test['template_group'];rng=np.random.default_rng(SEED+990)
    indices=[np.flatnonzero(group==g) for g in np.unique(group)];boots=np.array([np.concatenate([indices[i] for i in rng.integers(len(indices),size=len(indices))]) for _ in range(1000)])
    for name,p in predictions.items():
        if name.startswith('counts'):
            mse=float(np.mean((p-target)**2));d=p.reshape(-1,2,3)[:,1]-p.reshape(-1,2,3)[:,0];truth=target.reshape(-1,2,3)[:,1]-target.reshape(-1,2,3)[:,0]
            countresults[name]=dict(mse=mse,r2=1-mse/float(np.mean((target-target.mean(0))**2)),paired_difference_r2=1-float(np.sum((d-truth)**2)/np.sum(truth**2)))
        else:
            correct=(p.argmax(1)==y).astype(float);base=(test['existing'].argmax(1)==y).astype(float);diff=correct-base
            confusion=np.zeros((4,4),int);np.add.at(confusion,(y,p.argmax(1)),1)
            duration[name]=dict(ba=float(correct.mean()),macro_ovr_auc=auc(y,p),confusion=confusion.tolist(),delta_ba_vs_existing=float(diff.mean()),delta_ci95=np.quantile(diff[boots].mean(1),[.025,.975]).tolist(),ba_ci95=np.quantile(correct[boots].mean(1),[.025,.975]).tolist())
    save('summary.json',dict(status='completed',count_recovery=countresults,duration=duration,base_movies=1024,heldout_independent_template_clusters=128,
        interpretation='Paired count-difference R2 controls identical suffix; sensory path still contains opponent history. Adaptation is diagnostic-only. Duration probes use r+sensory. Failed probes do not establish erasure.',
        bootstrap='1000 canonical-pair-template cluster draws; all4rotations and paired movies remain together',wall_seconds=time.time()-START,frozen_parent_optimizer_steps=0,
        fitted_candidates=dict(ridge=21,mlp_architectures=1,mlp_target_feature_fits=3,mlp_epochs_each=150),platform=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),fp32=True,tf32=False)))
    print(json.dumps(dict(status='completed',wall_seconds=time.time()-START)),flush=True)
if __name__=='__main__':main()
