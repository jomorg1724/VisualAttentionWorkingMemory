"""Frozen orientation accessibility and comparison diagnostics; no parent updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time,json,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import rankdata
from WorkingMemory.stimuli import SequenceStream,BRIDGE_CONDITIONS,TASK_CLASSES
from WorkingMemory.RecurrentComparison.model import RecurrentOpponent
OUT=Path(__file__).resolve().parent;D=(0,4,12,24);SEED=27973001
CP=ROOT/'WorkingMemory/RecurrentComparison/Retention/runs/retention_20260913_084433/retention/checkpoint_014800.pt'
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check():
    if time.time()>json.loads((OUT/'supervisor.json').read_text())['deadline_epoch']-70:raise TimeoutError('Original diagnostic cap; report reserve70s')
class RecordedStream(SequenceStream):
    def _context(self,stream,family):
        result=super()._context(stream,family);self.contexts.append(copy.deepcopy(result));return result
    def _recall(self,*args,**kwargs):
        self.contexts=[];frames,meta=super()._recall(*args,**kwargs);theta=self.contexts[meta['target_index']]['theta']
        meta['actual_sample_theta']=float((theta+np.deg2rad(7.5*meta['values'][meta['target_index']]))%np.pi)
        meta['actual_probe_theta']=float((theta+np.deg2rad(7.5*meta['probe_value']))%np.pi)
        return frames,meta
def condition(d):return dict(BRIDGE_CONDITIONS['bridge_recall'],delay=d,post_delay=0,distractor=False)
def sensory(model,field):return model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
def extract(model,split,n):
    path=OUT/(split+'_features.npz')
    if path.exists():return dict(np.load(path))
    stream=RecordedStream(SEED+{'train':1,'val':2,'test':3}[split],split);chunks={};metadata=[];tick=time.time()
    def add(key,v):chunks.setdefault(key,[]).append(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
    for offset in range(0,n,8):
        state=copy.deepcopy(stream.state_dict());reference=None
        for d in D:
            stream.load_state_dict(state);x,y,meta=stream.batch(min(8,n-offset),'orientation',condition(d));stable=torch.cat((x[:,:3],x[:,-2:]),1)
            if reference is None:reference=stable.clone();reference_y=y.clone();reference_angles=[m['actual_sample_theta'] for m in meta]
            else:assert torch.equal(stable,reference) and torch.equal(y,reference_y) and reference_angles==[m['actual_sample_theta'] for m in meta]
            if split=='train' and offset==0 and d==0:
                ordinary=SequenceStream(SEED+1,'train');ordinary.load_state_dict(state);ox,oy,om=ordinary.batch(8,'orientation',condition(0))
                assert torch.equal(ox,x) and torch.equal(oy,y)
                assert repr(ordinary.state_dict())==repr(stream.state_dict())
                save('construction_check.json',dict(instrumentation_preserves_pixels_labels_rng=True,stage_indices=dict(sample=2,endblank='2+D',preprobe='3+D'),delay_pair_rasters_checked_every_batch=True))
            with torch.inference_mode():
                gx=x.cuda();s=None
                for t in range(gx.shape[1]):
                    (field,h,z),s,_=model.step(gx[:,t],s)
                    for stage,index in [('sample',2),('endblank',2+d),('preprobe',3+d)]:
                        if t==index:
                            for kind,v in [('r',s[1][0]),('ra',torch.cat(s[1],1)),('sensory',sensory(model,field))]:add(f'{stage}_{kind}_D{d}',v)
                sf=sensory(model,field);add(f'post_D{d}',torch.cat((sf,s[1][0]),1));add(f'existing_D{d}',model.classify(sf+model.memory_output(s[1][0]),'orientation'))
                if d==0:
                    probe_field=model._sensory(gx[:,-1])[0];add('probe',sensory(model,probe_field))
            if d==0:
                add('y',y);add('sample_angle',[m['actual_sample_theta'] for m in meta]);add('probe_angle',[m['actual_probe_theta'] for m in meta]);add('mismatch',[7.5*m['mismatch'] for m in meta])
                metadata.extend(dict(m,base_group=offset+j,evidence_sha256=hashlib.sha256(stable[j].numpy().tobytes()).hexdigest()) for j,m in enumerate(meta))
        if offset==0 and split=='train':
            seconds=time.time()-tick;estimate=seconds*(2048+512+512)/8
            save('profile.json',dict(base_groups=8,presentations=32,seconds=seconds,estimated_extraction_seconds=estimate,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            if estimate>1300:raise TimeoutError('Profile does not support predeclared exposure with fitting reserve; no automatic extension')
        if offset%128==0:print(json.dumps(dict(stage='extract',split=split,base_groups=offset+8,elapsed=time.time()-START)),flush=True)
        check()
    data={k:np.concatenate(v) for k,v in chunks.items()};np.savez_compressed(path,**data);save(split+'_metadata.json',metadata);return data
def target(data,name):
    if name=='label':return np.eye(2)[data['y']]
    angle=data[name+'_angle'];return np.stack((np.cos(2*angle),np.sin(2*angle)),1)
def standard(x):
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-6,scale,1.);return mean,scale
def predict(fit,x):
    x=(x-fit['mean'])/fit['scale']
    if fit['kind']=='ridge':return x@fit['w']+fit['b']
    with torch.no_grad():return fit['net'](torch.tensor(x,dtype=torch.float32)).numpy()
def fit_probe(x,v,y,vy,tag,kind='ridge'):
    mean,scale=standard(x);x=((x-mean)/scale).astype(np.float64);v=((v-mean)/scale).astype(np.float64);trace=[]
    if kind=='ridge':
        ym=y.mean(0);eig,u=eigh(x.T@x/len(x));rhs=u.T@(x.T@(y-ym)/len(x));best=None
        for lam in (1e-4,1e-2,1.):
            w=u@(rhs/(np.maximum(eig,0)[:,None]+lam));loss=float(np.mean((v@w+ym-vy)**2));trace.append(dict(lam=lam,val_mse=loss))
            if best is None or loss<best[0]:best=(loss,w,lam)
        fit=dict(kind=kind,mean=mean,scale=scale,w=best[1],b=ym);selection=dict(lambda_mean_mse=best[2],trace=trace)
        np.savez(OUT/(tag+'.npz'),**{k:val for k,val in fit.items() if k!='kind'})
    else:
        torch.random.default_generator.manual_seed(SEED+900);net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.SiLU(),torch.nn.Linear(64,2))
        opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);tx=torch.tensor(x,dtype=torch.float32);tv=torch.tensor(v,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32);tvy=torch.tensor(vy,dtype=torch.float32);best=None
        for epoch in range(1,151):
            for ix in torch.randperm(len(tx)).split(256):
                opt.zero_grad();loss=(net(tx[ix])-ty[ix]).square().mean();loss.backward();opt.step()
            if epoch%10==0:
                with torch.no_grad():value=float((net(tv)-tvy).square().mean())
                trace.append(dict(epoch=epoch,val_mse=value))
                if best is None or value<best[0]:best=(value,epoch,copy.deepcopy(net.state_dict()))
            check()
        net.load_state_dict(best[2]);net.eval();fit=dict(kind=kind,mean=mean,scale=scale,net=net);selection=dict(epoch=best[1],trace=trace)
        torch.save(dict(model=best[2],mean=mean,scale=scale,epoch=best[1]),OUT/(tag+'.pt'))
    check();return fit,selection
def angle_errors(pred,theta):
    estimated=np.arctan2(pred[:,1],pred[:,0])/2
    return np.rad2deg(np.abs((estimated-theta+np.pi/2)%np.pi-np.pi/2))
def angle_distance(a,b):return angle_errors(a,np.arctan2(b[:,1],b[:,0])/2)
def inputs(data,key):
    if key.startswith('comparator_'):return np.concatenate((data[key.replace('comparator','preprobe_r')],data['probe']),1)
    return data[key]
def main():
    global START
    START=json.loads((OUT/'supervisor.json').read_text())['start_epoch'];torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(CP)=='b8ffa90c9e23b662ec3b16b3de2cb56ca9488b60d05a096aa5675f9081c588bc'
    index=[json.loads(x) for x in (CP.parent/'checkpoint_index.jsonl').read_text().splitlines()];assert any(x['file']==CP.name and x['sha256']==sha(CP) for x in index)
    config=dict(version='orientation_frozen_v1',samples=dict(train=2048,val=512,test=512),delays=D,seed=SEED,budget_seconds=1800,batch=8,parent=str(CP),parent_sha256=sha(CP),ridge_lambdas=[1e-4,.01,1.],mlp=dict(hidden=64,epochs=150,validation_every=10,lr=.001,weight_decay=.0001,loss='MSE: axial vector regression or one-hot classification'),angle_mlp_trigger_val_mae_degrees=3.75,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'supervise.py',ROOT/'WorkingMemory/stimuli.py',ROOT/'WorkingMemory/RecurrentComparison/model.py',ROOT/'WorkingMemory/model.py']})
    save('config.json',config);cp=torch.load(CP,map_location='cpu');assert cp['version']=='ei_retention_core_readout_v1' and cp['step']==14800
    model=RecurrentOpponent(TASK_CLASSES,'ei_adaptive',activation_checkpoint=False);model.load_state_dict(cp['model'],strict=True);del cp;model.requires_grad_(False);model.eval();model.cuda()
    train=extract(model,'train',2048);val=extract(model,'val',512);fits={};selections={};specs={}
    def add(tag,key,targ,kind='ridge'):
        fits[tag],selections[tag]=fit_probe(inputs(train,key),inputs(val,key),target(train,targ),target(val,targ),tag,kind);specs[tag]=dict(key=key,target=targ)
    # Sample stage is identical across delays: fit once, not four redundant candidates.
    for kind in ('r','ra','sensory'):add('sample_'+kind,'sample_'+kind+'_D0','sample')
    for stage in ('endblank','preprobe'):
        for d in D:
            for kind in ('r','ra','sensory'):add(f'{stage}_{kind}_D{d}',f'{stage}_{kind}_D{d}','sample')
    add('probe_angle','probe','probe')
    if float(angle_errors(predict(fits['probe_angle'],val['probe']),val['probe_angle']).mean())>3.75:add('probe_angle_mlp','probe','probe','mlp')
    probe_fit=min([name for name in ('probe_angle','probe_angle_mlp') if name in fits],key=lambda name:angle_errors(predict(fits[name],val['probe']),val['probe_angle']).mean())
    for name,key in [('sample_r','sample_r_D0'),('preprobe_r_D0','preprobe_r_D0'),('preprobe_r_D24','preprobe_r_D24')]:
        mae=float(angle_errors(predict(fits[name],val[key]),val['sample_angle']).mean())
        if mae>3.75:add(name+'_mlp',key,'sample','mlp')
    # Fixed small operational comparison, including the D0 positive control, regardless of angular MAE.
    for d in (0,24):
        add(f'comparator_D{d}',f'comparator_D{d}','label','mlp')
        vp=predict(fits[f'comparator_D{d}'],inputs(val,f'comparator_D{d}'));base=val[f'existing_D{d}'].argmax(1)==val['y']
        if float((vp.argmax(1)==val['y']).mean()-base.mean())>=.02:add(f'post_D{d}',f'post_D{d}','label')
    circular={}
    for d in (0,24):
        names=[f'preprobe_r_D{d}']+([f'preprobe_r_D{d}_mlp'] if f'preprobe_r_D{d}_mlp' in fits else [])
        selected=min(names,key=lambda name:angle_errors(predict(fits[name],val[f'preprobe_r_D{d}']),val['sample_angle']).mean())
        dist=angle_distance(predict(fits[selected],val[f'preprobe_r_D{d}']),predict(fits[probe_fit],val['probe']))
        thresholds=np.r_[-1e-9,np.unique(dist),90.];scores=[float(((dist>t)==val['y']).mean()) for t in thresholds]
        threshold=float(thresholds[int(np.argmax(scores))]);circular[str(d)]=dict(sample_fit=selected,probe_fit=probe_fit,threshold_degrees=threshold,val_ba=max(scores),tie_rule='smallest threshold among equal validation BA')
    selections['circular_comparators']=circular
    save('fit_selection.json',dict(selections=selections,specs=specs,heldout_not_accessed=True))
    test=extract(model,'test',512);del model;torch.cuda.empty_cache();preds={tag:predict(fit,inputs(test,specs[tag]['key'])) for tag,fit in fits.items()}
    for stage in ('endblank','preprobe'):
        for d in D:preds[f'transfer_sample_r_to_{stage}_D{d}']=predict(fits['sample_r'],test[f'{stage}_r_D{d}'])
    for d in D:preds[f'existing_D{d}']=test[f'existing_D{d}']
    for d in (0,24):
        choice=circular[str(d)];dist=angle_distance(predict(fits[choice['sample_fit']],test[f'preprobe_r_D{d}']),predict(fits[choice['probe_fit']],test['probe']))
        score=dist-choice['threshold_degrees'];preds[f'circular_D{d}']=np.stack((-score,score),1)
    np.savez(OUT/'heldout_predictions.npz',**preds)
    rng=np.random.default_rng(SEED+999);boots=np.stack([np.concatenate([rng.choice(np.flatnonzero(test['y']==c),sum(test['y']==c),replace=True) for c in (0,1)]) for _ in range(1000)])
    recovery={};comparison={}
    for name,p in preds.items():
        if name.startswith(('comparator','post','existing','circular')):
            correct=p.argmax(1)==test['y'];d=int(name.split('_D')[-1]);base=test[f'existing_D{d}'].argmax(1)==test['y'];difference=correct.astype(float)-base
            rank=rankdata(p[:,1]-p[:,0]);positive=test['y']==1;n=positive.sum();auc=float((rank[positive].sum()-n*(n+1)/2)/(n*(len(rank)-n)))
            conf=np.zeros((2,2),int);np.add.at(conf,(test['y'],p.argmax(1)),1)
            strata={str(v):dict(n=int((test['mismatch']==v).sum()),accuracy=float(correct[test['mismatch']==v].mean())) for v in np.unique(test['mismatch'])}
            comparison[name]=dict(ba=float(correct.mean()),ba_ci95=np.quantile(correct[boots].mean(1),[.025,.975]).tolist(),auc=auc,confusion=conf.tolist(),delta_vs_existing=float(difference.mean()),delta_ci95=np.quantile(difference[boots].mean(1),[.025,.975]).tolist(),actual_change_degrees=strata)
        else:
            truth=test['probe_angle'] if name.startswith('probe_angle') else test['sample_angle'];err=angle_errors(p,truth)
            recovery[name]=dict(mae_degrees=float(err.mean()),mae_ci95=np.quantile(err[boots].mean(1),[.025,.975]).tolist(),median_degrees=float(np.median(err)),fraction_within3_75=float((err<=3.75).mean()),fraction_within7_5=float((err<=7.5).mean()))
    save('summary.json',dict(status='completed',recovery=recovery,comparison=comparison,base_groups=512,delays_paired=True,parent_updates=0,wall_seconds=time.time()-START,fit_count=len(fits),ridge_candidates=3*sum(x['kind']=='ridge' for x in fits.values()),mlp_fits=sum(x['kind']=='mlp' for x in fits.values()),platform=dict(torch=torch.__version__,neural_dtype='fp32',ridge_dtype='float64',tf32=False)))
    print(json.dumps(dict(status='completed',elapsed=time.time()-START)),flush=True)
if __name__=='__main__':main()
