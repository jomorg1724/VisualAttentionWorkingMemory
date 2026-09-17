"""Frozen framewise observations; analysis metadata does not affect pixels or model inputs."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='2'
import sys,json,time,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from WorkingMemory.SpatialComparison.stimuli import SpatialStream,FAMILY_HEAD
from WorkingMemory.RecurrentComparison.OrientationDiagnostic.run import RecordedStream
from WorkingMemory.PreUpdateAttention.model import AttentionMemory
from WorkingMemory.PreUpdateAttention.MechanismDiagnostic.diagnostic import attention
SEED=57973001
FAMILIES=('orientation_single','orientation_binding','motion_direction')
def save(path,x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stream(seed,split):
    s=SpatialStream(seed,split);s.native['orientation_single']=RecordedStream(seed+100003,split);return s
def phase(f,d,t):
    if t==0:return 'instruction'
    if f=='motion_direction':
        if t==1:return 'reference'
        if t<10:return 'moving'
        return 'report' if t==10+d else 'blank'
    if t<3:return 'sample'
    if t==3+d:return 'query'
    if t==4+d:return 'probe'
    return 'blank'
@torch.no_grad()
def observe(model,x,family,metadata,delay,split,offset):
    traces=();state=None;values={k:[] for k in ('H','R','A','output','attention','channel_R','site_R')};rows=[]
    for t in range(x.shape[1]):
        # Trace history is forwarded exactly as in AttentionMemory.forward.
        res=model._sensory(x[:,t],*traces)
        field,traces=res[0],res[1:];old=torch.zeros_like(field) if state is None else state[0]
        attended,stats=attention(model.attention,field,old)
        local=model.comparator(torch.cat((old,field),1));comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1)
        r,state,_=model.memory(model.memory_input(attended),state,False)
        sensory=model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
        memory=model.memory_output(torch.cat((r.mean((2,3)),r.amax((2,3))),1))
        output=sensory+memory+model.comparison_output(comp)
        for k,v in [('H',field),('R',r),('A',state[1])]:
            assert torch.isfinite(v).all(),k
            values[k].append(F.adaptive_avg_pool2d(v,(3,3)).flatten(1).cpu().numpy())
        for k,v in [('output',output),('attention',stats[:,:,0]),('channel_R',r.mean((2,3))),('site_R',r.mean(1).flatten(1))]:
            assert torch.isfinite(v).all(),k
            values[k].append(v.cpu().numpy())
        for i,m in enumerate(metadata):
            row=dict(split=split,family=family,delay=delay,episode=offset+i,group=f'{split}/{family}/{offset+i}',time=t,phase=phase(family,delay,t),label=int(m['label']),sample_angle=None,right_angle=None,direction=-1,final_direction=-1,counts=[0,0,0,0],evidence_steps=0)
            if family=='orientation_single':row['sample_angle']=m['actual_sample_theta']
            elif family=='orientation_binding':row.update(sample_angle=m['sample_angles_radians'][0],right_angle=m['sample_angles_radians'][1])
            else:
                dirs=m['directions'];n=max(0,min(8,t-1));row.update(direction=dirs[n-1] if n else -1,final_direction=dirs[-1],counts=np.bincount(dirs[:n],minlength=4).tolist(),evidence_steps=n)
            counts=sorted(row['counts']);row['evidence_margin']=(counts[-1]-counts[-2])/8
            rows.append(row)
    logits=model.classify(output,FAMILY_HEAD[family]).cpu()
    return {k:np.concatenate(v) for k,v in values.items()},rows,logits
def main():
    started=time.time();deadline=started+1800;save(HERE/'budget.json',dict(started=started,deadline=deadline,status='extracting',cap_seconds=1800))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((ROOT/'WorkingMemory/PreUpdateAttention/retrieval_receipt.json').read_text());cp=Path(receipt['results'])/'preupdate_attention/checkpoint_008400.pt'
    digest=sha(cp);weights=torch.load(cp,map_location='cpu');model=AttentionMemory(activation_checkpoint=False);model.load_state_dict(weights['model'],strict=True);del weights
    model.eval().requires_grad_(False).cuda();torch.cuda.reset_peak_memory_stats()
    # One ordinary-wrapper equivalence/pixel check and one bounded throughput profile.
    s=stream(SEED,'train');state=copy.deepcopy(s.state_dict());x,y,m=s.batch(8,'orientation_single',dict(delay=24));ordinary=SpatialStream(SEED,'train');ordinary.load_state_dict(state);ox,oy,_=ordinary.batch(8,'orientation_single',dict(delay=24))
    assert torch.equal(x,ox) and torch.equal(y,oy) and repr(s.state_dict())==repr(ordinary.state_dict())
    tick=time.time();_,_,a=observe(model,x.cuda(),'orientation_single',m,24,'train',0);torch.cuda.synchronize();cost=time.time()-tick
    with torch.no_grad():b=model(x.cuda(),'orientation').cpu()
    delta=float((a-b).abs().max());assert torch.allclose(a,b,atol=3e-5,rtol=3e-5),delta
    n=next((n for n in (256,192,128,64) if cost/8*(n+n//4+n//2)*114/29*1.6<1000),None)
    if n is None:raise RuntimeError('No bounded pilot fits extraction reserve')
    sizes=dict(train=n,val=n//4,test=n//2)
    config=dict(checkpoint=str(cp),checkpoint_sha256=digest,step=8400,sizes=sizes,seed=SEED,delays=[0,24],families=FAMILIES,features='H/R/A average 3x3 spatial bins x64 channels; output128; R channel64 and site169 means; lossless fp32 cache',profile_seconds_8_D24=cost,equivalence_max_abs=delta,pixels_labels_rng_preserved=True,main_weights_frozen=True,deadline=deadline,umap='not installed; use PCA and tSNE without modifying training environment')
    config['source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.py'))+[ROOT/'WorkingMemory/PreUpdateAttention/model.py',ROOT/'WorkingMemory/SpatialComparison/model.py',ROOT/'WorkingMemory/SpatialComparison/stimuli.py',ROOT/'WorkingMemory/stimuli.py']}
    save(HERE/'config.json',config)
    for split,n in sizes.items():
        for family in FAMILIES:
            chunks={d:{} for d in (0,24)};rows={d:[] for d in (0,24)};metas=[];pred=[];s=stream(SEED+{'train':0,'val':10,'test':20}[split],split)
            for offset in range(0,n,8):
                start=copy.deepcopy(s.state_dict());stable=None
                for d in (0,24):
                    s.load_state_dict(start);x,y,m=s.batch(min(8,n-offset),family,dict(delay=d,spacing='mixed'))
                    evidence=torch.cat((x[:,:10],x[:,-1:]),1) if family=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1)
                    if stable is None:stable=evidence
                    else:assert torch.equal(stable,evidence)
                    vals,rs,logits=observe(model,x.cuda(),family,m,d,split,offset)
                    for k,v in vals.items():chunks[d].setdefault(k,[]).append(v)
                    rows[d].extend(rs)
                    pred.extend(dict(group=f'{split}/{family}/{offset+i}',delay=d,label=int(y[i]),prediction=int(logits[i].argmax()),logits=logits[i].tolist()) for i in range(len(y)))
                    if d==0:metas.extend(m)
                if time.time()>deadline-550:raise TimeoutError('Finite extraction reserve reached')
            for d in (0,24):
                np.savez_compressed(HERE/f'{split}_{family}_D{d}.npz',**{k:np.concatenate(v) for k,v in chunks[d].items()});save(HERE/f'{split}_{family}_D{d}_rows.json',rows[d])
            save(HERE/f'{split}_{family}_metadata.json',metas);save(HERE/f'{split}_{family}_behavior.json',pred)
            print(json.dumps(dict(stage='extracted',split=split,family=family,base_episodes=n,elapsed=time.time()-started)),flush=True)
    assert sha(cp)==digest;save(HERE/'extraction_receipt.json',dict(status='completed',seconds=time.time()-started,checkpoint_unchanged=True,peak_gpu_bytes=torch.cuda.max_memory_allocated(),config=config))
    del model;torch.cuda.empty_cache()
    from analyze import main as analyze
    analyze(deadline)
    save(HERE/'completion_receipt.json',dict(status='completed',seconds=time.time()-started,checkpoint_unchanged=sha(cp)==digest,deadline=deadline))
if __name__=='__main__':main()
