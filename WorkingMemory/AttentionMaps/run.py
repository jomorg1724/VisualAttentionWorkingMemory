"""Observe genuine joint attention weights without changing a model or task."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import sys,time,json,copy,hashlib,base64,io,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from WorkingMemory.PreUpdateAttention.model import AttentionMemory
from WorkingMemory.SpatialComparison.stimuli import SpatialStream,FAMILY_HEAD
FAMILIES=('orientation_single','orientation_binding','motion_direction')
def save(name,obj):
    p=HERE/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2),encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def png(frame):
    a=np.clip(frame.transpose(1,2,0)*255,0,255).astype('uint8');b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def weights(module,field,old):
    b=len(field);visual=field.flatten(2).transpose(1,2);memory=old.flatten(2).transpose(1,2)
    q=module.query(module.query_norm(memory)+module.position+module.source[1]);raw=torch.cat((visual,memory),1)
    identities=torch.cat((module.position+module.source[0],module.position+module.source[1]),0)
    k=module.key(module.key_norm(raw)+identities);v=module.value(raw)
    q=q.reshape(b,169,2,32).transpose(1,2);k=k.reshape(b,338,2,32).transpose(1,2);v=v.reshape(b,338,2,32).transpose(1,2)
    bias=module.source_bias.repeat_interleave(169,dim=1)[:,None,:]-F.softplus(module.raw_locality)[:,None,None]*module.distance_squared
    w=(q@k.transpose(-1,-2)/math.sqrt(32)+bias[None]).softmax(-1)
    result=module.output((w@v).transpose(1,2).reshape(b,169,64)).transpose(1,2).reshape(b,64,13,13)
    return result,w
def stages(f,d):
    if f=='motion_direction':return {'instruction':[0],'moving':[2,3,4,5,6,7,8,9],**({'blank':list(range(10,10+d))} if d else {}),'report':[10+d]}
    return {'instruction':[0],'sample':[1,2],**({'blank':list(range(3,3+d))} if d else {}),'query':[3+d],'probe':[4+d]}
def condition(f,m):
    if f=='orientation_binding':return 'locations '+json.dumps(m['positions_xy'],separators=(',',':'))
    if f=='motion_direction':return 'winner '+['right','up','left','down'][m['label']]
    return 'unchanged' if m['label']==0 else 'changed'
@torch.no_grad()
def capture(model,x,f,d,meta,first=False):
    traces=();state=None;summaries={};samples={};phase=stages(f,d);lookup={t:k for k,ids in phase.items() for t in ids}
    for t in range(x.shape[1]):
        res=model._sensory(x[:,t],*traces);field,traces=res[0],res[1:];old=torch.zeros_like(field) if state is None else state[0]
        drive,w=weights(model.attention,field,old);local=model.comparator(torch.cat((old,field),1));comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1)
        r,state,_=model.memory(model.memory_input(drive),state,False)
        if t in lookup:
            tag=lookup[t];split=w.reshape(len(x),2,169,2,169);key=split.mean(2);receiver=split.sum(-1).permute(0,1,3,2)
            rec=summaries.setdefault(tag,dict(key=[],receiver=[]));rec['key'].append(key.cpu().numpy());rec['receiver'].append(receiver.cpu().numpy())
            if t==phase[tag][-1]:
                # Store one scene per batch for group references; weights only first trial/family.
                samples[tag]=dict(time=t,current=x[:,t].cpu().numpy(),reference=x[:,9 if f=='motion_direction' else 2].cpu().numpy())
                if first:samples[tag]['weights']=w[0].cpu().numpy()
    sensory=model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1));memory=model.memory_output(torch.cat((r.mean((2,3)),r.amax((2,3))),1))
    logits=model.classify(sensory+memory+model.comparison_output(comp),FAMILY_HEAD[f]).cpu()
    return {tag:{k:np.mean(v,axis=0) for k,v in rec.items()} for tag,rec in summaries.items()},samples,logits
def main():
    start=time.time();deadline=start+1800;save('budget.json',dict(started=start,deadline=deadline,cap_seconds=1800,reuses_pending_latent_allocation=True,status='capturing'))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((ROOT/'WorkingMemory/PreUpdateAttention/retrieval_receipt.json').read_text());cp=Path(receipt['results'])/'preupdate_attention/checkpoint_008400.pt';digest=sha(cp)
    model=AttentionMemory(activation_checkpoint=False);model.load_state_dict(torch.load(cp,map_location='cpu')['model'],strict=True);model.eval().requires_grad_(False).cuda()
    # The wrapper returns the unmodified network's logits with only observation added.
    x,y,m=SpatialStream(58973001,'test').batch(8,'orientation_single',dict(delay=24));tick=time.time();_,_,a=capture(model,x.cuda(),'orientation_single',24,m);torch.cuda.synchronize();cost=time.time()-tick
    with torch.no_grad():b=model(x.cuda(),'orientation').cpu()
    error=float((a-b).abs().max());assert torch.allclose(a,b,atol=3e-5,rtol=3e-5),error
    n=64 if cost*64/8*114/29*1.5<1100 else 32
    if cost*n/8*114/29*1.5>1100:raise RuntimeError('Capture cannot fit original pending allowance')
    cfg=dict(checkpoint=str(cp),checkpoint_sha256=digest,step=8400,episodes_per_family=n,delays=[0,24],seed=58973021,paired_delays=True,full_weight_shape='B x 2 heads x 169 receiving queries x 338 keys [169 visual,169 memory]',profile_seconds=cost,wrapper_max_abs=error,model_frozen=True,main_model_experiments=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'render.py',HERE/'supervise.py',ROOT/'WorkingMemory/PreUpdateAttention/model.py']})
    save('config.json',cfg);groups={};records=[];pertrial_key=[];pertrial_receiver=[];representatives=[];full={};behavior=[]
    for f in FAMILIES:
        s=SpatialStream(cfg['seed'],'test')
        for offset in range(0,n,8):
            state=copy.deepcopy(s.state_dict());stable=None
            for d in (0,24):
                s.load_state_dict(state);x,y,m=s.batch(8,f,dict(delay=d,spacing='mixed'));evidence=torch.cat((x[:,:10],x[:,-1:]),1) if f=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1)
                if stable is None:stable=evidence
                else:assert torch.equal(stable,evidence)
                data,scenes,logits=capture(model,x.cuda(),f,d,m,first=offset==0 and d==24)
                for i in range(8):behavior.append(dict(family=f,delay=d,episode=offset+i,label=int(y[i]),prediction=int(logits[i].argmax())))
                for phase,vals in data.items():
                    for i in range(8):
                        cond=condition(f,m[i]);key=f'{f}|D{d}|{phase}|{cond}';g=groups.setdefault(key,dict(family=f,delay=d,phase=phase,condition=cond,n=0,key_sum=np.zeros((2,2,169)),receiver_sum=np.zeros((2,2,169)),current=png(scenes[phase]['current'][i]),reference=png(scenes[phase]['reference'][i]),reference_caption='Earlier final moving frame (scene reference)' if f=='motion_direction' else 'Earlier sample frame (remembered-scene reference)'))
                        g['n']+=1;g['key_sum']+=vals['key'][i];g['receiver_sum']+=vals['receiver'][i]
                        records.append(dict(family=f,delay=d,phase=phase,condition=cond,episode=offset+i,group=f'{f}/{offset+i}',source_mass=vals['key'][i].sum(-1).tolist()));pertrial_key.append(vals['key'][i]);pertrial_receiver.append(vals['receiver'][i])
                    if offset==0 and d==24:
                        q=scenes[phase];name=f'{f}_{phase}';full[name]=q['weights'];representatives.append(dict(id=name,family=f,phase=phase,time=q['time'],condition=condition(f,m[0]),current=png(q['current'][0]),reference=png(q['reference'][0]),weights=base64.b64encode(q['weights'].astype('<f4').tobytes()).decode()))
                if time.time()>deadline-240:raise TimeoutError('Capture reached original rendering reserve')
            print(json.dumps(dict(stage='capture',family=f,base_episodes=offset+8,elapsed=time.time()-start)),flush=True)
    averaged=[]
    for key,g in groups.items():
        g['key_map']=(g.pop('key_sum')/g['n']).tolist();g['receiver_map']=(g.pop('receiver_sum')/g['n']).tolist();g['source_mass']=np.array(g['key_map']).sum(-1).tolist();averaged.append(dict(id=key,**g))
    np.savez_compressed(HERE/'pertrial_maps.npz',key=np.array(pertrial_key),receiver=np.array(pertrial_receiver));np.savez_compressed(HERE/'representative_full_weights.npz',**full)
    save('pertrial_metadata.json',records);save('condition_maps.json',averaged);save('representatives.json',representatives);save('behavior.json',behavior)
    assert sha(cp)==digest;del model;torch.cuda.empty_cache()
    from render import main as render
    render(averaged,representatives,records,cfg)
    save('completion_receipt.json',dict(status='completed',elapsed_seconds=time.time()-start,checkpoint_unchanged=True,condition_groups=len(averaged),paired_base_episodes=3*n,total_presentations=6*n,no_training_or_interventions=True,deadline=deadline))
if __name__=='__main__':main()
