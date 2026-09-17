"""Frozen, paired temporal-order diagnostic. No fitting or optimizer."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import sys,time,json,hashlib,itertools,collections,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from WorkingMemory.stimuli import SequenceStream,setting,TASK_CLASSES,duration_oracle
from WorkingMemory.RecurrentComparison.model import RecurrentOpponent
OUT=Path(__file__).resolve().parent
RUN=ROOT/'WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158'
SEED=14973001
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def center(s):return float(np.flatnonzero(np.asarray(s)==0).mean())
def switches(s):return sum(a!=b for a,b in zip(s,s[1:]))
def bank():
    groups=collections.defaultdict(list)
    for s in itertools.product(range(4),repeat=8):
        c=tuple(s.count(i) for i in range(4))
        if c[0]<=max(c[1:]):continue
        groups[(c,s[0],s[-2:],switches(s))].append(s)
    options={False:[],True:[]}
    for key,seqs in sorted(groups.items()):
        seqs.sort(key=lambda s:(center(s),s))
        a,b=seqs[0],seqs[-1]
        if center(b)>center(a):options[a[-1]==0].append((a,b))
    rng=np.random.default_rng(SEED)
    selected={flag:[choices[i] for i in rng.choice(len(choices),64,replace=len(choices)<64)] for flag,choices in options.items()}
    pairs=[]
    for winner in range(4):
        for flag in (False,True):
            for early,late in selected[flag]:
                a=[(x+winner)%4 for x in early];b=[(x+winner)%4 for x in late]
                wa,ca=duration_oracle(a);wb,cb=duration_oracle(b)
                assert wa==wb==winner and ca==cb and a[0]==b[0] and a[-2:]==b[-2:] and switches(a)==switches(b)
                ce=float(np.flatnonzero(np.asarray(a)==winner).mean());cl=float(np.flatnonzero(np.asarray(b)==winner).mean())
                assert cl>ce
                counts=sorted(ca,reverse=True);i=len(pairs)
                pairs.append(dict(group=i,render_seed=SEED+10000+i,winner=winner,early=a,late=b,counts=ca,
                    count_margin=counts[0]-counts[1],final_is_winner=flag,switch_count=switches(a),
                    early_center=ce,late_center=cl,center_shift=cl-ce))
    return pairs,{str(k):len(v) for k,v in options.items()}
def render(pair):
    stream=SequenceStream(pair['render_seed'],split='test');state=copy.deepcopy(stream.state_dict())
    images=[];meta=[]
    for name in ('early','late'):
        stream.load_state_dict(state)
        x,y,m=stream.batch(1,'motion_direction',setting(length=8,delay=0,directions=pair[name]))
        assert int(y[0])==pair['winner'] and x.shape[1]==11
        images.append(x[0]);meta.append(m[0])
    # Last two evidence images, plus common report. The instruction is also identical.
    assert torch.equal(images[0][-3:],images[1][-3:])
    assert torch.equal(images[0][0],images[1][0])
    return torch.stack(images),meta
def measures(logits,y):
    pred=logits.argmax(-1);correct=(pred==y[:,None]).astype(float)
    z=logits-logits.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True)
    true=logits[np.arange(len(y))[:,None],np.arange(2)[None,:],y[:,None]]
    other=logits.copy();other[np.arange(len(y))[:,None],np.arange(2)[None,:],y[:,None]]=-np.inf
    return dict(correct=correct,probability=p[np.arange(len(y))[:,None],np.arange(2)[None,:],y[:,None]],margin=true-other.max(-1))
def mean_ba(correct,y):return float(np.mean([correct[y==k].mean() for k in np.unique(y)]))
def main():
    start=time.time();save('running.json',dict(pid=os.getpid(),start_epoch=start,deadline_epoch=start+600,status='running'))
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.random.default_generator.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pairs,counts=bank();save('pairs.json',pairs)
    cps={'lstm':RUN/'lstm/checkpoint_005000.pt','ei_adaptive':RUN/'remote_retrieval/remote_results/ei_adaptive/checkpoint_005000.pt'}
    identities={}
    for arm,path in cps.items():
        digest=sha(path);index=[json.loads(line) for line in (path.parent/'checkpoint_index.jsonl').read_text().splitlines()]
        assert any(r.get('sha256')==digest and r.get('step')==5000 for r in index)
        identities[arm]=dict(path=str(path),sha256=digest)
    save('config.json',dict(version='paired_recency_L8_v1',seed=SEED,pairs=512,variants=2,arms=list(cps),batch=8,
        budget_seconds=600,order='late-minus-early',selection='min/max winner center in count/first/last2/switch-matched groups;64 bank pairs per final-status, rotated over4classes',
        feasible_banks=counts,checkpoint=identities,source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'WorkingMemory/stimuli.py',ROOT/'WorkingMemory/RecurrentComparison/model.py',ROOT/'WorkingMemory/model.py',ROOT/'PreAttentiveVision/neuroscience_stimuli.py',ROOT/'PreAttentiveVision/TemporalIntegration/accumulators.py']}))
    outputs={};render_hashes={};timings={}
    with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as out:
        for arm,path in cps.items():
            t=time.time();cp=torch.load(path,map_location='cpu');assert cp['step']==5000
            model=RecurrentOpponent(TASK_CLASSES,arm,activation_checkpoint=False);model.load_state_dict(cp['model'],strict=True)
            model.requires_grad_(False);model.eval();model.to('cuda');del cp
            torch.cuda.reset_peak_memory_stats();all_logits=[]
            for offset in range(0,len(pairs),4):
                batch=pairs[offset:offset+4];xs=[];metas=[]
                for pair in batch:
                    x,m=render(pair);digest=hashlib.sha256(x.numpy().tobytes()).hexdigest()
                    if arm=='lstm':render_hashes[pair['group']]=digest
                    else:assert render_hashes[pair['group']]==digest
                    xs.append(x);metas.append(m)
                t0=time.time()
                with torch.inference_mode():logits=model(torch.cat(xs).cuda(),'motion_direction').cpu().numpy().reshape(len(batch),2,4)
                assert np.isfinite(logits).all();all_logits.append(logits)
                if offset==0:
                    save('profile_'+arm+'.json',dict(batch_presentations=8,inference_seconds=time.time()-t0,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                for pair,lgs,meta in zip(batch,logits,metas):
                    out.write(json.dumps(dict(arm=arm,**pair,logits=lgs.tolist(),metadata=meta,movie_sha256=render_hashes[pair['group']]))+'\n')
                out.flush()
                if offset%64==0:print(json.dumps(dict(arm=arm,groups=offset+len(batch),elapsed=time.time()-start)),flush=True)
                if time.time()-start>570:raise TimeoutError('Leave30seconds for receipt; no budget extension')
            outputs[arm]=np.concatenate(all_logits);timings[arm]=dict(wall_seconds=time.time()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            del model;torch.cuda.empty_cache()
    y=np.array([p['winner'] for p in pairs]);flags=np.array([p['final_is_winner'] for p in pairs]);rng=np.random.default_rng(SEED+999)
    strata=[np.flatnonzero((y==k)&(flags==f)) for k in range(4) for f in (False,True)]
    boots=np.array([np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(2000)])
    vals={arm:measures(lgs,y) for arm,lgs in outputs.items()};summary={}
    for arm,v in vals.items():
        record={}
        for metric,a in v.items():
            d=a[:,1]-a[:,0];ci=np.quantile(d[boots].mean(1),[.025,.975]).tolist()
            record[metric]=dict(early=float(a[:,0].mean()),late=float(a[:,1].mean()),delta=float(d.mean()),delta_ci95=ci)
        record['strata']={}
        for field in ('count_margin','final_is_winner','switch_count'):
            record['strata'][field]={}
            for value in sorted(set(p[field] for p in pairs)):
                ix=np.array([p[field]==value for p in pairs]);a=v['correct'][ix];yy=y[ix]
                record['strata'][field][str(value)]=dict(n=int(ix.sum()),early_ba=mean_ba(a[:,0],yy),late_ba=mean_ba(a[:,1],yy),mean_center_shift=float(np.mean([p['center_shift'] for p in pairs if p[field]==value])))
        summary[arm]=record
    did={}
    for metric in ('correct','probability'):
        delta=(vals['ei_adaptive'][metric][:,1]-vals['ei_adaptive'][metric][:,0])-(vals['lstm'][metric][:,1]-vals['lstm'][metric][:,0])
        did[metric]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[boots].mean(1),[.025,.975]).tolist())
    checks=dict(all_oracles_matched=True,all_first_directions_matched=True,all_last2_directions_matched=True,all_switch_counts_matched=True,all_final2_evidence_rasters_and_report_exact=True,all_model_movies_sha_matched=True,all_positive_center_shifts=True,
        unique_ordered_schedule_pairs=len(set((tuple(p['early']),tuple(p['late'])) for p in pairs)),unique_individual_schedules=len(set(tuple(p[k]) for p in pairs for k in ('early','late'))),
        center_shift_min=min(p['center_shift'] for p in pairs),center_shift_mean=float(np.mean([p['center_shift'] for p in pairs])),center_shift_max=max(p['center_shift'] for p in pairs))
    save('summary.json',dict(status='completed',n_groups=512,n_movies=1024,n_model_presentations=2048,models=summary,difference_in_differences_ei_minus_lstm=did,checks=checks,timings=timings,
        uncertainty='2000 paired render-group bootstrap draws stratified by true class and final-winner status; conditional on selected schedule bank; rotations/order templates are not independent discovered mechanisms',wall_seconds=time.time()-start,
        platform=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),fp32=True,tf32=False,threads=2,optimizer_steps=0)))
    print(json.dumps(dict(status='completed',wall_seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
