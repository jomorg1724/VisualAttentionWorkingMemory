"""Finite frozen-model GPU diagnostic, cached identical sensory fields per model."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from common import *
import time,copy,threading
import numpy as np
import torch
from diagnostic import forward,sensory_fields,stages
from WorkingMemory.SpatialComparison.stimuli import SpatialStream,FAMILY_HEAD

def variants(family,delay):
    if family=='orientation_single':return [('baseline',None)]+[('exclude_'+s,s) for s in ('sample','blanks','query','probe') if s!='blanks' or delay]
    return [('baseline',None),('exclude_moving','moving'),('bypass_moving','moving')]

def evaluate_batch(model,arm,x,family,delay):
    fields=sensory_fields(model,x.cuda());phase=stages(family,delay);answers={}
    for name,stage in variants(family,delay) if arm=='attention' else [('baseline',None)]:
        exclusion=phase[stage] if name.startswith('exclude') else ()
        bypass=phase[stage] if name.startswith('bypass') else ()
        answers[name]=forward(model,fields,FAMILY_HEAD[family],exclude_frames=exclusion,bypass_frames=bypass)
    return answers

def run():
    root=HERE/'run';root.mkdir(exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('No automatic budget renewal')
    review=read(HERE/'review_ready.json');assert review['status']=='passed'
    for name,digest in review['source_hashes'].items():assert sha(ROOT/name)==digest,name
    modelset=models();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    started=time.time();deadline=started+1800
    timer=threading.Timer(1800,lambda:os._exit(124));timer.daemon=True;timer.start()
    ledger=dict(status='profiling',started_unix=started,deadline_unix=deadline,source_hashes=review['source_hashes'],checkpoint_sha256={a:sha(p) for a,p in paths().items()},events=[])
    write(root/'budget.json',ledger)
    for model in modelset.values():model.cuda()
    try:
        torch.cuda.reset_peak_memory_stats();profile=[]
        for family in ('orientation_single','motion_direction'):
            for delay in (0,24):
                x,y,m=SpatialStream(47973001,'test').batch(8,family,dict(delay=delay,spacing='mixed'));tick=time.time()
                for arm,model in modelset.items():evaluate_batch(model,arm,x,family,delay)
                torch.cuda.synchronize();profile.append(dict(family=family,delay=delay,seconds=time.time()-tick))
        # Identical family evidence prefixes pair D0/D24; orientationD0 is a128-example anchor.
        reserve=180
        def estimate(n):return 1.4*sum(p['seconds']*(min(n,128) if p['family']=='orientation_single' and p['delay']==0 else n)/8 for p in profile)+reserve
        n=next((n for n in (512,384,256,128) if estimate(n)<deadline-time.time()-30),None)
        if n is None:raise RuntimeError('No finite prefix fits cap')
        cfg=dict(n_primary=n,n_orientation_D0=min(n,128),batch_size=8,test_seed=46973001,profile=profile,estimated_seconds=estimate(n),peak_allocated_bytes=torch.cuda.max_memory_allocated(),stages={f'{f}_D{d}':stages(f,d) for f in ('orientation_single','motion_direction') for d in (0,24)},teaching='unchanged frozen inference only',one_gpu_worker=True)
        write(root/'fixed_config.json',cfg);ledger['status']='evaluating';write(root/'budget.json',ledger)
        oldrows={}
        remote=Path(read(HERE.parent/'retrieval_receipt.json')['results'])
        for line in (remote/'test/predictions.jsonl').read_text().splitlines():
            r=json.loads(line);oldrows.setdefault(r['condition'],[]).append(r)
        for rows in oldrows.values():rows.sort(key=lambda r:int(r['paired_base_id'].split('/')[-1]))
        with (root/'predictions.jsonl').open('w') as output:
            for family in ('orientation_single','motion_direction'):
                stream=SpatialStream(cfg['test_seed'],'test')
                for offset in range(0,n,8):
                    state=copy.deepcopy(stream.state_dict());reference=None
                    for delay in (0,24):
                        stream.load_state_dict(state);x,y,meta=stream.batch(8,family,dict(delay=delay,spacing='mixed'))
                        stable=torch.cat((x[:,:10],x[:,-1:]),1) if family=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1)
                        if reference is None:reference=(stable,y)
                        else:assert torch.equal(reference[0],stable) and torch.equal(reference[1],y)
                        if family=='orientation_single' and delay==0 and offset>=cfg['n_orientation_D0']:continue
                        cell=('single' if family=='orientation_single' else 'motion')+f'_D{delay}';original=oldrows[cell][offset:offset+8]
                        assert all(r['label']==int(label) and r['metadata']==m for r,label,m in zip(original,y,meta))
                        phase=stages(family,delay)
                        for arm,model in modelset.items():
                            answers=evaluate_batch(model,arm,x,family,delay)
                            for name,a in answers.items():
                                probabilities=a['logits'].softmax(1).tolist()
                                for i,(label,m) in enumerate(zip(y.tolist(),meta)):
                                    stats=None if a['attention_stats'] is None else {s:a['attention_stats'][i,ids].mean(0).tolist() for s,ids in phase.items() if ids}
                                    r=dict(arm=arm,variant=name,cell=cell,family=family,delay=delay,episode=offset+i,label=label,probabilities=probabilities[i],logits=a['logits'][i].tolist(),branch_logits=a['parts'][i].tolist(),head_bias=a['head_bias'].tolist(),branch_reconstruction_max_abs=a['branch_reconstruction_max_abs'],stage_attention=stats,metadata=m)
                                    output.write(json.dumps(r)+'\n')
                            output.flush()
                        if time.time()>deadline-150:raise TimeoutError('Diagnostic evaluation reserve reached')
                    if offset%64==0:
                        event=dict(family=family,episodes=offset+8,elapsed_seconds=time.time()-started);print(json.dumps(event),flush=True);ledger['events'].append(event);write(root/'budget.json',ledger)
        # Frozen source and checkpoint hashes must still match; no original tensor is trained.
        assert all(not p.requires_grad for m in modelset.values() for p in m.parameters())
        assert all(sha(p)==ledger['checkpoint_sha256'][a] for a,p in paths().items())
        ledger['status']='evaluated';write(root/'budget.json',ledger)
        del modelset;torch.cuda.empty_cache()
        from analyze import main
        main();ledger['status']='completed'
    except BaseException as e:ledger.update(status='failed',error=repr(e));raise
    finally:
        ledger['elapsed_seconds']=time.time()-started;write(root/'budget.json',ledger);write(root/'exit.json',dict(status=ledger['status'],error=ledger.get('error'),elapsed_seconds=ledger['elapsed_seconds'],deadline_unix=deadline));timer.cancel()
if __name__=='__main__':run()
