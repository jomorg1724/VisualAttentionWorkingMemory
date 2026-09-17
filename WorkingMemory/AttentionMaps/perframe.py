"""Capture every frame of unchanged frozen-model trials within the original allowance."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import sys,json,time,copy,threading,base64
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from WorkingMemory.AttentionMaps.run import weights,png,sha,save
from WorkingMemory.PreUpdateAttention.model import AttentionMemory
from WorkingMemory.SpatialComparison.stimuli import SpatialStream,FAMILY_HEAD
def phase(f,d,t):
    if t==0:return 'instruction'
    if f=='motion_direction':
        if t==1:return 'initial dot reference'
        if t<10:return f'motion transition {t-1}/8'
        return 'report' if t==10+d else f'blank {t-9}/{d}'
    if t<=2:return f'sample {t}/2'
    if t<3+d:return f'blank {t-2}/{d}'
    return 'query cue' if t==3+d else 'comparison probe / report'
def packed(a):return base64.b64encode(a.astype('<f4').tobytes()).decode()
@torch.no_grad()
def main():
    start=time.time();deadline=json.loads((HERE/'budget.json').read_text())['deadline'];remaining=deadline-start
    if remaining<90:raise TimeoutError('Original attention visualization budget has insufficient time; not renewed')
    timer=threading.Timer(remaining,lambda:os._exit(124));timer.daemon=True;timer.start()
    cfg=json.loads((HERE/'config.json').read_text());cp=Path(cfg['checkpoint']);assert sha(cp)==cfg['checkpoint_sha256']
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AttentionMemory(activation_checkpoint=False);model.load_state_dict(torch.load(cp,map_location='cpu')['model'],strict=True);model.eval().requires_grad_(False).cuda();movies=[];arrays={};max_error=0.
    for f in ('orientation_single','orientation_binding','motion_direction'):
        stream=SpatialStream(cfg['seed'],'test');initial=copy.deepcopy(stream.state_dict());stable=None
        for delay in (0,24):
            stream.load_state_dict(initial);x,y,metadata=stream.batch(8,f,dict(delay=delay,spacing='mixed'));evidence=torch.cat((x[:,:10],x[:,-1:]),1) if f=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1)
            if stable is None:stable=evidence
            else:assert torch.equal(stable,evidence)
            gx=x.cuda();traces=();state=None;batchmovies=[dict(id=f'{f}_D{delay}_trial{i}',family=f,delay=delay,trial=i,label=int(y[i]),metadata=metadata[i],frames=[]) for i in range(8)];keys=[];receivers=[]
            for t in range(x.shape[1]):
                res=model._sensory(gx[:,t],*traces);field,traces=res[0],res[1:];old=torch.zeros_like(field) if state is None else state[0]
                drive,w=weights(model.attention,field,old);local=model.comparator(torch.cat((old,field),1));comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1);r,state,_=model.memory(model.memory_input(drive),state,False)
                split=w.reshape(8,2,169,2,169);key=split.mean(2).cpu().numpy();receiver=split.sum(-1).permute(0,1,3,2).cpu().numpy();max_error=max(max_error,float(np.max(np.abs(receiver.sum(2)-1))));keys.append(key);receivers.append(receiver)
                ref=1 if f=='motion_direction' else 2
                for i in range(8):batchmovies[i]['frames'].append(dict(t=t,phase=phase(f,delay,t),current=png(x[i,t].numpy()),reference=png(x[i,ref].numpy()) if t>=ref else None,reference_label='Initial dot reference (frame1)' if f=='motion_direction' else 'Earlier sample (frame2)',key=packed(key[i]),receiver=packed(receiver[i])))
            sensory=model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1));memory=model.memory_output(torch.cat((r.mean((2,3)),r.amax((2,3))),1));logits=model.classify(sensory+memory+model.comparison_output(comp),FAMILY_HEAD[f]).cpu()
            for i,movie in enumerate(batchmovies):movie['prediction']=int(logits[i].argmax());movies.append(movie)
            arrays[f'{f}_D{delay}_key']=np.stack(keys,axis=1);arrays[f'{f}_D{delay}_receiver']=np.stack(receivers,axis=1)
            print(json.dumps(dict(family=f,delay=delay,trials=8,frames_per_trial=x.shape[1],elapsed=time.time()-start)),flush=True)
    del model;torch.cuda.empty_cache();np.savez_compressed(HERE/'perframe_maps.npz',**arrays);save('perframe_movies.json',movies)
    from frame_view import render
    render(movies,cfg)
    save('perframe_receipt.json',dict(status='completed',started=start,elapsed_seconds=time.time()-start,original_deadline=deadline,remaining_at_start=remaining,additional_budget_seconds=0,episodes_per_family=8,paired_delays=[0,24],movies=len(movies),frame_observations=sum(len(m['frames']) for m in movies),temporal_averaging=False,trial_averaging=False,source_sum_max_abs_error=max_error,checkpoint_unchanged=sha(cp)==cfg['checkpoint_sha256'],no_training_or_interventions=True));timer.cancel()
if __name__=='__main__':main()
