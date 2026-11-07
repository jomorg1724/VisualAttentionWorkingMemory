"""Runnable queued experiment; never executes merely by importing the module."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import torch
from .dataset import generate
from .model import MotionEncoder, contrastive_loss, distance
from SecondPass.PredictiveMotionChange.dataset import generate as comparison_pair, SPEEDS, ANGLES
from SecondPass.VariationalMotionPredictor.worker import atomic_json, cpu_tree


@torch.no_grad()
def evaluate(model, split, geometry_count, comparison_count, deadline, threshold=None):
    if split != 'val' and threshold is None:
        raise ValueError('Test evaluation requires a validation-selected threshold')
    model.eval()
    observed, targets = [], []
    for offset in range(0, geometry_count, 4):
        if time.time() >= deadline:
            raise TimeoutError('Evaluation wall cap')
        rows = [generate(i,split) for i in range(offset,min(offset+4,geometry_count))]
        first = torch.stack([r[0] for r in rows]).to('mps')
        second = torch.stack([r[1] for r in rows]).to('mps')
        observed.append(distance(model(first),model(second)).cpu())
        targets.extend(r[2]/math.pi for r in rows)
    observed = torch.cat(observed);targets = torch.tensor(targets)
    scores, labels, metadata = [], [], []
    start = 12000000 if split == 'val' else 24000000
    for offset in range(0, comparison_count, 4):
        if time.time() >= deadline:
            raise TimeoutError('Comparison evaluation wall cap')
        rows = [comparison_pair(i,split) for i in range(start+offset,start+min(offset+4,comparison_count))]
        first = torch.stack([r[0] for r in rows]).to('mps')
        second = torch.stack([r[1] for r in rows]).to('mps')
        scores.append(distance(model(first),model(second)).cpu())
        labels.extend(r[2] for r in rows);metadata.extend(r[3] for r in rows)
    scores = torch.cat(scores);labels = torch.tensor(labels)
    if threshold is None:
        sorted_scores = torch.sort(scores).values
        candidates = torch.cat((sorted_scores[:1]-1e-6,
                                .5*(sorted_scores[:-1]+sorted_scores[1:]), sorted_scores[-1:]+1e-6))
        accuracy = ((scores[:,None]>candidates[None,:])==labels[:,None].bool()).float().mean(0)
        threshold = float(candidates[accuracy.argmax()])
    predictions = scores > threshold
    cells = []
    for speed in SPEEDS:
        for angle in ANGLES:
            ix = torch.tensor([i for i,m in enumerate(metadata) if m['speed']==speed and m['angle']==angle])
            y,p,s = labels[ix],predictions[ix],scores[ix]
            difference = s[y==1][:,None]-s[y==0][None,:]
            cells.append(dict(speed=speed,angle=angle,n=len(ix),
                auc=float((difference.gt(0).float()+.5*difference.eq(0)).mean()),
                balanced_accuracy=.5*float(p[y==1].float().mean()+(~p[y==0]).float().mean())))
    return dict(geometry_n=geometry_count,comparison_n=comparison_count,
        angular_loss=float(.5*(observed-targets).square().mean()),
        collapsed_baseline_loss=float(.5*targets.square().mean()),
        distance_target_correlation=float(torch.corrcoef(torch.stack((observed,targets)))[0,1].nan_to_num()),
        mean_same_distance=float(observed[targets==0].mean()),threshold=threshold,cells=cells,
        mean_auc=sum(c['auc'] for c in cells)/6,
        mean_balanced_accuracy=sum(c['balanced_accuracy'] for c in cells)/6)


def main(run):
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'budget.json').exists():raise RuntimeError('Existing attempt; no automatic restart')
    lock=Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();deadline=start+28800
    atomic_json(run/'budget.json',dict(cap_started=start,hard_deadline=deadline,deadline=deadline-180,wall_cap_seconds=28800))
    atomic_json(run/'activation.json',dict(supervisor_pid=os.getpid(),worker_pid=os.getpid()))
    random.seed(186211);torch.manual_seed(186211);torch.mps.manual_seed(186211)
    model=MotionEncoder().to('mps');optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    shuffle=torch.Generator().manual_seed(186212)
    config=dict(fresh_whole_model=True,checkpoint_inputs=[],all_trainable=True,
        parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),
        batch_pairs=32,microbatch_pairs=4,lr=1e-4,clipping=None,precision='fp32',cpu_threads=2,
        pool_size=1000,pool_epochs=2,target_updates=25024,hard_deadline=deadline)
    atomic_json(run/'config.json',config)
    state=dict(step=0,presentations=0,unique_pairs=0,pool_index=-1,best_step=None,best_key=None,threshold=None)
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))

    def status(phase,**extra):
        atomic_json(run/'live_status.json',dict(phase=phase,pid=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**state,**extra))
    def save(best=False):
        payload=cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,config=config,
            torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),shuffle_rng=shuffle.get_state()))
        tmp=run/'latest.pt.tmp';torch.save(payload,tmp);os.replace(tmp,run/'latest.pt')
        atomic_json(run/'latest_checkpoint.json',dict(step=state['step'],optimizer_states=len(optimizer.state)))
        if best:
            tmp=run/'best.pt.tmp';os.link(run/'latest.pt',tmp);os.replace(tmp,run/'best.pt')
    def validate():
        status('validation')
        val=evaluate(model,'val',192,768,deadline-120)
        val['step']=state['step'];atomic_json(run/f"validation_{state['step']:06d}.json",val)
        key=(val['mean_auc'],-val['angular_loss'])
        better=state['best_key'] is None or key>tuple(state['best_key'])
        if better:state.update(best_key=key,best_step=state['step'],threshold=val['threshold'])
        save(best=better)
    reason='planned_complete';final=None
    try:
        while state['step']<25024:
            if stopped[0] or time.time()>=deadline-180:
                reason='signal' if stopped[0] else 'wall_cap';break
            state['pool_index']+=1;status('generating_pool')
            pool_start=state['unique_pairs']
            pool=[generate(i,'train') for i in range(pool_start,pool_start+1000)]
            state.update(unique_pairs=pool_start+1000,pool_start=pool_start)
            for epoch in range(2):
                order=torch.randperm(1000,generator=shuffle)
                state.update(epoch=epoch,order=order.tolist(),cursor=0)
                for indices in order.split(32):
                    if stopped[0] or time.time()>=deadline-180:break
                    model.train();optimizer.zero_grad(set_to_none=True);loss_value=0.
                    for chunk in indices.split(4):
                        rows=[pool[int(i)] for i in chunk]
                        clips=torch.stack([r[0] for r in rows]+[r[1] for r in rows]).to('mps')
                        encoded=model(clips);first,second=encoded.chunk(2)
                        angles=torch.tensor([r[2] for r in rows],device='mps')
                        loss=contrastive_loss(first,second,angles);weight=len(rows)/len(indices)
                        (loss*weight).backward();loss_value+=float(loss.detach())*weight
                    if not math.isfinite(loss_value) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError('Missing/nonfinite gradient')
                    optimizer.step();state['step']+=1;state['presentations']+=len(indices);state['cursor']+=len(indices)
                    row=dict(step=state['step'],presentations=state['presentations'],unique_pairs=state['unique_pairs'],loss=loss_value)
                    with (run/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    status('training',train_loss=loss_value)
                    if state['step']==1 or state['step']%250==0:save()
                    if state['step']==64 or state['step']%1000==0:validate()
                if stopped[0] or time.time()>=deadline-180:break
            del pool
        if not stopped[0] and time.time()<deadline-120:validate()
        save()
        if state['best_step'] is not None and not stopped[0]:
            selected=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(selected['model']);threshold=selected['state']['threshold'];del selected
            status('final_test');final=evaluate(model,'test',384,3072,deadline-15,threshold)
        atomic_json(run/'report.json',dict(stop_reason=reason,**state,final=final,config=config))
        status('finished',stop_reason=reason,final=final)
        atomic_json(run/'local_supervisor_result.json',dict(status='complete',step=state['step']))
    except Exception as exc:
        atomic_json(run/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        if state['step']:save()
        status('failed',error=str(exc));atomic_json(run/'local_supervisor_result.json',dict(status='failed'))
        raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);main(parser.parse_args().run)
