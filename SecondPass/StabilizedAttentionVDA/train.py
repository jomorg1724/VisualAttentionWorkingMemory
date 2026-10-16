"""Explicit-launch local 16-task continuation; never modifies the parent run."""
import argparse,copy,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from WorkingMemory.BatteryAudit.observers import auc_binary
from SecondPass.PixelContrastiveVAE.train import GPULock,atomic_json,append_json,cpu_tree,replace_with_retry,configure
from .dataset import Stream,TASKS,CATALOG,all_cells
from .model import initialize

PARENT=Path(__file__).resolve().parents[1]/'StabilizedAttentionSuite/LocalRuntime/continuation02/best.pt'

class Scheduler:
    def __init__(self,seed=201211):self.rng=random.Random(seed);self.tasks=[];self.cells={t:[] for t in TASKS}
    def next(self):
        if not self.tasks:self.tasks=list(TASKS);self.rng.shuffle(self.tasks)
        task=self.tasks.pop()
        if not self.cells[task]:self.cells[task]=[c['id'] for c in TASKS[task]['conditions']];self.rng.shuffle(self.cells[task])
        return task,self.cells[task].pop()
    def state_dict(self):return copy.deepcopy(dict(rng=self.rng.getstate(),tasks=self.tasks,cells=self.cells))

def update(model,opt,x,y,metadata,task,device='cpu',micro=4):
    model.train();opt.zero_grad(set_to_none=True);loss_sum=0.
    # E[value] = 3 under the native uniform red/green/blue draw. Fixed scaling
    # preserves trial utility ratios without changing with batch composition.
    weights=torch.tensor([m['value']/3 if task.startswith('vda') else 1. for m in metadata],device=device)
    for start in range(0,len(x),micro):
        logits,details=model(x[start:start+micro].to(device),task,True)
        ce=(F.cross_entropy(logits,y[start:start+micro].to(device),reduction='none')*weights[start:start+micro]).sum()/len(x)
        loss=ce+.05*details['prediction_loss']*len(logits)/len(x)
        loss.backward();loss_sum+=float(loss.detach())
    active=[(n,p) for n,p in model.named_parameters() if not n.startswith('heads.') or n.startswith(f'heads.{task}.')]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for _,p in active):raise FloatingPointError('Missing or nonfinite active gradient')
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in active],100,error_if_nonfinite=True);opt.step()
    return dict(loss=loss_sum,gradient_norm=float(norm))

@torch.no_grad()
def evaluate(model,device,split,count,deadline):
    stream=Stream(split);model.eval();rows=[]
    for task,cell in all_cells():
        labels=[];probs=[];meta=[]
        for _ in range(0,count,4):
            if time.time()>deadline:raise TimeoutError('Local allocation exhausted during evaluation')
            x,y,m=stream.batch(min(4,count-len(labels)),task,cell)
            p=model(x.to(device),task).softmax(-1).cpu().numpy();labels.extend(y.tolist());probs.extend(p.tolist());meta+=m
        yy=np.array(labels);pp=np.array(probs);pred=pp.argmax(1);classes=TASKS[task]['classes']
        degenerate=len(np.unique(yy))<classes
        ba=None if degenerate else float(np.mean([(pred[yy==i]==i).mean() for i in range(classes)]))
        auc=None if degenerate else float(np.mean([auc_binary((yy==i).astype(int),pp[:,i]) for i in range(classes)]))
        row=dict(task=task,cell=cell,n=count,ba=ba,auc=auc,accuracy=float((pred==yy).mean()),predictions=[dict(label=y,probabilities=p,metadata=m) for y,p,m in zip(labels,probs,meta)])
        if task.startswith('vda'):
            values=np.array([m['value'] for m in meta]);row['fraction_available_reward']=float((values*(pred==yy)).sum()/values.sum())
        rows.append(row)
    task_auc=[np.mean([r['auc'] for r in rows if r['task']==t and r['auc'] is not None]) for t in TASKS]
    return dict(cells=rows,equal_task_mean_auc=float(np.mean(task_auc)),split=split)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',type=int,default=16000)
    p.add_argument('--wall-seconds',type=int,default=28800);p.add_argument('--device',default='cuda');p.add_argument('--fresh',action='store_true');a=p.parse_args()
    if a.steps<16 or a.steps%16 or not 0<a.wall_seconds<=28800:raise ValueError('Use complete 16-task cycles and a finite cap <= 8 hours')
    a.run.mkdir(parents=True,exist_ok=False);started=time.time();deadline=started+a.wall_seconds
    lock=GPULock(PARENT.parents[4]/'LocalRuntime/local_gpu_worker.lock')
    try:
        run(a,started,deadline)
    finally:lock.close()

def run(a,started,deadline):
    configure(a.device);random.seed(201211);np.random.seed(201211);torch.manual_seed(201211)
    source_hash=hashlib.sha256(PARENT.read_bytes()).hexdigest();model,opt=initialize(None if a.fresh else PARENT);model.to(a.device)
    for state in opt.state.values():
        for k,v in state.items():
            if torch.is_tensor(v) and k!='step':state[k]=v.to(a.device)
    scheduler=Scheduler();stream=Stream();step=0;best=-float('inf')
    config=dict(catalog=CATALOG,parent=None if a.fresh else str(PARENT),parent_sha256=source_hash,
        fresh=a.fresh,optimizer_moments_preserved=not a.fresh,new_heads=3,steps=a.steps,started=started,deadline=deadline,
        loss='CE for original13; value/3 weighted terminal CE for VDA; .05 pixel loss; clip100',batch=32,microbatch=4,
        source_sha256={str(f.relative_to(Path(__file__).parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.rglob('*.py')})
    atomic_json(a.run/'config.json',config)
    def save(name='latest'):
        payload=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),stream=stream.state_dict(),
            step=step,best=best,config=config,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        temp=a.run/(name+'.pt.tmp');torch.save(payload,temp);replace_with_retry(temp,a.run/(name+'.pt'))
    try:
        save()
        atomic_json(a.run/'status.json',dict(phase='baseline_validation',step=0,target=a.steps,deadline=deadline))
        baseline=evaluate(model,a.device,'val',64,deadline-90)
        atomic_json(a.run/'validation_000000.json',baseline)
        best=baseline['equal_task_mean_auc'];save('best')
        reason='planned_complete'
        for step in range(1,a.steps+1):
            if time.time()>deadline-120 or (a.run/'stop_requested.json').exists():
                step-=1;reason='time_cap' if time.time()>deadline-120 else 'user_stop';break
            task,cell=scheduler.next();x,y,m=stream.batch(32,task,cell);metrics=update(model,opt,x,y,m,task,a.device)
            append_json(a.run/'progress.jsonl',dict(step=step,task=task,cell=cell,**metrics))
            atomic_json(a.run/'status.json',dict(phase='training',step=step,target=a.steps,task=task,cell=cell,elapsed=time.time()-started))
            if step%16==0:save()
            if step==16 or step%320==0:
                atomic_json(a.run/'status.json',dict(phase='validating',step=step,target=a.steps,deadline=deadline))
                result=evaluate(model,a.device,'val',64,deadline-90);atomic_json(a.run/f'validation_{step:06d}.json',result)
                if result['equal_task_mean_auc']>best:best=result['equal_task_mean_auc'];save('best')
        save()
        if reason=='planned_complete':
            # Final validation also covers runs shorter than the normal cadence.
            result=evaluate(model,a.device,'val',64,deadline-60);atomic_json(a.run/'validation_final.json',result)
            if result['equal_task_mean_auc']>best:best=result['equal_task_mean_auc'];save('best')
            selected=copy.deepcopy(model)
            selected.load_state_dict(torch.load(a.run/'best.pt',map_location=a.device,weights_only=False)['model'])
            result=evaluate(selected,a.device,'test',256,deadline);atomic_json(a.run/'final_test.json',result)
        atomic_json(a.run/'exit.json',dict(reason=reason,step=step,training_complete=reason=='planned_complete',parent_unchanged=hashlib.sha256(PARENT.read_bytes()).hexdigest()==source_hash))
    except BaseException as e:
        save();atomic_json(a.run/'failure.json',dict(error=repr(e),step=step));raise

if __name__=='__main__':main()
