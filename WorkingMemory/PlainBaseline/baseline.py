"""Plain baseline for the five-task spatial battery: per-frame CNN + GRU + per-task linear heads.

Ladder rung 1 of the 2026-09-16 handoff. Everything is trainable from scratch, one learning rate, no
gradient clipping, no fixed priors, no frozen parts. One task per run. The result is the floor that
every later model must beat on the same seeds.

Model (shapes for one frame, batch B):
  x: (B,3,100,100)
  conv 5x5 s2 -> (B,32,50,50); conv 3x3 s2 -> (B,64,25,25); conv 3x3 s2 -> (B,96,13,13); conv 3x3 s2 -> (B,128,7,7)
  each conv followed by GroupNorm(8) and ReLU; flatten -> (B,6272) -> Linear -> (B,256) -> ReLU
  GRU over frames with hidden 256; the final hidden state -> Linear head for the task -> logits.

Usage: python -m WorkingMemory.PlainBaseline.baseline --task orientation_cued --out <dir> [--episodes 100000]
"""
from __future__ import annotations
import argparse,csv,hashlib,json,queue,sys,threading,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from WorkingMemory.SpatialTaskBattery import stimuli as S
from WorkingMemory.BatteryAudit.observers import balanced_accuracy
from WorkingMemory.PlainBaseline.variants import VariantStream,VARIANT_TASKS
ALL_TASK_CLASSES={**S.TASK_CLASSES,**VARIANT_TASKS}
def make_stream(seed,split,task):return VariantStream(seed,split) if task in VARIANT_TASKS else S.SpatialBatteryStream(seed,split)

VERSION='plain_baseline_v1'
SEEDS=dict(train=61973001,scheduler=62973001,val=63973001,test=64973001)
# Rung-1 cells: no retention delay for the delay tasks; the Krauzlis baselines and recognition holds are
# part of the task's definition and are cycled within the run.
RUNG1_CELLS={'orientation_cued':[dict(delay=0)],'spatial_binding':[dict(delay=0)],'motion_duration_cued':[dict(delay=0)],
    'image_recognition':[dict(load=4,probe_hold=h) for h in (3,4,5)],'krauzlis_cued_motion':[dict(baseline_transitions=b) for b in (12,20,28)],
    'orientation_single':[dict(delay=0)],'orientation_ring':[dict(delay=0)],'orientation_sign':[dict(delay=0)]}
DELAY_CELLS={'orientation_cued':[dict(delay=d) for d in S.DELAYS],'spatial_binding':[dict(delay=d) for d in S.DELAYS],'motion_duration_cued':[dict(delay=d) for d in S.DELAYS],
    'image_recognition':[dict(load=n,probe_hold=h) for n in (4,12,24) for h in (3,4,5)],'krauzlis_cued_motion':[dict(baseline_transitions=b) for b in (12,20,28)]}

def cell_name(task,c):
    if 'delay' in c:return f"{task}_D{c['delay']}"
    if 'load' in c:return f"{task}_N{c['load']}_H{c['probe_hold']}"
    return f"{task}_B{c['baseline_transitions']}"

class PlainBaseline(nn.Module):
    """stack=K feeds frames t-K+1..t as 3K input channels (zero-padded before the first frame); stack=1 is per-frame.
    feature_norm='layernorm' applies LayerNorm to the 256-d frame feature after the ReLU so it cannot be scaled to a constant."""
    def __init__(self,task_classes,hidden=256,stack=1,feature_norm='none',center=False,zero_head=False):
        super().__init__();self.stack=int(stack);self.center=bool(center)
        def block(i,o,k,s):return nn.Sequential(nn.Conv2d(i,o,k,s,k//2),nn.GroupNorm(8,o),nn.ReLU(inplace=True))
        self.cnn=nn.Sequential(block(3*self.stack,32,5,2),block(32,64,3,2),block(64,96,3,2),block(96,128,3,2))
        self.proj=nn.Linear(128*7*7,hidden);self.norm=nn.LayerNorm(hidden) if feature_norm=='layernorm' else nn.Identity()
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.heads=nn.ModuleDict({t:nn.Linear(hidden,int(n)) for t,n in task_classes.items()})
        if zero_head:
            for h in self.heads.values():nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    def frames(self,images):
        if self.center:images=images-0.5  # gray background -> 0; only stimulus pixels drive the first convolution
        if self.stack==1:return images
        B,T=images.shape[:2];pad=images.new_zeros(B,self.stack-1,*images.shape[2:]);x=torch.cat([pad,images],1)
        return torch.cat([x[:,k:k+T] for k in range(self.stack)],2)  # (B,T,3K,H,W), oldest frame first
    def forward(self,images,task):
        images=self.frames(images);B,T=images.shape[:2]
        f=self.cnn(images.flatten(0,1));f=self.norm(F.relu(self.proj(f.flatten(1)))).view(B,T,-1)
        _,h=self.gru(f)
        return self.heads[task](h[-1])

def _worker(q,seed,task,cells,batch,offset,stride):
    import torch;torch.set_num_threads(1)
    stream=make_stream(seed,'train',task);i=offset
    while True:
        c=cells[i%len(cells)];i+=stride
        x,y,m=stream.batch(batch,task,c);q.put((x,y,cell_name(task,c)))

class Producer:
    """Renders batches in the background. workers=0: one thread on the train-seed stream (the seeds table is exact).
    workers>=1: that many processes, each with its own stream seeded train_seed + 7919*(worker+1), cycling cells in an
    interleaved order; fresh independent episodes either way."""
    def __init__(self,task,cells,batch,workers=0,depth=8):
        self.task,self.cells,self.batch,self.workers=task,cells,batch,workers;self.stop=False;self.i=0;self.procs=[]
        if workers==0:
            self.stream=make_stream(SEEDS['train'],'train',task);self.q=queue.Queue(depth);self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        else:
            import torch.multiprocessing as mp;ctx=mp.get_context('spawn');self.q=ctx.Queue(depth)
            for w in range(workers):
                pr=ctx.Process(target=_worker,args=(self.q,SEEDS['train']+7919*(w+1),task,cells,batch,w,workers),daemon=True);pr.start();self.procs.append(pr)
    def _run(self):
        while not self.stop:
            c=self.cells[self.i%len(self.cells)];self.i+=1
            x,y,m=self.stream.batch(self.batch,self.task,c);self.q.put((x,y,cell_name(self.task,c)))
    def get(self):return self.q.get()
    def close(self):
        self.stop=True
        for pr in self.procs:pr.terminate()

def evaluate(model,task,cells,split,n,device,batch=64):
    """Fresh stream at the fixed split seed so every evaluation sees identical trials."""
    model.eval();stream=make_stream(SEEDS[split],split,task);out={}
    with torch.no_grad():
        for c in cells:
            ys=[];ps=[];probs=[]
            for off in range(0,n,batch):
                x,y,_=stream.batch(min(batch,n-off),task,c);logits=model(x.to(device),task).float().cpu()
                ys+=y.tolist();ps+=logits.argmax(1).tolist();probs+=logits.softmax(1).tolist()
            k=logits.shape[1];ys=np.array(ys);ps=np.array(ps);probs=np.array(probs)
            confusion=np.zeros((k,k),int)
            for a,b in zip(ys,ps):confusion[a,b]+=1
            if k==2:
                from WorkingMemory.BatteryAudit.observers import auc_binary;auc=auc_binary(ys,probs[:,1])
            else:
                from WorkingMemory.BatteryAudit.observers import auc_binary;auc=float(np.mean([auc_binary((ys==j).astype(int),probs[:,j]) for j in range(k)]))
            out[cell_name(task,c)]=dict(n=int(len(ys)),balanced_accuracy=balanced_accuracy(ys,ps,k),accuracy=float((ys==ps).mean()),chance=1/k,macro_ovr_auc=auc,confusion=confusion.tolist(),loss=float(F.cross_entropy(torch.tensor(np.log(probs+1e-12),dtype=torch.float32),torch.tensor(ys,dtype=torch.long))))
    model.train();return out

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--out',required=True);ap.add_argument('--episodes',type=int,default=100000)
    ap.add_argument('--batch',type=int,default=64);ap.add_argument('--chunk',type=int,default=32);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--cells',default='rung1',help="'rung1', 'delay', or a comma list of delays such as 'D0,D1,D2'");ap.add_argument('--val-every',type=int,default=200);ap.add_argument('--val-n',type=int,default=256);ap.add_argument('--test-n',type=int,default=512)
    ap.add_argument('--profile',type=int,default=0,help='run this many updates and exit');ap.add_argument('--device',default='cuda')
    ap.add_argument('--stack',type=int,default=1,help='frames stacked as input channels');ap.add_argument('--feature-norm',default='none',choices=('none','layernorm'));ap.add_argument('--workers',type=int,default=0,help='producer processes; 0 = single thread on the exact train seed')
    ap.add_argument('--encoder',default='plain',choices=('plain','accum'));ap.add_argument('--accumulator',default='convgru',choices=('convgru','opponent','kda','none'))
    ap.add_argument('--init',default=None,help='checkpoint whose model weights initialise this run (curriculum); the receipt records its hash');ap.add_argument('--center',action='store_true',help='subtract 0.5 from the input so gray is zero');ap.add_argument('--zero-head',action='store_true',help='zero-initialise the task heads')
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.backends.cudnn.benchmark=True
    if a.cells=='rung1':cells=RUNG1_CELLS[a.task]
    elif a.cells=='delay':cells=DELAY_CELLS[a.task]
    else:cells=[dict(delay=int(c.strip().lstrip('D'))) for c in a.cells.split(',')]
    if a.encoder=='accum':
        from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline
        model=AccumulatorBaseline(ALL_TASK_CLASSES,stack=a.stack,feature_norm=a.feature_norm,center=a.center,zero_head=a.zero_head,accumulator=a.accumulator).to(device)
    else:model=PlainBaseline(ALL_TASK_CLASSES,stack=a.stack,feature_norm=a.feature_norm,center=a.center,zero_head=a.zero_head).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    n_params=sum(p.numel() for p in model.parameters());init_info=None
    if a.init:
        saved=torch.load(a.init,map_location=device);missing,unexpected=model.load_state_dict(saved['model'],strict=False)
        if unexpected or any(not k.startswith('heads.') for k in missing):raise RuntimeError(f'init mismatch beyond task heads: missing {missing} unexpected {unexpected}')
        init_info=dict(path=a.init,sha256=sha(a.init),parent_task=saved.get('task'),parent_step=saved.get('step'),parent_args=saved.get('args'),fresh_heads=sorted({k.split('.')[1] for k in missing}))
    producer=Producer(a.task,cells,a.batch,a.workers)
    updates=a.profile or int(np.ceil(a.episodes/a.batch))
    receipt=dict(version=VERSION,task=a.task,cells=[cell_name(a.task,c) for c in cells],args=vars(a),params=n_params,updates_planned=updates,
        sources={p:sha(p) for p in (S.__file__,__file__)},init=init_info,seeds=SEEDS,model_seed=a.seed,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),platform=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=1),encoding='utf-8')
    log=open(out/'metrics.csv','w',newline='');writer=csv.writer(log);writer.writerow(['update','episodes','cell','loss','accuracy','grad_norm','seconds','wait_seconds']);best=None;history=[]
    t_start=time.time()
    for step in range(1,updates+1):
        tw=time.time();x,y,name=producer.get();wait=time.time()-tw;t0=time.time()
        opt.zero_grad(set_to_none=True);total=0.;correct=0
        for i in range(0,len(x),a.chunk):
            xb=x[i:i+a.chunk].to(device,non_blocking=True);yb=y[i:i+a.chunk].to(device)
            logits=model(xb,a.task);loss=F.cross_entropy(logits,yb)*(len(xb)/len(x));loss.backward()
            total+=float(loss);correct+=int((logits.argmax(1)==yb).sum())
        gnorm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
        if not np.isfinite(total) or not np.isfinite(gnorm):raise FloatingPointError(f'non-finite at update {step}')
        opt.step()
        if device.type=='cuda':torch.cuda.synchronize()
        writer.writerow([step,step*a.batch,name,f'{total:.5f}',f'{correct/len(x):.4f}',f'{gnorm:.4f}',f'{time.time()-t0:.3f}',f'{wait:.3f}']);log.flush()
        if a.profile:print(f'update {step} {name} loss {total:.4f} acc {correct/len(x):.3f} gnorm {gnorm:.3f} {time.time()-t0:.2f}s wait {wait:.2f}s peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB' if device.type=='cuda' else '',flush=True);continue
        if step%a.val_every==0 or step==updates:
            v=evaluate(model,a.task,cells,'val',a.val_n,device);mba=float(np.mean([r['balanced_accuracy'] for r in v.values()]))
            history.append(dict(update=step,episodes=step*a.batch,mean_ba=mba,cells=v,elapsed=time.time()-t_start))
            (out/'validation.json').write_text(json.dumps(history,indent=1),encoding='utf-8')
            torch.save(dict(version=VERSION,task=a.task,step=step,model=model.state_dict(),args=vars(a)),out/'terminal.pt')
            if best is None or mba>best[0]:best=(mba,step);torch.save(dict(version=VERSION,task=a.task,step=step,model=model.state_dict(),args=vars(a)),out/'best.pt')
            print(f'[{a.task}] update {step}/{updates} episodes {step*a.batch} val mean BA {mba:.3f} '+' '.join(f"{k}={r['balanced_accuracy']:.3f}" for k,r in v.items())+f' ({time.time()-t_start:.0f}s)',flush=True)
    producer.close()
    if a.profile:return
    final={}
    for tag in ('best','terminal'):
        saved=torch.load(out/f'{tag}.pt',map_location=device);model.load_state_dict(saved['model']);final[tag]=dict(step=saved['step'],test=evaluate(model,a.task,cells,'test',a.test_n,device),sha256=sha(out/f'{tag}.pt'))
    receipt.update(finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),seconds=time.time()-t_start,updates_done=updates,episodes=updates*a.batch,best_val=dict(mean_ba=best[0],step=best[1]),final=final)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=1),encoding='utf-8')
    print(f'[{a.task}] done. test BA per cell (terminal): '+' '.join(f"{k}={r['balanced_accuracy']:.3f}" for k,r in final['terminal']['test'].items()),flush=True)

if __name__=='__main__':main()
