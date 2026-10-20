"""Resumable, frozen local inference; never touches optimizer or training streams."""
import argparse, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from .design import *
from .recording import RecordedSuite,SOURCE
from SecondPass.MemoryGuidedSearch.model import ExpandedSuite
from SecondPass.MemoryGuidedSearch import dataset as D
from SecondPass.PixelContrastiveVAE.train import configure,GPULock

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def new_stream(split,namespace):
    D.NAMESPACES[split]=namespace
    return D.Stream(split)

def draw(stream,n,task,cell):
    rows=[stream.batch(min(4,n-i),task,cell) for i in range(0,n,4)]
    return torch.cat([r[0] for r in rows]),torch.cat([r[1] for r in rows]),sum([r[2] for r in rows],[])

def save_npz(path,**values):
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**values);tmp.replace(path)

class Collector:
    def __init__(self,out=OUT,device='cuda',hours=16):
        self.out=out;self.out.mkdir(parents=True,exist_ok=True);self.device=device
        self.start=time.time();self.deadline=self.start+hours*3600;self.presentations=0
        self.models={}
        for name,path in CHECKPOINTS.items():
            assert sha(path)==EXPECTED[name],f'Checkpoint changed: {path}'
            c=torch.load(path,map_location='cpu',weights_only=False)
            model=RecordedSuite().eval().requires_grad_(False);model.load_state_dict(c['model'],strict=True)
            self.models[name]=model
        self.model=self.models['best'].to(device)

    def status(self,**kw):
        if (self.out/'stop_requested.json').exists():raise InterruptedError('Analysis stop requested')
        if time.time()>self.deadline:raise TimeoutError('Analysis wall-clock cap reached')
        write(self.out/'status.json',dict(pid=os.getpid(),time=time.time(),elapsed=time.time()-self.start,
            presentations=self.presentations,deadline=self.deadline,**kw))

    @torch.inference_mode()
    def infer(self,x,task,record=False,bias_value=None,lesion=None):
        p,d=self.model(x.to(self.device),task,True,
            None if bias_value is None else torch.as_tensor(bias_value,device=self.device),record,lesion)
        p=p.softmax(-1).cpu().numpy();a=d['attention'].cpu().numpy()
        assert np.isfinite(p).all() and np.isfinite(a).all()
        assert np.allclose(a.sum((2,3)),1,atol=1e-5)
        self.presentations+=len(x)
        return p,a,{k:v.numpy() for k,v in d['recordings'].items()}

    def check(self):
        native=ExpandedSuite().eval().requires_grad_(False).to(self.device)
        native.load_state_dict(self.model.state_dict());checks={};s=new_stream('test',NS+99)
        for task,spec in TASKS.items():
            cell=neural_cells(task)[0];x,y,m=draw(s,2,task,cell);x=x.to(self.device)
            with torch.inference_mode():
                z,old=native(x,task,True);zz,d=self.model(x,task,True,record=True)
                torch.testing.assert_close(z,zz,rtol=0,atol=0)
                torch.testing.assert_close(old['attention'],d['attention'],rtol=0,atol=0)
                torch.testing.assert_close(d['recordings']['probabilities'][:,-1],z.softmax(-1).cpu(),rtol=0,atol=0)
                mid=max(1,x.shape[1]//2)
                prefix=self.model(x[:,:mid],task).softmax(-1).cpu()
                torch.testing.assert_close(prefix,d['recordings']['probabilities'][:,mid-1],rtol=1e-5,atol=2e-6)
                zero=self.model(x,task,attention_bias=x.new_zeros(2,x.shape[1],100))
                torch.testing.assert_close(z,zero,rtol=0,atol=0)
                global_shift=self.model(x,task,attention_bias=x.new_full((2,x.shape[1],100),4))
                torch.testing.assert_close(z.softmax(-1),global_shift.softmax(-1),rtol=1e-4,atol=3e-5)
            checks[task]=dict(native_logits_exact=True,attention_exact=True,final_readout_exact=True,prefix_causal=True,batch_zero_exact=True,global_shift_invariant=True)
        del native;torch.cuda.empty_cache() if self.device=='cuda' else None
        (self.out/'audited_forward.py').write_text(SOURCE,encoding='utf8')
        write(self.out/'checks.json',checks)

    def native(self,task):
        folder=self.out/task/'native';folder.mkdir(parents=True,exist_ok=True)
        for c in TASKS[task]['conditions']:
            cell=c['id'];path=folder/(cell+'.npz')
            if path.exists():continue
            s=new_stream('test',NS);labels=[];probs=[];maps=[];temporal=[];metadata=[];example=None
            for start in range(0,NATIVE_N,BATCH):
                self.status(phase='native',task=task,cell=cell,completed=start,total=NATIVE_N)
                x,y,m=draw(s,min(BATCH,NATIVE_N-start),task,cell)
                p,a,r=self.infer(x,task,record=True)
                labels.extend(y.tolist());probs.extend(p);maps.extend(a);temporal.extend(r['probabilities']);metadata+=m
                if example is None:example=x[0].numpy()
            write(folder/(cell+'.json'),metadata)
            save_npz(path,y=np.array(labels),p=np.array(probs),attention=np.array(maps),temporal=np.array(temporal),example=example)
        # Matched checkpoint robustness: first 32 trials of every native cell.
        self.models['best'].cpu();self.model=self.models['latest'].to(self.device)
        try:
            for c in TASKS[task]['conditions']:
                cell=c['id'];path=folder/(cell+'_latest.npz')
                if path.exists():continue
                self.status(phase='latest_robustness',task=task,cell=cell)
                s=new_stream('test',NS);yy=[];pp=[]
                for i in range(0,32,BATCH):
                    x,y,m=draw(s,BATCH,task,cell);p,a,_=self.infer(x,task);yy.extend(y.tolist());pp.extend(p)
                save_npz(path,y=np.array(yy),p=np.array(pp))
        finally:self.models['latest'].cpu();self.model=self.models['best'].to(self.device)

    def neural(self,task):
        folder=self.out/task/'neural';folder.mkdir(parents=True,exist_ok=True)
        for cell in neural_cells(task):
            for split,count,namespace in [('val',PROBE_TRAIN,NS+1),('test',PROBE_TEST,NS+2)]:
                path=folder/f'{cell}_{split}.npz'
                if path.exists():continue
                stream=new_stream(split,namespace);data={};metadata=[];examples=[]
                for start in range(0,count,BATCH):
                    self.status(phase='neural',task=task,cell=cell,split=split,completed=start,total=count)
                    x,y,m=draw(stream,min(BATCH,count-start),task,cell);p,a,r=self.infer(x,task,record=True)
                    # Dense 8x8 RGB description of the last sample/study image.
                    # Generic content regression is distinct from final-label decoding.
                    content=[]
                    for xx,mm in zip(x,m):
                        ph=phases(task,mm,len(xx));ix=[j for j,v in enumerate(ph) if v in ('sample','study','reference')]
                        frame=xx[ix[-1] if ix else 0]
                        content.append(torch.nn.functional.adaptive_avg_pool2d(frame,(8,8)).flatten().numpy())
                    values=dict(y=y.numpy(),p=p,attention=a,content=np.array(content),**r)
                    for name,value in values.items():data.setdefault(name,[]).append(value.astype(np.float16) if name in ('appearance','fusion','spatial') else value)
                    metadata+=m
                    if len(examples)<8:examples.extend(x.numpy()[:8-len(examples)])
                write(folder/f'{cell}_{split}.json',metadata)
                save_npz(path,**{k:np.concatenate(v) for k,v in data.items()},examples=np.array(examples))

    def interventions(self,task):
        folder=self.out/task/'interventions';folder.mkdir(parents=True,exist_ok=True)
        # All neural conditions, with exact pixel pairing across manipulations.
        for cell in neural_cells(task):
            path=folder/f'{cell}.npz'
            if path.exists():continue
            stream=new_stream('test',NS+3);data={};metadata=[];names=None
            for start in range(0,INTERVENTION_N,BATCH):
                x,y,m=draw(stream,min(BATCH,INTERVENTION_N-start),task,cell);metadata+=m
                vals={};active={}
                for arm in arms():
                    self.status(phase='interventions',task=task,cell=cell,completed=start,arm=arm['name'])
                    b=bias(task,m,len(x[0]),arm)
                    p,a,_=self.infer(x,task,bias_value=b);vals[arm['name']]=p;active[arm['name']]=np.any(b!=0,axis=(1,2))
                    if arm['name']=='sham':base=p
                    if arm['site']=='global':assert np.allclose(base,p,atol=3e-5)
                when=retention_boundary(task,m[0],len(x[0]))
                assert all(retention_boundary(task,z,len(x[0]))==when for z in m)
                for state in ('cls','spatial','comparison'):
                    for gain in (0.,1.5):
                        name=f'{state}_gain{gain:g}_t{when}'
                        p,a,_=self.infer(x,task,lesion=dict(state=state,gain=gain,time=when));vals[name]=p;active[name]=np.ones(len(x),bool)
                if names is None:names=list(vals)
                for key,value in dict(y=y.numpy(),p=np.stack([vals[k] for k in names],1),active=np.stack([active[k] for k in names],1)).items():data.setdefault(key,[]).append(value)
            write(folder/f'{cell}.json',dict(metadata=metadata,arms=names,retention_boundary=when))
            save_npz(path,**{k:np.concatenate(v) for k,v in data.items()})

    def psychometrics(self,task):
        from SecondPass.StabilizedAttentionSuite.Analysis import renderers as R
        from SecondPass.StabilizedAttentionSuite.Analysis.design import STRENGTHS
        from SecondPass.StabilizedAttentionSuite.Analysis.experiment import controlled
        from SecondPass.StabilizedAttentionVDA.Analysis.design import movie
        from WorkingMemory.PlainBaseline.variants import VariantStream
        folder=self.out/task/'psychometric';folder.mkdir(parents=True,exist_ok=True)
        if task in STRENGTHS:
            levels=STRENGTHS[task]
            cases=[(v,e) for v in levels for e in (['target','foil','catch'] if task=='krauzlis_cued_motion' else ['native'])]
        elif task.startswith('vda'):cases=[(v,e) for v in (0,2,4,8,12,18,26,40,65) for e in ('cued','uncued','catch') if e!='catch' or v==0]
        else:return # full native factorial already varies difficulty, load, exposure and delay
        for level,event in cases:
            if task=='orientation_ring' and level==0:continue
            if task=='krauzlis_cued_motion' and event=='catch' and level!=0:continue
            path=folder/f'{level}_{event}.npz'
            if path.exists():continue
            ps=[];ys=[];metas=[]
            for start in range(0,PSYCH_N,BATCH):
                self.status(phase='psychometric',task=task,strength=level,event=event,completed=start)
                xx=[];yy=[];mm=[]
                if task=='motion_direction':
                    stream=R.CardinalMotionStream(NS*100+start,'test');x,y,mm=R.motion_batch(stream,BATCH,displacement=level)
                elif task=='orientation_ring':
                    stream=VariantStream(NS*100+start,'test',magnitudes=(level,level,level));x,y,mm=stream.batch(BATCH,task,dict(delay=0))
                else:
                    for rep in range(start,start+BATCH):
                        if task.startswith('vda'):
                            v,m=movie(int(task[3:]),rep,level,event,.75,namespace=NS+4);label=m['label']
                        elif task in ('orientation_cued','krauzlis_cued_motion'):
                            v,m=controlled(task,12,rep,level,event if task.startswith('krauzlis') else 'target',NS+4);label=m['label']
                        else:
                            seed=NS*100+rep;label=rep%2
                            if task=='natural_spectrum':
                                stream=R.NaturalSpectrum('test');stream.analysis_strength=level;v,m=R.spectrum_sample(stream,np.random.default_rng(seed),label)
                            else:
                                stream=R.TaskStream(seed,'test');stream.analysis_strength=level;v,m=getattr(R,'sensory_'+task)(stream,label,0)
                            v=torch.from_numpy(v)
                        xx.append(v);yy.append(label);mm.append(m)
                    x=torch.stack(xx);y=torch.tensor(yy)
                p,a,_=self.infer(x,task);ps.extend(p);ys.extend(y.tolist());metas+=mm
            write(path.with_suffix('.json'),metas)
            save_npz(path,p=np.array(ps),y=np.array(ys),strength=level)

    def vda_counterfactual(self,task):
        if not task.startswith('vda'):return
        from SecondPass.StabilizedAttentionVDA.Analysis.design import movie,bias as vbias,masks,PRIMARY_ARMS
        size=int(task[3:]);folder=self.out/task/'counterfactual';folder.mkdir(exist_ok=True,parents=True)
        for event in ('cued','uncued','catch'):
            path=folder/f'{event}.npz'
            if path.exists():continue
            ps=[];ys=[];mm=[]
            for start in range(0,256,BATCH):
                trials=[movie(size,r,18 if event!='catch' else 0,event,1.,cue=size-1,namespace=NS+5) for r in range(start,start+BATCH)]
                x=torch.stack([r[0] for r in trials]);m=[r[1] for r in trials];p=[]
                for arm in PRIMARY_ARMS:
                    self.status(phase='counterfactual',task=task,event=event,arm=arm['name'],completed=start)
                    pp,aa,_=self.infer(x,task,bias_value=vbias(m[0],arm,masks(size)));p.append(pp)
                ps.extend(np.stack(p,1));ys.extend([z['label'] for z in m]);mm+=m
            save_npz(path,p=np.array(ps),y=np.array(ys));write(path.with_suffix('.json'),dict(metadata=mm,arms=[a['name'] for a in PRIMARY_ARMS]))

    def run(self,tasks):
        write(self.out/'protocol.json',dict(checkpoints={k:dict(path=str(v),sha256=EXPECTED[k]) for k,v in CHECKPOINTS.items()},
            tasks=list(TASKS),conditions=sum(len(v['conditions']) for v in TASKS.values()),native_trials_per_cell=NATIVE_N,
            psychometric_trials_per_point=PSYCH_N,probe_train=PROBE_TRAIN,probe_test=PROBE_TEST,intervention_trials_per_cell=INTERVENTION_N,
            thresholds=THRESHOLDS,neural_cells={t:neural_cells(t) for t in TASKS},namespace=NS,
            analysis_source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            uncertainty='Resample procedural trials within cells. Cluster natural-image comparisons by source identity. Model is one trained subject.',
            chronometry='First post-evidence threshold crossing of unchanged terminal readout applied to prefix states; noncrossers are censored; units model frames.',
            primary_intervention='Target and distractor +/-4 evidence logit bias; Holm across tasks/arms. Other doses/epochs exploratory.',
            no_model_training=True,no_optimizer=True,start=time.time()))
        self.check()
        for task in tasks:
            self.native(task);self.neural(task);self.interventions(task);self.psychometrics(task);self.vda_counterfactual(task)
            write(self.out/task/'collection_complete.json',dict(time=time.time(),task=task))
        assert all(sha(p)==EXPECTED[k] for k,p in CHECKPOINTS.items())
        write(self.out/'collection_complete.json',dict(tasks=tasks,presentations=self.presentations,elapsed=time.time()-self.start,checkpoints_unchanged=True))
        self.status(phase='collection_complete')

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--tasks',nargs='*');p.add_argument('--hours',type=float,default=16);a=p.parse_args()
    configure('cuda');lock=GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:
        runner=Collector(hours=a.hours)
        if a.check:runner.check();print('All 19 task checks passed',flush=True)
        else:runner.run(a.tasks or list(TASKS))
    except BaseException:
        write(OUT/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
    finally:lock.close()

if __name__=='__main__':main()
