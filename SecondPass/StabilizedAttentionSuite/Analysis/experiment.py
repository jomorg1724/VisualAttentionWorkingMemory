"""Frozen suite analysis. Every forward uses native RGB pixels; metadata only designs/reports tests."""
import json,time,hashlib,os,copy,traceback
from pathlib import Path
import numpy as np
import torch
from .design import *
from . import renderers as R
from SecondPass.StabilizedAttentionSuite.model import StabilizedAttentionSuite
from SecondPass.StabilizedAttentionSuite.train import hashes
from SecondPass.TaskSuite.suite import SuiteStream,TASKS
from SecondPass.PixelContrastiveVAE.train import configure,GPULock,atomic_json
from WorkingMemory.PlainBaseline.variants import VariantStream

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(*x):return int(np.random.SeedSequence(x).generate_state(1)[0])

class FreshStream(SuiteStream):
    def __init__(self,namespace):super().__init__('test');self.namespace=namespace
    def stream_seed(self,task,cell):return seed(self.namespace,TASKS[task]['stream_id'],self._cell(task,cell)[0])

def controlled(task,condition,rep,angle,event='target',namespace=200201):
    s=seed(namespace,TASKS[task]['stream_id'],condition,rep)
    stream=R.SpatialBatteryStream(s,'test');rng=np.random.default_rng(s)
    if task=='krauzlis_cued_motion':
        target=rep%2;label=int(event=='target');stream._event_kind=event
        frames,meta=R.krauzlis(stream,rng,label,target,dict(baseline_transitions=condition,analysis_angle=angle,analysis_sign=1 if rep//2%2 else -1))
    else:
        target=(rep//2)%4;label=(rep//8)%2;stream._cue_sign=1 if rep%2 else -1
        frames,meta=R.cued_orientation(stream,rng,label,target,dict(delay=condition,analysis_angle=angle))
        event='aligned' if label else 'opposite' if meta['rotations_degrees'][target] else 'unchanged'
    meta.update(task=task,label=label,condition=condition,rep=rep,seed=s,angle=angle,event=event,
        cluster=f'{condition}:{rep}',analysis_foil_location=(target+1)%len(meta['positions_xy']))
    return torch.from_numpy(np.stack(frames)),meta

def renderer_checks():
    checks={}
    for task in ('orientation','contrast','spatial_frequency','chromatic_increment','contour'):
        for lev,val in enumerate(NATIVE[task]):
            for label in (0,1):
                a=R.TaskStream(72,'test');b=R.TaskStream(72,'test');b.analysis_strength=val
                x,_=getattr(a,'_'+task)(label,lev);y,_=getattr(R,'sensory_'+task)(b,label,lev)
                assert np.array_equal(x,y),task
        checks[task]=True
    for v in NATIVE['motion_direction']:
        a=R.CardinalMotionStream(23,'test');b=R.CardinalMotionStream(23,'test')
        x,y,_=a.batch(8,displacement=v);xx,yy,_=R.motion_batch(b,8,displacement=v)
        assert torch.equal(x,xx) and torch.equal(y,yy)
    checks['motion_direction']=True
    p=R.NaturalSpectrum('test')
    x,m=p.sample(np.random.default_rng(72),1);p.analysis_strength=m['beta_delta']
    xx,mm=R.spectrum_sample(p,np.random.default_rng(72),1);assert np.array_equal(x,xx)
    checks['natural_spectrum']=True
    for task in ('orientation_cued','krauzlis_cued_motion'):
        a=R.SpatialBatteryStream(23,'test');a._cue_sign=1;a._event_kind='target'
        native=a._orientation if task=='orientation_cued' else a._krauzlis
        adapter=R.cued_orientation if task=='orientation_cued' else R.krauzlis
        x,m=native(np.random.default_rng(72),1,0,{})
        xx,mm=adapter(a,np.random.default_rng(72),1,0,{})
        assert np.array_equal(x,xx)
        checks[task]=True
    # Zero-angle extension preserves the schedule, not a physically positive class.
    x,m=controlled('orientation_cued',4,8,0);assert all(v==0 for v in m['rotations_degrees'])
    x,m=controlled('krauzlis_cued_motion',12,0,26);xx,_=controlled('krauzlis_cued_motion',12,0,60)
    assert torch.equal(x[:m['first_postchange_frame']],xx[:m['first_postchange_frame']])
    return checks

class Runner:
    def __init__(self):
        self.deadline=json.loads((OUT/'budget.json').read_text())['deadline'];self.started=time.time();self.n=0
        self.phase='checks';self.last_status=0;self.model=None
    def status(self,**kw):
        if time.time()>self.deadline-30:raise TimeoutError('Finite local analysis allocation exhausted')
        atomic_json(OUT/'status.json',dict(phase=self.phase,presentations=self.n,seconds=time.time()-self.started,time=time.time(),**kw))
    def infer(self,x,task,bias=None):
        self.status(task=task)
        assert x.dtype==torch.float32 and x.ndim==5 and x.shape[2]==3 and bool(torch.isfinite(x).all())
        pp=[];aa=[]
        with torch.inference_mode():
            for i in range(0,len(x),4):
                z,d=self.model(x[i:i+4].cuda(),task,True,None if bias is None else torch.as_tensor(bias,dtype=torch.float32,device='cuda'))
                assert bool(torch.isfinite(z).all());a=d['attention'].cpu().numpy()
                assert np.allclose(a.sum((2,3)),1,atol=1e-5)
                pp.extend(z.softmax(-1).cpu().tolist());aa.extend(a);self.n+=len(z)
        return np.asarray(pp),np.asarray(aa)
    def write(self,path,rows):
        with (OUT/path).open('a',encoding='utf8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    def records(self,task,metas,p,a,**extra):
        rows=[]
        for m,prob,alpha in zip(metas,p,a):
            r=dict(task=task,label=m['label'],choice=int(prob.argmax()),probability=prob.tolist(),metadata=m,**extra)
            if 'positions_xy' in m:
                masks=[spatial_mask(c).reshape(10,10) for c in m['positions_xy']]
                r['attention_mass_by_location']=[[float((v*mask).sum()) for mask in masks] for v in alpha]
            rows.append(r)
        return rows
    def psychometrics(self):
        self.phase='psychometrics'
        natural=R.NaturalSpectrum('test')
        for task,vals in STRENGTHS.items():
            if task in ('orientation_cued','krauzlis_cued_motion'):continue
            for val in vals:
                if task=='orientation_ring' and val==0:
                    # Variant's assertion rejects ambiguous zero; matched limit tested in cued orientation.
                    continue
                for start in range(0,256,8):
                    if task=='orientation_ring':
                        stream=VariantStream(seed(200101,TASKS[task]['stream_id'],start),'test',magnitudes=(val,val,val))
                        x,y,mm=stream.batch(8,task,dict(delay=0))
                    elif task=='motion_direction':
                        stream=R.CardinalMotionStream(seed(200101,TASKS[task]['stream_id'],start),'test')
                        x,y,mm=R.motion_batch(stream,8,displacement=val)
                    else:
                        xs=[];mm=[];ys=[]
                        for rep in range(start,start+8):
                            s=seed(200101,TASKS[task]['stream_id'],rep);label=rep%2
                            if task=='natural_spectrum':
                                natural.analysis_strength=val;v,m=R.spectrum_sample(natural,np.random.default_rng(s),label)
                            else:
                                stream=R.TaskStream(s,'test');stream.analysis_strength=val
                                v,m=getattr(R,'sensory_'+task)(stream,label,0)
                            xs.append(v);ys.append(label);mm.append(m)
                        x=torch.from_numpy(np.stack(xs));y=torch.tensor(ys)
                    for j,m in enumerate(mm):m.update(label=int(y[j]),strength=val,condition='strength',cluster=str(start+j))
                    p,a=self.infer(x,task);self.write('psychometric_trials.jsonl',self.records(task,mm,p,a,strength=val,cell='strength'))
        for task,conditions,nrep in [('orientation_cued',[0,4,12,24],128),('krauzlis_cued_motion',[12,20,28],32)]:
            for cond in conditions:
                for val in STRENGTHS[task]:
                    for event in (['target','foil'] if task=='krauzlis_cued_motion' else ['target']):
                        for start in range(0,nrep,8):
                            trials=[controlled(task,cond,r,val,event,200102) for r in range(start,start+8)]
                            x=torch.stack([v[0] for v in trials]);mm=[v[1] for v in trials]
                            p,a=self.infer(x,task);self.write('psychometric_trials.jsonl',self.records(task,mm,p,a,strength=val,cell=str(cond)))
                if task=='krauzlis_cued_motion':
                    for start in range(0,nrep,8):
                        trials=[controlled(task,cond,r,0,'catch',200102) for r in range(start,start+8)]
                        p,a=self.infer(torch.stack([v[0] for v in trials]),task)
                        self.write('psychometric_trials.jsonl',self.records(task,[v[1] for v in trials],p,a,strength=0,cell=str(cond)))
        native=FreshStream(200103)
        for task in ('motion_duration_cued','spatial_binding','image_recognition'):
            for c in TASKS[task]['conditions']:
                for start in range(0,512,8):
                    x,y,mm=native.batch(8,task,c['id']);p,a=self.infer(x,task)
                    val=c['kwargs'].get('delay',c['kwargs'].get('load'))
                    for m in mm:m['cluster']=m['suite_trial_id']
                    self.write('psychometric_trials.jsonl',self.records(task,mm,p,a,strength=val,cell=c['id']))
        self.status(stage_complete=True)
    def interventions(self):
        self.phase='interventions'
        for task,conditions,nrep,angles in [('krauzlis_cued_motion',[12,20,28],32,[8,18,26,28,40]),('orientation_cued',[0,4,12,24],64,[8,15,30,45])]:
            arms=interventions(task)
            for cond in conditions:
                cases=[(v,e) for v in angles for e in ('target','foil')] + [(0,'catch')] if task=='krauzlis_cued_motion' else [(v,'target') for v in angles]
                for val,event in cases:
                    # Same target and selected distractor in each batch; independent nuisance seeds.
                    stride=4 if task=='krauzlis_cued_motion' else 16
                    for group in range(stride):
                        trials=[controlled(task,cond,r,val,event) for r in range(group,nrep,stride)]
                        x=torch.stack([v[0] for v in trials]);mm=[v[1] for v in trials]
                        pixels=hashlib.sha256(x.numpy().tobytes()).hexdigest();sham=None
                        for arm in arms:
                            bias=bias_for(task,mm[0],arm,x.shape[1])
                            assert all(np.array_equal(bias,bias_for(task,m,arm,x.shape[1])) for m in mm)
                            p,a=self.infer(x,task,bias)
                            if arm['name']=='sham':sham=(p,a)
                            if arm['site']=='global':
                                assert np.max(np.abs(p-sham[0]))<3e-4,'Global shift changed choices materially'
                                assert np.array_equal(p.argmax(1),sham[0].argmax(1))
                                assert np.max(np.abs(a-sham[1]))<3e-4
                            self.write('intervention_trials.jsonl',self.records(task,mm,p,a,arm=arm['name'],site=arm['site'],dose=arm['dose'],epoch=arm['epoch'],strength=val,pixel_batch_sha256=pixels))
        self.status(stage_complete=True)
    def atlas(self):
        self.phase='attention_atlas';stream=FreshStream(200303);entries=[]
        atlas=OUT/'atlas';atlas.mkdir(exist_ok=True)
        for task,spec in TASKS.items():
            for c in spec['conditions']:
                groups={}
                for start in range(0,256,8):
                    x,y,mm=stream.batch(8,task,c['id']);p,a=self.infer(x,task)
                    for xx,yy,m,pp,aa in zip(x.numpy(),y.tolist(),mm,p,a):
                        if task=='krauzlis_cued_motion':group=f"side{m['target_location']}_{m['event_type']}"
                        elif task=='image_recognition':group='seen' if yy else 'novel'
                        elif 'target_location' in m:
                            if m['target_location']!=0:continue
                            group='target_top_left'
                            if task=='orientation_cued':group+=f"_cue{'plus' if m['cue_sign']>0 else 'minus'}"
                        else:
                            group=str(m.get('difficulty','all'))
                            # Remove nuisance pedestal in contrast to avoid many tiny averaging cells.
                            if task=='contrast':group=f"increment_{m['contrast_increment']:g}"
                        if group not in groups:groups[group]=dict(n=0,sx=np.zeros_like(xx,dtype=np.float64),sa=np.zeros_like(aa,dtype=np.float64),sa2=np.zeros_like(aa,dtype=np.float64),example=xx,example_attention=aa,metadata=m,label=yy,probabilities=pp.tolist())
                        g=groups[group];g['n']+=1;g['sx']+=xx;g['sa']+=aa;g['sa2']+=aa**2
                for group,g in groups.items():
                    key=f'{task}--{c["id"]}--{group}';n=g['n'];avg=g['sa']/n
                    np.savez_compressed(atlas/(key+'.npz'),mean_images=(g['sx']/n).astype('float32'),mean_attention=avg.astype('float32'),attention_se=np.sqrt(np.maximum(0,g['sa2']/n-avg**2)/max(1,n-1)).astype('float32'),images=g['example'],attention=g['example_attention'])
                    entries.append(dict(id=key,task=task,cell=c['id'],group=group,n=n,timesteps=len(avg),phases=phase_labels(task,g['metadata'],len(avg)),metadata=g['metadata'],label=g['label'],probabilities=g['probabilities']))
                atomic_json(atlas/'index.json',entries)
        self.status(stage_complete=True,groups=len(entries));return len(entries)
    def main(self):
        self.status();checks=renderer_checks();configure('cuda')
        old_hashes={name:sha(CHECKPOINT.with_name(name+'.pt')) for name in ('best','latest')}
        ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);assert ck['state']['step']==54808
        assert ck['config']['source_hashes']==hashes(),'Original source changed since checkpoint'
        self.model=StabilizedAttentionSuite().cuda();self.model.load_state_dict(ck['model']);self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        original={k:v.clone().cpu() for k,v in self.model.state_dict().items()};del ck
        x,m=controlled('krauzlis_cued_motion',12,0,26)
        with torch.inference_mode():
            z=self.model(x[None].cuda(),'krauzlis_cued_motion')
            zz=self.model(x[None].cuda(),'krauzlis_cued_motion',attention_bias=torch.zeros(len(x),100,device='cuda'))
            assert torch.equal(z,zz)
        atomic_json(OUT/'checks.json',dict(native_renderer_bitwise_equality=checks,zero_bias_logits_bitwise_equal=True,all_weights_frozen=True,optimizer_created=False))
        atomic_json(OUT/'protocol.json',dict(checkpoint=str(CHECKPOINT),step=54808,checkpoint_sha256=old_hashes,source_hashes=hashes(),analysis_source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},strengths=STRENGTHS,native_strengths=NATIVE,namespaces=[200101,200102,200103,200201,200303],sensory_n_per_point=256,cued_orientation_n_per_angle_delay=128,krauzlis_n_clusters=96,memory_n_per_cell=512,atlas_n_per_native_cell=256,intervention_krauzlis_clusters=96,intervention_orientation_clusters=256,intervention_angles=dict(krauzlis=[8,18,26,28,40],orientation=[8,15,30,45]),arms={t:interventions(t) for t in ['krauzlis_cued_motion','orientation_cued']},primary_benefits=['target_+4_all','target_+8_all','distractor_-4_all','distractor_-8_all'],near_threshold_angles=dict(krauzlis=[8,18],orientation=[8,15]),uncertainty='Paired cluster bootstrap, no trial independence across angles/arms. Pointwise exploratory intervals; Holm correction across eight task-by-benefit primary tests.',attention='Actual single CLS-query weights alpha. No NxN attention exists in this checkpoint. Query average equals alpha; mass sums one.',zero_angle='Scheduled labels only; no physical change, no true hit interpretation.',no_signal_ring='Zero omitted because native variant rejects physically ambiguous positive label.',intervention_units='Additive pre-softmax priority logits, not biological current.',orientation_distractor='One fixed next-index uncued location, chosen independently of its sign/label; subgroup by foil congruence.',cue_timing='Cued-orientation cue-only arm affects instruction frame0, sample arms affect1,2 where glyph also remains visible.',no_training=True,local_cap_seconds=14400))
        self.psychometrics();self.interventions();groups=self.atlas()
        assert all(torch.equal(original[k],v.cpu()) for k,v in self.model.state_dict().items())
        assert old_hashes=={name:sha(CHECKPOINT.with_name(name+'.pt')) for name in ('best','latest')}
        self.phase='inference_complete';self.status(groups=groups)
        atomic_json(OUT/'completion.json',dict(status='inference_complete',presentations=self.n,groups=groups,seconds=time.time()-self.started,checkpoint_hashes_unchanged=True,all_model_tensors_bitwise_unchanged=True,time=time.time()))

def main():
    lock=GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:Runner().main()
    except Exception:
        atomic_json(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
    finally:lock.close()

if __name__=='__main__':main()
