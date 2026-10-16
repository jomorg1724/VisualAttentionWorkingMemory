import copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from .design import *
from ..dataset import Stream,private_numpy,native_env,adapt_pixels
from ..native.sequences import collect_supervised_sequence
from ..model import ExpandedSuite
from ..Continuation.resume import verify_sources
from SecondPass.PixelContrastiveVAE.train import configure,GPULock,atomic_json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def renderer_checks():
    checks={}
    for size in (4,9,16):
        s=seed(203201,size,7)
        with private_numpy(np.random.RandomState(s)):
            native=adapt_pixels(collect_supervised_sequence(native_env(size,65)))
        x,_=movie(size,7,18,'cued',.75,overrides=False)
        assert torch.equal(x,native)
        a,_=movie(size,7,0,'uncued',1,cue=size-1)
        b,_=movie(size,7,0,'catch',1,cue=size-1)
        assert torch.equal(a,b)
        c,_=movie(size,7,18,'uncued',1,cue=size-1)
        assert torch.equal(b[:5],c[:5])
        cells,off,occupancy=masks(size)
        assert np.all(cells.sum(0)==1) and not np.any(off&(cells[0]|cells[-1]))
        checks[str(size)]=dict(native_pixels_exact=True,zero_angle_matches_catch=True,
                              unchanged_prechange_frames=True,site_token_counts=cells.sum(1).tolist(),
                              offsite_gabor_occupancy=occupancy)
    return checks

class Runner:
    def __init__(self):
        self.started=time.time();self.deadline=json.loads((OUT/'budget.json').read_text())['deadline']
        self.n=0;self.phase='checks';self.maps={s:masks(s) for s in (4,9,16)};self.atlas={}
    def status(self,**kw):
        if time.time()>self.deadline-30:raise TimeoutError('Finite analysis allocation exhausted')
        atomic_json(OUT/'status.json',dict(phase=self.phase,presentations=self.n,elapsed=time.time()-self.started,**kw))
    def infer(self,x,size,b=None):
        self.status(size=size)
        pp=[];aa=[]
        with torch.inference_mode():
            for i in range(0,len(x),4):
                z,d=self.model(x[i:i+4].cuda(),f'vda{size}',True,None if b is None else torch.as_tensor(b,device='cuda'))
                p=z.softmax(-1).cpu().numpy();a=d['attention'].cpu().numpy()
                assert np.isfinite(p).all() and np.isfinite(a).all() and np.allclose(a.sum((2,3)),1,atol=1e-5)
                pp.extend(p);aa.extend(a);self.n+=len(p)
        return np.asarray(pp),np.asarray(aa)
    def record(self,phase,x,meta,p,a,arm='sham'):
        rows=[]
        for xx,m,prob,attention in zip(x.numpy(),meta,p,a):
            cells,off,_=self.maps[m['size']]
            mass={key:(attention.reshape(7,100)*mask).sum(1).tolist() for key,mask in
                  [('cued',cells[m['cue']]),('uncued',cells[m['uncued']]),('offsite',off)]}
            rows.append(dict(**m,arm=arm,p1=float(prob[1]),choice=int(prob.argmax()),
                             pixels_sha256=hashlib.sha256(xx.tobytes()).hexdigest(),attention_mass=mass))
        with (OUT/f'{phase}.jsonl').open('a',encoding='utf8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    def add_atlas(self,x,meta,p,a,arm):
        for xx,m,prob,aa in zip(x.numpy(),meta,p,a):
            key=f'vda{m["size"]}_cue{m["cue"]}_{m["event"]}_{arm}'
            if key not in self.atlas:
                self.atlas[key]=dict(n=0,sx=np.zeros_like(xx,dtype=np.float64),sa=np.zeros_like(aa,dtype=np.float64),
                    examples=[],maps=[],probabilities=[],metadata=m,arm=arm)
            g=self.atlas[key];g['n']+=1;g['sx']+=xx;g['sa']+=aa
            if len(g['examples'])<3:
                g['examples'].append(xx.copy());g['maps'].append(aa.copy());g['probabilities'].append(prob.tolist())
    def native(self):
        self.phase='fresh_native'
        stream=Stream('test')
        for size in (4,9,16):
            for i,theta in enumerate((8,26,65)):
                key=(f'vda{size}',f'TH{theta}');stream.rngs[key]=np.random.RandomState(seed(203101,size,i));stream.counts[key]=0
                for _ in range(64):
                    x,y,mm=stream.batch(8,*key);p,a=self.infer(x,size);meta=[]
                    for rep,(m,label) in enumerate(zip(mm,y.tolist())):
                        meta.append(dict(size=size,theta=theta,rep=m['trial_id'],cue=m['cue_index'],uncued=size-1-m['cue_index'],
                            event='catch' if label==0 else 'cued' if m['cue_index']==m['change_index'] else 'uncued',
                            angle=abs(m['orientation_change']),validity=m['displayed_validity'],color=m['cue_color'],value=m['value'],
                            label=label,change_index=m['change_index'],cluster=f'203101:{size}:{theta}:{m["trial_id"]}',
                            counterfactual_cue_violation=False))
                    self.record('native',x,meta,p,a)
    def psychometrics(self):
        self.phase='psychometrics'
        for size in (4,9,16):
            validities=[0,.25,.5,.75,1] if size==9 else [.25,.5,.75,1]
            for validity in validities:
                cases=[(0,'catch')]+[(angle,event) for angle in ANGLES for event in ('cued','uncued')
                      if not (validity==1 and event=='uncued') and not (validity==0 and event=='cued')]
                for angle,event in cases:
                    for start in range(0,96,8):
                        trials=[movie(size,r,angle,event,validity,namespace=203201) for r in range(start,start+8)]
                        x=torch.stack([v[0] for v in trials]);mm=[v[1] for v in trials];p,a=self.infer(x,size)
                        self.record('psychometrics',x,mm,p,a)
    def interventions(self,primary=False):
        self.phase='primary_hypothesis' if primary else 'interventions'
        selected_arms=PRIMARY_ARMS if primary else arms()
        for size in ((4,) if primary else (4,9,16)):
            for geometry in ((1,) if primary else (0,1)):
                cue=size-1 if geometry else 0
                for validity in ([1.] if primary else [.75,1.]):
                    cases=[(0,'catch')]+[(angle,event) for angle in ([18] if primary else [18,40]) for event in ('cued','uncued')]
                    for angle,event in cases:
                        count=384 if primary else 48
                        for start in range(0,count,8):
                            mm=[];xx=[]
                            for r in range(start,start+8):
                                x,m=movie(size,r,angle,event,validity,cue=cue,namespace=203403 if primary else 203301)
                                m['geometry']=geometry;m['cluster']+=f':geometry{geometry}'
                                xx.append(x);mm.append(m)
                            x=torch.stack(xx);reference=None
                            for arm in selected_arms:
                                b=bias(mm[0],arm,self.maps[size]);p,a=self.infer(x,size,b)
                                if arm['name']=='sham':reference=(p,a)
                                if arm['site']=='global':
                                    assert np.allclose(p,reference[0],atol=2e-5) and np.allclose(a,reference[1],atol=2e-5)
                                    assert np.array_equal(p.argmax(1),reference[0].argmax(1))
                                self.record('primary' if primary else 'interventions',x,mm,p,a,arm['name'])
                                wanted={'sham','cued_-4_change','cued_+4_change','uncued_-4_change','uncued_+4_change'}
                                if not primary and validity==1 and angle in (0,18) and arm['name'] in wanted:
                                    self.add_atlas(x,mm,p,a,arm['name'])
    def save_atlas(self):
        folder=OUT/'atlas';folder.mkdir(exist_ok=True);index=[]
        for key,g in self.atlas.items():
            np.savez_compressed(folder/f'{key}.npz',images=np.array(g['examples']),attention=np.array(g['maps']),
                mean_images=(g['sx']/g['n']).astype(np.float32),mean_attention=(g['sa']/g['n']).astype(np.float32))
            index.append(dict(id=key,n=g['n'],metadata=g['metadata'],arm=g['arm'],probabilities=g['probabilities'],phases=PHASES))
        atomic_json(folder/'index.json',index)
        return len(index)
    def run(self):
        configure('cuda');checks=renderer_checks();saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        verify_sources(saved);old_hashes={name:sha(CHECKPOINT.with_name(name+'.pt')) for name in ('best','latest')}
        assert saved['step']==16000
        self.model=ExpandedSuite().cuda();self.model.load_state_dict(saved['model']);self.model.eval().requires_grad_(False)
        original={n:v.cpu().clone() for n,v in self.model.state_dict().items()}
        x,m=movie(4,0,18,'uncued',1,cue=3)
        p,a=self.infer(x[None],4);pp,aa=self.infer(x[None],4,np.zeros((7,100),np.float32))
        assert np.array_equal(p,pp) and np.array_equal(a,aa)
        self.n=0
        atomic_json(OUT/'checks.json',dict(renderer=checks,zero_bias_exact=True,all_weights_frozen=True,optimizer_created=False))
        atomic_json(OUT/'protocol.json',dict(checkpoint=str(CHECKPOINT),checkpoint_step=16000,checkpoint_hashes=old_hashes,
            original_vda_sources=saved['config']['source_sha256'],analysis_sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            primary='VDA4: 100% bottom-right cue, top-left change; bottom-right inhibition ONLY frames5,6',
            primary_angle=18,primary_clusters=384,primary_arms=PRIMARY_ARMS,primary_events=['cued','uncued','catch'],
            primary_directional_tests='Four benefits and four reciprocal costs at dose4; exact paired binary tests, Holm across8',
            main_clusters_per_geometry=48,main_validities=[.75,1.],main_angles=[18,40],main_arms=arms(),
            psychometric_angles=ANGLES,psychometric_clusters=96,native_trials_per_theta=512,
            namespaces=[203101,203201,203301,203403],attention='Actual single CLS-query row; query average equals this row; no NxN matrix',
            site='All model patch centers in the native stimulus cell; per-token additive priority-logit dose',
            cue_violation='Uncued change with100% cue is a deliberate counterfactual, absent from native training distribution',
            labels='All physical changes, including uncued, are correct change reports; false alarms are catch reports only',
            budget_seconds=7200,no_training=True))
        # The requested hypothesis is measured first, before exploratory sweeps.
        self.interventions(primary=True);self.native();self.psychometrics();self.interventions()
        groups=self.save_atlas()
        assert all(torch.equal(original[n],v.cpu()) for n,v in self.model.state_dict().items())
        assert old_hashes=={name:sha(CHECKPOINT.with_name(name+'.pt')) for name in ('best','latest')}
        self.phase='inference_complete';self.status(groups=groups)
        atomic_json(OUT/'completion.json',dict(presentations=self.n,groups=groups,elapsed=time.time()-self.started,
                    checkpoint_hashes_unchanged=True,model_tensors_bitwise_unchanged=True,status='inference_complete'))

def main():
    lock=GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:Runner().run()
    except BaseException:
        atomic_json(OUT/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
    finally:lock.close()

if __name__=='__main__':main()
