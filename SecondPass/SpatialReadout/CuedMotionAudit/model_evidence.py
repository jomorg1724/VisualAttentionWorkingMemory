"""Small causal evidence supplement within the ORIGINAL model_budget deadline."""
from SecondPass.SpatialReadout.CuedMotionAudit.model_diagnostic import *
class EventCapture(SpatialBatteryStream):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.event_states=[]
    def _krauzlis(self,rng,label,target,cfg):
        self.event_states.append((copy.deepcopy(rng.bit_generator.state),label,target,dict(cfg),self._event_kind))
        return super()._krauzlis(rng,label,target,cfg)
def event_removal_records(n,stream=None):
    st=stream if stream is not None else EventCapture(92735039,'test');start=len(st.event_states)
    x,y,m=st.batch(n,'krauzlis_cued_motion',dict(baseline_transitions=20));other=[];meta=[]
    for i,(state,label,target,cfg,event) in enumerate(st.event_states[start:]):
        rng=np.random.default_rng();rng.bit_generator.state=state;st._event_kind='catch'
        frames,mm=SpatialBatteryStream._krauzlis(st,rng,0,target,cfg);xx=torch.from_numpy(np.stack(frames));t=m[i]['virtual_event_frame']
        assert torch.equal(x[i,:t],xx[:t]);assert event!='catch' or torch.equal(x[i],xx)
        mm.update(trial_id=m[i]['trial_id'],label=0,original_event_type=event,original_label=int(y[i]),paired_preevent_exact=True)
        other.append(xx);meta.append(mm)
    return [dict(task='krauzlis_cued_motion',condition='native_B20_replicate',x=x,y=y,metadata=m),dict(task='krauzlis_cued_motion',condition='native_law_event_removed_B20',x=torch.stack(other),y=torch.zeros_like(y),metadata=meta)]

def run_supplement():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    budget=json.loads((OUT/'model_budget.json').read_text());deadline=budget['deadline_unix'];assert time.time()<deadline-180
    assert json.loads((OUT/'model_completion.json').read_text())['status']=='complete'
    assert sha(CKPT)==SHA;cp=torch.load(CKPT,map_location='cpu',weights_only=False);model=SpatialReadout(task_classes());model.load_state_dict(cp['model']);model.eval();model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    def cap(*args):os._exit(124)
    signal.signal(signal.SIGALRM,cap);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    dump('model_evidence_budget.json',dict(original_start=budget['start_unix'],original_deadline=deadline,supplement_start=time.time(),pid=os.getpid(),renewed=False,reason='Cheap primary run leaves original budget; native-law event removal distinguishes actual event response from temporal drift; B12/28 native-length checks and explicitly OOD reversal.'))
    model.to('mps');rows=[];parity=[]
    def score(records):
        for r in records:
            assert time.time()<deadline-90
            xx=r['x'].to('mps');z=trajectory(model,xx,r['task']);err=float((model(xx,r['task'])-z[:,-1]).abs().max().cpu());assert err<=1e-6
            zz=z.cpu().tolist();rows.append(dict(task=r['task'],condition=r['condition'],labels=r['y'].tolist(),logits=[a[-1] for a in zz],trajectory=zz,metadata=r['metadata']));parity.append(err)
            with (OUT/'model_evidence_trials.jsonl').open('a') as f:f.write(json.dumps(rows[-1])+'\n')
    assert not (OUT/'model_evidence_trials.jsonl').exists()
    # Same production scenes/seeds as primary diagnostic.
    for b in range(0,128,16):
        r=duration_records(16,92635039+b)[0];rr=copy.deepcopy(r);rr['condition']='OOD_time_reversal_D0';rr['x']=r['x'].clone();ts=list(range(1,10));rr['x'][:,ts]=r['x'][:,list(reversed(ts))];rr['y']=(r['y']+2)%4
        for i,m in enumerate(rr['metadata']):
            m.update(directions_by_patch=((np.array(m['directions_by_patch'])[:,::-1]+2)%4).tolist(),winner_by_patch=((np.array(m['winner_by_patch'])+2)%4).tolist(),duration_counts_by_patch=np.array(m['duration_counts_by_patch'])[:,[2,3,0,1]].tolist(),label=int(rr['y'][i]),OOD='Reversed native raster movie flips surviving displacement vectors; backward replacement process is not native rendering law.')
        score([rr])
    st=EventCapture(92735039,'test')
    for b in range(0,200,8):score(event_removal_records(8,st))
    for baseline in (12,28):
        st=SpatialBatteryStream(92935039+baseline,'test')
        for b in range(0,100,8):score(krauzlis_records(min(8,100-b),stream=st,baseline=baseline)[:1])
    frozen=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items());assert frozen and sha(CKPT)==SHA
    rows=[json.loads(a) for a in (OUT/'model_evidence_trials.jsonl').read_text().splitlines()]
    groups={}
    for r in rows:
        g=groups.setdefault(r['condition'],dict(labels=[],logits=[],trajectory=[],metadata=[]))
        for k in g:g[k].extend(r[k])
    summary={'conditions':[]}
    for k,v in groups.items():
        if len(set(v['labels']))==1:
            pred=np.array(v['logits']).argmax(1);metric=dict(n=len(pred),accuracy=float((pred==np.array(v['labels'])).mean()),balanced_accuracy=None,auc_macro=None,prediction_counts=np.bincount(pred,minlength=2).tolist(),reason='All-negative counterfactual: BA/AUC undefined; report specificity/accuracy instead')
        else:metric=metrics(v['labels'],v['logits'])
        summary['conditions'].append(dict(condition=k,**metric))
    a=groups['native_B20_replicate'];b=groups['native_law_event_removed_B20'];p=probs(a['trajectory'])[:,:,1];q=probs(b['trajectory'])[:,:,1];events=np.array([m['event_type'] for m in a['metadata']])
    primary=[json.loads(a) for a in (OUT/'model_trials.jsonl').read_text().splitlines()];pz=np.array([z for r in primary if r['condition']=='native_B20' for z in r['logits']]);assert np.array_equal(pz,np.array(a['logits']))
    summary['event_present_minus_removed_p1']={e:{epoch:bootstrap((p-q)[events==e,t]) for epoch,t in [('pre_event',27),('first_postevent',28),('last_moving',35),('report',36)]} for e in ('target','foil','catch')}
    summary['event_removal_argmax_changes']=int((np.array(a['logits']).argmax(1)!=np.array(b['logits']).argmax(1)).sum())
    base={k:[] for k in ('labels','logits')}
    for r in primary:
        if r['condition']=='native_D0':
            for k in base:base[k].extend(r[k])
    summary['duration_reversal_pair']=paired(base,groups['OOD_time_reversal_D0'])
    summary.update(original_deadline=deadline,elapsed_from_original_start=time.time()-budget['start_unix'],all_weights_bitwise_unchanged=frozen,checkpoint_hash_unchanged=True,max_wrapper_parity_error=max(parity),native_replicate_logits_exact=True,training_updates=0,probe_fits=0,completed_unix=time.time())
    dump('model_evidence_summary.json',summary);assert time.time()<deadline;signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run_supplement()
