"""Saved-feature-only, bounded exploratory diagnostic; never imports neural model."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,time,hashlib,signal,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'FrozenDiagnostic'
ROOT=OUT.parents[4]
ALPHAS=[1.,100.,10000.]

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def access(data,kind,p):
    phases=['final']*3 if kind=='final' else ['cue','baseline','post']
    z=np.concatenate([np.roll(data['memory__'+ph],1045*j,axis=1)@p for j,ph in enumerate(phases)],axis=1)
    i,j=np.triu_indices(z.shape[1]);return np.concatenate([z,z[:,i]*z[:,j]],axis=1)
def relation(c,e,s):return e*((1-c)*(1-s)+c*s)
def calibration(c,e,s):return np.column_stack([c,e,s,c*e,c*s,e*s,c*e*s])
def ba(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
def predict(f,x):return (x-f['mean'])/f['scale']@f['weight']+f['intercept']
def targets(meta):
    joint=np.array([2 if m['changed_patch'] is None else m['changed_patch'] for m in meta])
    return dict(joint=joint,event=(joint!=2).astype(int),side=np.minimum(joint,1),cue=np.array([m['target_location'] for m in meta]),label=np.array([m['label'] for m in meta]))
def fit_candidates(x,y):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
    z=(x-mean)/scale;intercept=y.mean(0);yc=y-intercept
    dual=len(x)<x.shape[1];gram=z@z.T if dual else z.T@z
    eig,q=np.linalg.eigh(gram);eig=np.maximum(eig,0);rhs=q.T@(yc if dual else z.T@yc)
    for alpha in ALPHAS:
        w=q@(rhs/(eig[:,None]+alpha));w=z.T@w if dual else w
        yield dict(mean=mean,scale=scale,weight=w,intercept=intercept,alpha=np.array(alpha))
def binary(p):return np.column_stack([1-p,p])
def components(fits,x,kind):
    return [np.clip(predict(fits[kind+'/'+t],x)[:,1],0,1) for t in ['cue','event','side']]
def outputs(fits,x,kind):
    c,e,s=components(fits,x,kind)
    out={t:predict(fits[kind+'/'+t],x) for t in ['cue','event','side','joint','label','shuffled_label']}
    out.update(hard=binary(((e>=.5)&((c>=.5)==(s>=.5))).astype(float)),soft=binary(relation(c,e,s)),calibrated=predict(fits[kind+'/calibrator'],calibration(c,e,s)))
    return out

def main():
    assert not (OUT/'budget.json').exists(),'Nonrenewable budget already exists; no refit permitted'
    identity=json.loads((SRC/'identity.json').read_text());old=json.loads((SRC/'receipt.json').read_text())
    checks={}
    for name in ['train_features.npz','val_features.npz','test_features.npz','projections.npz']:
        checks[name]=sha(SRC/name);assert checks[name]==old['artifacts'][name]
    checks['checkpoint']=sha(Path(identity['checkpoint']));assert checks['checkpoint']==identity['checkpoint_sha256']
    checks['extraction_source']=sha(SRC/'diagnostic.py');assert checks['extraction_source']==identity['extractor_sha256']
    for name,h in identity['source_hashes'].items():assert sha(ROOT/name)==h
    meta={s:json.loads((SRC/f'{s}_metadata.json').read_text()) for s in ['train','val','test']}
    for s,n in [('train',425),('val',100),('test',175)]:
        m=meta[s];assert len(m)==2*n and len({v['group'] for v in m})==n
        for a,b in zip(m[::2],m[1::2]):
            assert a['group']==b['group'] and a['variant']==0 and b['variant']==1
            assert a['noncue_sha256']==b['noncue_sha256'] and a['changed_patch']==b['changed_patch']
            assert a['target_location']==1-b['target_location']
            for v in (a,b):assert v['label']==int(v['changed_patch']==v['target_location'])
    for field in ['group','trial_id','noncue_sha256','movie_sha256']:
        sets=[set(v[field] for v in meta[s]) for s in meta]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    rng=np.random.default_rng(92401);order=rng.permutation(425);ag=np.sort(order[:300]);bg=np.sort(order[300:])
    ar=(2*ag[:,None]+np.arange(2)).ravel();br=(2*bg[:,None]+np.arange(2)).ravel()
    assert not set(ar)&set(br)
    group_contract=dict(component_groups=[meta['train'][2*i]['group'] for i in ag],calibration_groups=[meta['train'][2*i]['group'] for i in bg],counts=dict(train=425,val=100,test=175,component=300,calibrator=125),disjoint_checks=True)
    dump(OUT/'split_contract.json',group_contract)
    checks.update({f'{s}_metadata.json':sha(SRC/f'{s}_metadata.json') for s in meta});dump(OUT/'input_receipt.json',checks)
    protocol=dict(status='exploratory reused-test; no new independent confirmation',access=['final','phases'],features='96 fixed projected coordinates plus quadratic upper triangle =4752',alphas=ALPHAS,selection='validation paired balanced accuracy, first tie; no test choices',component_groups=300,calibrator_groups=125,targets=['event','side conditional on event','joint left/right/catch','cue','label'],bootstrap=500,seed=92401,control='whole component group-pair label permutation',calibrator='7 multilinear predicted cue/event/side terms, fitted only on independent 125 train groups',capacity='matched final/phases; factorized auxiliary supervision differs from direct label-only',no_neural_extraction=True)
    dump(OUT/'protocol.json',protocol)
    p=np.load(SRC/'projections.npz')['32'];np.savez(OUT/'projection.npz',p=p)
    data={s:np.load(SRC/f'{s}_features.npz') for s in ['train','val']}
    xs={s:{k:access(data[s],k,p) for k in ['final','phases']} for s in data}
    ys={s:targets(meta[s]) for s in meta};fits={};selection={};arrays={}
    # Budget starts immediately before the first fit, not preparation or file validation.
    started=time.time();deadline=started+600
    dump(OUT/'budget.json',dict(started=started,deadline=deadline,seconds=600,threads=2,renewable=False))
    def expire(*args):raise TimeoutError('600-second nonrenewable cap expired')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,600)
    shuffled=np.random.default_rng(4433).permutation(len(ag));shuffle_rows=(2*ag[shuffled,None]+np.arange(2)).ravel()
    def select(key,x,y,vx,vy):
        candidates=list(fit_candidates(x,y));scores=[ba(vy,predict(f,vx).argmax(1)) for f in candidates]
        best=int(np.argmax(scores));fits[key]=candidates[best]
        selection[key]=dict(alpha=ALPHAS[best],validation_ba=scores,input_dimensions=x.shape[1],fit_rows=len(x),coefficients=int(candidates[best]['weight'].size),fit_groups='calibration' if key.endswith('calibrator') else 'component')
        for k,v in fits[key].items():arrays[key+'__'+k]=v
    for kind in ['final','phases']:
        x=xs['train'][kind];vx=xs['val'][kind]
        for target in ['cue','event','side','joint','label','shuffled_label']:
            name='label' if target=='shuffled_label' else target
            tr=ar[ys['train']['event'][ar]==1] if target=='side' else ar
            va=ys['val']['event']==1 if target=='side' else np.ones(len(vx),bool)
            ty=ys['train'][name][shuffle_rows] if target=='shuffled_label' else ys['train'][name][tr]
            select(kind+'/'+target,x[tr],np.eye(3 if target=='joint' else 2)[ty],vx[va],ys['val'][name][va])
        c,e,s=components(fits,x[br],kind);vc,ve,vs=components(fits,vx,kind)
        select(kind+'/calibrator',calibration(c,e,s),np.eye(2)[ys['train']['label'][br]],calibration(vc,ve,vs),ys['val']['label'])
    np.savez(OUT/'fitted_models.npz',**arrays);dump(OUT/'selection.json',selection)
    dump(OUT/'test_freeze.json',dict(time=time.time(),fits_sha256=sha(OUT/'fitted_models.npz'),selection_sha256=sha(OUT/'selection.json'),protocol_sha256=sha(OUT/'protocol.json'),test_reused=True,all_choices_frozen=True))
    test=np.load(SRC/'test_features.npz');saved={};y=ys['test']
    for kind in ['final','phases']:
        x=access(test,kind,p)
        scores=outputs(fits,x,kind)
        if kind=='final':
            minimal={'memory__final':test['memory__final']}
            poison=dict(minimal,**{'memory__'+t:np.full_like(test['memory__final'],np.nan) for t in ['cue','baseline','post']})
            for alternative in [minimal,poison]:
                alt=outputs(fits,access(alternative,kind,p),kind)
                for key in scores:assert np.array_equal(alt[key],scores[key])
        c,e,s=components(fits,x,kind);hc=(c>=.5).astype(int);he=(e>=.5).astype(int);hs=(s>=.5).astype(int)
        for name,cc,ee,ss in [('oracle_cue',y['cue'],he,hs),('oracle_event',hc,y['event'],hs),('oracle_side_on_events',hc,he,np.where(y['event'],y['side'],hs)),('oracle_event_side',hc,y['event'],y['side']),('oracle_all',y['cue'],y['event'],y['side'])]:scores[name]=binary(relation(cc,ee,ss))
        for key,v in scores.items():saved[kind+'/'+key]=v
    saved['native/label']=test['native_logits'];np.savez(OUT/'test_predictions.npz',**saved)
    for key,v in y.items():saved['truth/'+key]=v
    np.savez(OUT/'scoring_inputs.npz',**saved)
    import replay
    result=replay.verify_and_report(deadline)
    assert time.time()<deadline
    dump(OUT/'completion.json',dict(complete=True,elapsed_seconds=time.time()-started,within_cap=True,deadline=deadline,model_optimizer_steps=0,new_neural_extractions=0,checkpoint_unchanged=sha(Path(identity['checkpoint']))==identity['checkpoint_sha256'],threads=2,artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}))
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
