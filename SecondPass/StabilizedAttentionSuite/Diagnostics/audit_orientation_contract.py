"""Read-only CPU audit of the running experiment's cued-orientation contract.

No optimizer, CUDA, model writes, final-test evaluation, or training interruption.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import collections, datetime, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch
from SecondPass.StabilizedAttentionSuite.dataset import Stream
from SecondPass.StabilizedAttentionSuite.model import StabilizedAttentionSuite
from WorkingMemory.SpatialTaskBattery.stimuli import CENTERS
from WorkingMemory.BatteryAudit.observers import GaborBank, decode_sign_glyph, wrap_half_pi

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'SecondPass/StabilizedAttentionSuite/LocalRuntime/continuation02'
OUT = Path(__file__).with_name('orientation_contract_run01')
TASK = 'orientation_cued'
CELLS = ('D0', 'D4', 'D12', 'D24')
START = time.monotonic()

def bounded():
    if time.monotonic()-START > 600:
        raise TimeoutError('Read-only CPU audit exceeded 600 seconds')

def emit(stage, value):
    bounded()
    print(json.dumps({'stage': stage, 'result': value}), flush=True)

def metrics(y, p):
    y=np.asarray(y);p=np.asarray(p);pred=p.argmax(1)
    confusion=np.zeros((2,2),dtype=int)
    for a,b in zip(y,pred):confusion[a,b]+=1
    pos=p[y==1,1];neg=p[y==0,1]
    auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
    return dict(n=len(y),confusion=confusion.tolist(),balanced_accuracy=float(np.mean(np.diag(confusion)/confusion.sum(1))),auc=auc)

def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    payload=(RUN/'best.pt').read_bytes()
    saved=torch.load(io.BytesIO(payload),map_location='cpu')
    config=saved['config'];step=saved['state']['step']
    source_checks={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in config['source_hashes'].items()}
    assert all(source_checks.values())
    out=dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_step=step,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),source_hashes_match=source_checks,
        scope='CPU2 only; frozen in-memory model; no optimizer or original file mutations; 600-second cap')
    class MatchedStream(Stream):
        def stream_seed(self,task,cell):return 791001+self._cell(task,cell)[0]
    equality=[]
    for cell in CELLS:
        a=MatchedStream('train');b=MatchedStream('val');c=MatchedStream('train')
        x,y,m=a.batch(32,TASK,cell);xx,yy,mm=b.batch(32,TASK,cell)
        chunks=[c.batch(4,TASK,cell) for _ in range(8)]
        assert torch.equal(x,xx) and torch.equal(y,yy)
        assert torch.equal(x,torch.cat([v[0] for v in chunks])) and torch.equal(y,torch.cat([v[1] for v in chunks]))
        equality.append(dict(cell=cell,n=32,train_val_same_seed_pixels_labels_bit_exact=True,batch32_vs_8x4_bit_exact=True))
    out['split_and_batch_equivalence']=equality;emit('split_and_batch_equivalence',equality)
    bank=GaborBank();pixel=[]
    for split in ('train','val'):
        stream=Stream(split)
        for cell in CELLS:
            counts=collections.Counter();errors=[];cases=collections.Counter()
            for _ in range(4):
                x,y,metadata=stream.batch(32,TASK,cell)
                for frames,label,m in zip(x.numpy(),y.tolist(),metadata):
                    target,sign,_=decode_sign_glyph(frames[0]);counts['n']+=1
                    counts['glyph_matches_metadata']+=int(target==m['target_location'] and sign==m['cue_sign'])
                    angle_delta=wrap_half_pi(np.array(m['probe_angles_radians'])-np.array(m['sample_angles_radians']))
                    truth=int(sign*angle_delta[target]>1e-6)
                    counts['independent_angle_label_match']+=int(truth==label)
                    cx,cy=CENTERS[target]
                    sample,_=bank.angle(frames[2,0],cx,cy);probe,_=bank.angle(frames[-1,0],cx,cy)
                    measured=float(np.rad2deg(wrap_half_pi(probe-sample)))
                    expected=m['rotations_degrees'][target];errors.append(abs(measured-expected))
                    # Fixed halfway threshold below the smallest 15-degree signal;
                    # never selected using these data or the network's predictions.
                    pixel_label=int(sign*measured>7.5)
                    counts['pixel_observer_label_match']+=int(pixel_label==label)
                    cases[(label,target,sign)]+=1
                bounded()
            assert counts['glyph_matches_metadata']==counts['independent_angle_label_match']==counts['n']
            pixel.append(dict(split=split,cell=cell,**counts,max_pixel_angle_error_degrees=max(errors),balanced_label_location_sign_counts=dict((str(k),v) for k,v in cases.items())))
    out['pixel_label_audit']=pixel;emit('pixel_label_audit',pixel)
    model=StabilizedAttentionSuite();model.load_state_dict(saved['model']);model.requires_grad_(False)
    modes=[]
    with torch.no_grad():
        s=Stream('val')
        for cell in CELLS:
            x,y,m=s.batch(4,TASK,cell)
            model.train();a,details=model(x,TASK,True)
            model.eval();b=model(x,TASK);single=torch.cat([model(z[None],TASK) for z in x])
            assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
            assert torch.allclose(b,single,atol=1e-4,rtol=1e-4)
            modes.append(dict(cell=cell,train_details_vs_eval_max_logit_difference=float((a-b).abs().max()),batch4_vs_single_max_logit_difference=float((b-single).abs().max())))
    out['forward_equivalence']=modes;emit('forward_equivalence',modes)
    recorded=json.loads((RUN/('validation_%06d.json'%step)).read_text())
    model.eval();results=[]
    with torch.no_grad():
        for split in ('train','val'):
            stream=Stream(split)
            for cell in CELLS:
                ys=[];ps=[];metas=[]
                for _ in range(16):
                    x,y,m=stream.batch(4,TASK,cell);ps.extend(model(x,TASK).softmax(-1).tolist());ys.extend(y.tolist());metas.extend(m);bounded()
                row=dict(split=split,cell=cell,**metrics(ys,ps))
                if split=='val':
                    old=next(c for c in recorded['cells'] if c['task']==TASK and c['cell']==cell)
                    oy=[p['label'] for p in old['predictions']];op=[p['probabilities'] for p in old['predictions']]
                    assert ys==oy and all(m==p['metadata'] for m,p in zip(metas,old['predictions']))
                    assert np.array_equal(np.argmax(ps,axis=1),np.argmax(op,axis=1))
                    independent=metrics(oy,op)
                    assert abs(independent['balanced_accuracy']-old['balanced_accuracy'])<1e-12
                    assert abs(independent['auc']-old['auc'])<1e-12
                    row.update(saved_labels_metadata_exact=True,saved_predictions_same=True,cpu_cuda_max_probability_difference=float(np.max(np.abs(np.array(ps)-np.array(op)))),independent_scoring_matches=True)
                results.append(row);emit('frozen_model_cell',row)
    out['frozen_model_train_val']=results
    out['summary']={split:{key:float(np.mean([r[key] for r in results if r['split']==split])) for key in ('balanced_accuracy','auc')} for split in ('train','val')}
    out['summary']['pixel_observer_correct']=sum(v['pixel_observer_label_match'] for v in pixel)
    out['summary']['pixel_observer_total']=sum(v['n'] for v in pixel)
    out['summary']['interpretation']='No demonstrated train/validation contract mismatch on cued orientation; scoped tests, not proof every possible bug is absent.'
    out['seconds']=time.monotonic()-START
    (OUT/'report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    emit('complete',out['summary'])

if __name__=='__main__':main()
