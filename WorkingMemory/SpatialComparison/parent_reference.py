"""Optional unchanged14800 inference on existing tasks after both required tests."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,time,json,copy,threading,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from WorkingMemory.SpatialComparison.stimuli import SpatialStream,FAMILY_HEAD
from WorkingMemory.RecurrentComparison.model import RecurrentOpponent
from WorkingMemory.stimuli import TASK_CLASSES
from WorkingMemory.SpatialComparison.analyze import metrics
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def main():
    start=time.time();agg=read(P/'results.json');assert agg['status']=='completed';run=Path(agg['run_root']);budget=read(run/'budget.json');assert budget['active'] is None and all(e['returncode']==0 for e in budget['events'])
    deadline=budget['deadline_unix'];assert deadline-time.time()>180
    timer=threading.Timer(deadline-time.time()-30,lambda:os._exit(124));timer.daemon=True;timer.start()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp_path=Path(agg['config']['parent']);assert hashlib.sha256(cp_path.read_bytes()).hexdigest()==agg['config']['parent_sha256']
    cp=torch.load(cp_path,map_location='cpu');assert cp['version']=='ei_retention_core_readout_v1' and cp['step']==14800
    model=RecurrentOpponent(TASK_CLASSES,'ei_adaptive',activation_checkpoint=False);model.load_state_dict(cp['model'],strict=True);del cp;model.requires_grad_(False);model.eval();model.cuda();rows=[]
    with (P/'parent_predictions.jsonl').open('w') as f:
        for family in ('orientation_single','motion_direction'):
            stream=SpatialStream(agg['config']['test_seed'],'test');cells=[(n,c) for n,c in agg['config']['recipe']['cells'].items() if c['family']==family]
            for offset in range(0,512,8):
                state=copy.deepcopy(stream.state_dict())
                for name,c in cells:
                    stream.load_state_dict(state);x,y,meta=stream.batch(8,family,c['condition'])
                    with torch.inference_mode():probs=model(x.cuda(),FAMILY_HEAD[family]).softmax(1).cpu().tolist()
                    for j,(label,p,m) in enumerate(zip(y.tolist(),probs,meta)):
                        row=dict(condition=name,label=label,probabilities=p,paired_base_id=f'{family}/id/{offset+j}',metadata=m);rows.append(row);f.write(json.dumps(row)+'\n')
                    f.flush()
            print(json.dumps(dict(parent_reference_completed_family=family,elapsed=time.time()-start)),flush=True)
    del model;torch.cuda.empty_cache();bycell={}
    for row in rows:bycell.setdefault(row['condition'],[]).append(row)
    other={arm:[json.loads(x) for x in Path(v['test']['predictions']).read_text().splitlines()] for arm,v in agg['runs'].items()};rng=np.random.default_rng(37973001);result={}
    for cell,rs in bycell.items():
        y=np.array([r['label'] for r in rs]);p=np.array([r['probabilities'] for r in rs]);boot=np.array([np.concatenate([rng.choice(np.flatnonzero(y==c),sum(y==c),replace=True) for c in np.unique(y)]) for _ in range(1000)]);base=metrics(y,p);basecorrect=p.argmax(1)==y
        base['ba_ci95']=np.quantile([metrics(y[ix],p[ix])['ba'] for ix in boot],[.025,.975]).tolist();result[cell]=dict(parent=base,comparisons={})
        for arm,allrows in other.items():
            candidate=[r for r in allrows if r['condition']==cell];assert all(a['paired_base_id']==b['paired_base_id'] and a['label']==b['label'] and a['metadata']==b['metadata'] for a,b in zip(rs,candidate)) and len(candidate)==512
            q=np.array([r['probabilities'] for r in candidate]);diff=(q.argmax(1)==y).astype(float)-basecorrect
            result[cell]['comparisons'][arm]=dict(ba_difference=float(diff.mean()),ba_difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist())
    summary=dict(status='completed',checkpoint=str(cp_path),parent_step=14800,cells=result,n_per_cell=512,model_updates=0,worker_seconds=time.time()-start,deadline_epoch=deadline,scope='unchanged reference on existing single-item and motion tasks only; no binding zero-shot inference')
    (P/'parent_reference.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');timer.cancel();print(json.dumps(dict(status='parent_reference_completed',seconds=summary['worker_seconds'])),flush=True)
if __name__=='__main__':main()
