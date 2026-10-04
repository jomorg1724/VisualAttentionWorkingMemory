"""Independent artifact/metric verification, no new inference or fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import json,time,signal
from pathlib import Path
import numpy as np
from diagnostic import class_metrics,angular_error,circular_targets,sha,dump,CHECKPOINT,ROOT
out=Path(__file__).parent;b=json.loads((out/'budget.json').read_text());left=b['deadline']-time.time()
assert left>0,'Nonrenewable deadline expired'
signal.setitimer(signal.ITIMER_REAL,left)
p=np.load(out/'test_predictions.npz');m=json.loads((out/'metrics.json').read_text());s=json.loads((out/'selection.json').read_text())
for key,z in s.items():
    scores=p[z['archive_prefix']]
    if z['target']=='angle':
        patch=int(key.rsplit('/',1)[1]);phase=0 if '__baseline/' in key else 1
        err=angular_error(p['angles'][:,phase,patch],scores)
        assert float(err.mean())==m[key]['mean_absolute_circular_error_degrees']
    else:
        truth=p['labels'] if z['target'] in ('label','shuffled_label') else p[z['target']]
        actual=class_metrics(truth,scores)
        for metric,v in actual.items():assert v==m[key][metric],(key,metric)
for split in ('train','val','test'):
    rows=json.loads((out/f'{split}_metadata.json').read_text());dots=np.load(out/f'{split}_dot_angles.npz')['angles']
    assert len(dots)*2==len(rows)
    for i,z in enumerate(rows[::2]):
        baseline=z['baseline_transitions'];angles=dots[i,:baseline+8]
        assert np.array_equal(circular_targets(angles,baseline),np.array(z['actual_circular_angles_radians']))
        assert np.isnan(dots[i,baseline+8:]).all()
identity=json.loads((out/'identity.json').read_text())
for f,h in identity['source_hashes'].items():assert sha(ROOT/f)==h
assert sha(out/'diagnostic.py')==identity['extractor_sha256']
assert sha(CHECKPOINT)==identity['checkpoint_sha256']
result=dict(complete=True,probe_count=len(s),all_metrics_recomputed_exactly=True,actual_dot_angle_targets_recomputed_exactly=True,source_and_checkpoint_hashes_unchanged=True,elapsed_from_original_start_seconds=time.time()-b['started'],within_original_cap=time.time()<b['deadline'],report_sha256=sha(out/'REPORT.md'))
dump(out/'verified_completion.json',result);print(json.dumps(result,indent=2));signal.setitimer(signal.ITIMER_REAL,0)
