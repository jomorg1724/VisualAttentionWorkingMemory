"""Statistical and gallery contract tests on tiny explicitly synthetic fixtures."""
import importlib.util
from pathlib import Path

def test_pairing_and_finite_rates():
    path=Path(__file__).with_name('render.py')
    assert path.exists(), 'Statistical renderer not implemented'
    spec=importlib.util.spec_from_file_location('analysis_render',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rate=module.binomial([1]*8)
    assert rate['low']<1 and rate['high']==1 and rate['n']==8
    for n in range(1,257):
        for value in (0,1):
            rate=module.binomial([value]*n)
            assert rate['low']<=rate['rate']<=rate['high'], 'Finite roundoff cannot produce negative error bars'
    rows=[{'label':0,'prediction':0,'correct':True,'prob1':.1,'base_id':'a'}, {'label':1,'prediction':1,'correct':True,'prob1':.9,'base_id':'b'}]
    s=module.summary(rows)
    assert s['dprime']>0 and s['criterion']==0 and s['balanced_accuracy']==1
    paired=module.paired_effect(rows,rows)
    assert paired['delta_accuracy']==0 and paired['accuracy_low']==0 and paired['accuracy_high']==0
    assert paired['response_flip_wilson_high']>0, 'No observed flips does not prove zero future flip probability'
    relabeled=[dict(r,label=1-r['label'],prediction=1-r['prediction'],prob1=1-r['prob1']) for r in rows]
    shifted=module.paired_effect(rows,relabeled)
    assert shifted['dprime_low']==0 and shifted['dprime_high']==0, 'Reassigned targets require condition-specific signal labels'
