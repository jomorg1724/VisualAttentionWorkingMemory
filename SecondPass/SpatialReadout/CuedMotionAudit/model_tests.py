"""CPU paired-scene acceptance tracer; never activates MPS."""
import importlib.util
from pathlib import Path
import torch
p=Path(__file__).with_name('model_diagnostic.py')
assert p.exists(), 'Frozen diagnostic implementation missing'
spec=importlib.util.spec_from_file_location('diagnostic',p); d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
r=d.duration_records(8)
assert len(r)==3
assert torch.equal(r[0]['y'],r[1]['y'])
assert all(a['checks']['matched_delay_nonblank_exact'] for a in r[0]['metadata'])
k=d.krauzlis_records(8)
assert torch.equal(k[0]['x'][:,2:],k[1]['x'][:,2:])
assert all(int(y)==int(m['event_type']=='target') for y,m in zip(k[1]['y'],k[1]['metadata']))
print('PASS native duration matched delay/cue preservation; Krauzlis movie exact and labels recomputed')
