"""Supplement test: native event-removal pairs preserve all preevent pixels."""
import importlib.util
from pathlib import Path
import torch
p=Path(__file__).with_name('model_evidence.py')
assert p.exists(), 'Evidence intervention missing'
spec=importlib.util.spec_from_file_location('evidence',p);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
a,b=d.event_removal_records(8)
for i,m in enumerate(a['metadata']):
    assert torch.equal(a['x'][i,:m['virtual_event_frame']],b['x'][i,:m['virtual_event_frame']])
assert not b['y'].any()
print('PASS native-law catch counterparts preserve all preevent input; labels negative')
