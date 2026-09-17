import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def paths():
    remote=Path(read(HERE.parent/'retrieval_receipt.json')['results'])
    control=read(ROOT/'WorkingMemory/SelectiveMaintenance/results.json')
    return dict(attention=remote/'preupdate_attention/checkpoint_008400.pt',continuation=Path(control['runs']['continuation']['selected_checkpoint']))
def models():
    import torch
    from WorkingMemory.PreUpdateAttention.model import AttentionMemory
    from WorkingMemory.SpatialComparison.model import Competitor
    result={}
    for arm,p in paths().items():
        cp=torch.load(p,map_location='cpu');assert cp['step']==8400
        model=AttentionMemory(activation_checkpoint=False) if arm=='attention' else Competitor('spatial_ei',activation_checkpoint=False)
        model.load_state_dict(cp['model'],strict=True);model.eval();model.requires_grad_(False);result[arm]=model
    return result
