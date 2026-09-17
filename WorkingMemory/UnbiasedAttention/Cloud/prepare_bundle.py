"""Portable immutable source, existing photograph fixtures and exact parent."""
import json,shutil,tarfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import torch
    receipt=HERE/'cloud_provisioning.json';cfg=read(receipt) if receipt.exists() else dict(local_run=str(HERE/'runs/prepared'),deadline_unix=None,pod_id=None)
    run=Path(cfg['local_run']);bundle=run/'portable_bundle';bundle.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/checkpoint_008400.pt';cp=torch.load(parent,map_location='cpu')
    names=set(cp['source_hashes']);names.update('WorkingMemory/TrainingExposure/'+n for n in ('protocol.py','sweep.py'))
    names.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (HERE.parent).glob('*.py'))
    names.add('WorkingMemory/SpatialTaskBattery/stimuli.py')
    hashes={n:sha(ROOT/n) for n in sorted(names)}
    for name in names:
        target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    photos=ROOT/'PreAttentiveVision/data/bsds500';photo_manifest=read(photos/'manifest.json')
    fixtures=[photos/'manifest.json']+[photos/r['file'] for r in photo_manifest['images']]
    for p in fixtures:
        target=bundle/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for folder in (HERE.parent,ROOT/'WorkingMemory/SpatialTaskBattery'):
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ('.md','.json'):
                target=bundle/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (bundle/'portable').mkdir(exist_ok=True);shutil.copy2(parent,bundle/'portable/parent_checkpoint_008400.pt')
    manifest=dict(pod_id=cfg['pod_id'],deadline_unix=cfg['deadline_unix'],source_hashes=hashes,files={str(p.relative_to(bundle)).replace('\\','/'):sha(p) for p in bundle.rglob('*') if p.is_file() and p!=bundle/'portable/manifest.json'})
    (bundle/'portable/manifest.json').write_text(json.dumps(manifest,indent=2));archive=run/'unbiased_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in bundle.rglob('*'):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(bundle)))
    result=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest['files']),source_hashes=hashes,parent_sha256=sha(parent))
    (run/'bundle_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
if __name__=='__main__':main()
