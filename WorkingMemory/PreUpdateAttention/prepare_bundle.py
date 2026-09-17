"""Freeze portable source, reviewed implementation, parent and actual initializer."""
import sys,json,shutil,tarfile,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha
from WorkingMemory.PreUpdateAttention.remote_sweep import recipe
from WorkingMemory.PreUpdateAttention.model import migrate
import torch
torch.set_num_threads(1)
root=HERE/'runs/attention_20260913_143459';bundle=root/'portable_bundle';bundle.mkdir(parents=True,exist_ok=True)
parent=ROOT/'WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/spatial_ei/checkpoint_004400.pt'
names=list(read(parent.parents[1]/'budget.json')['source_hashes'])+['WorkingMemory/PreUpdateAttention/'+n for n in ('model.py','train.py','remote_sweep.py')]
hashes={n:sha(ROOT/n) for n in names}
for name in names+['WorkingMemory/PreUpdateAttention/'+n for n in ('README.md','decision_review.md','review_ready.json','model_checks.json')]:
    dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
portable=bundle/'portable';portable.mkdir(exist_ok=True);shutil.copy2(parent,portable/'parent_checkpoint_004400.pt')
cp=torch.load(parent,map_location='cpu');model,opt,lineage=migrate(cp,'preupdate_attention',recipe());torch.save(model.state_dict(),portable/'attention_initialization.pt')
created=datetime.fromisoformat('2026-09-13T21:34:59.090+00:00').timestamp()
manifest=dict(pod_id='uk1sp5850dynxw',created_unix=created,deadline_unix=created+14400,parent_sha256=sha(parent),source_hashes=hashes,files={str(p.relative_to(bundle)).replace('\\','/'):sha(p) for p in bundle.rglob('*') if p.is_file()},lineage=lineage)
write(portable/'manifest.json',manifest)
archive=root/'attention_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in bundle.rglob('*'):
        if p.is_file():tar.add(p,arcname=str(p.relative_to(bundle)))
write(root/'bundle_receipt.json',dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest['files']),initializer_sha256=sha(portable/'attention_initialization.pt')))
receipt=dict(account_email='jonathan@palladio.ai',pod_id='uk1sp5850dynxw',pod_name='vawm-preupdate-attention-32000',created_unix=created,deadline_unix=created+14400,created_utc='2026-09-13T21:34:59.090Z',hard_deadline_utc='2026-09-14T01:34:59.090Z',gpu='NVIDIA GeForce RTX 3090',gpu_count=1,cloud='SECURE',data_center='EUR-IS-2',gpu_usd_per_hour=.5,approximate_total_usd_per_hour=.505,container_gb=30,volume_gb=10,remote_root='/workspace/vawm_attention',remote_results='/workspace/vawm_attention/remote_results',local_run=str(root),ssh_identity='C:/Users/jomor/.ssh/palladio_recurrent_20260912',known_hosts='C:/Users/jomor/.ssh/palladio_recurrent_known_hosts',status='provisioned_setup_pending',target_updates=4000,target_fresh_episodes=32000,cleanup='Local SSH watcher retrieves and verifies indexed artifacts; coordinator connected-API heartbeat stops on completion or deadline and deletes after verified retrieval.')
write(HERE/'cloud_provisioning.json',receipt);print(json.dumps(read(root/'bundle_receipt.json')))
