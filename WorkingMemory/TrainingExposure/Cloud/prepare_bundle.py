"""Copy immutable shared sources and trained parent; no model/GPU imports."""
import json,shutil,tarfile,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-manifest',required=True);args=parser.parse_args();provision=read(HERE/'cloud_provisioning.json');run=Path(provision['local_run']);bundle=run/'portable_bundle';bundle.mkdir(parents=True,exist_ok=True)
    source=read(args.source_manifest);hashes=source['source_hashes'];hashes.update(read(HERE.parent/'review_ready.json')['source_hashes']);names=list(hashes)
    entry='WorkingMemory/TrainingExposure/Cloud/remote_sweep.py';hashes[entry]=sha(ROOT/entry);names.append(entry)
    for name in names:
        assert sha(ROOT/name)==hashes[name],name
        dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    for p in (ROOT/'WorkingMemory/TrainingExposure').glob('*.json'):
        if p.name in ('review_ready.json','checks.json','model_checks.json'):dest=bundle/'WorkingMemory/TrainingExposure'/p.name;shutil.copy2(p,dest)
    for name in ('README.md','decision_review.md'):
        p=ROOT/'WorkingMemory/TrainingExposure'/name
        if p.exists():shutil.copy2(p,bundle/'WorkingMemory/TrainingExposure'/name)
    parent=ROOT/'WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/checkpoint_008400.pt';portable=bundle/'portable';portable.mkdir(exist_ok=True);shutil.copy2(parent,portable/'parent_checkpoint_008400.pt')
    manifest=dict(pod_id=provision['pod_id'],created_unix=provision['created_unix'],deadline_unix=provision['deadline_unix'],parent_sha256=sha(parent),source_hashes=hashes,files={str(p.relative_to(bundle)).replace('\\','/'):sha(p) for p in bundle.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (portable/'manifest.json').write_text(json.dumps(manifest,indent=2));archive=run/'exposure_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in bundle.rglob('*'):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(bundle)))
    receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest['files']),parent_sha256=sha(parent));(run/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
