"""Snapshot only the completed old arm while the new GPU arm keeps running."""
import json,subprocess,shlex,hashlib,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=json.loads((HERE/'cloud_provisioning.json').read_text());run=Path(cfg['local_run'])/'old_arm_report';run.mkdir(exist_ok=True)
    common=['-i',cfg['ssh_identity'],'-o','UserKnownHostsFile='+cfg['known_hosts'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh','-n',*common,'-p',str(cfg['ssh_port']),'root@'+cfg['ssh_host']]
    code='''import json,pathlib,tarfile,hashlib,io
r=pathlib.Path('/workspace/vawm_unbiased/remote_results');a=json.loads((r/'aggregate.json').read_text());assert a['runs']['old_unbiased']['status']=='completed'
live=a['runs'].get('spatial_unbiased',{});a['live_spatial_snapshot']=dict(status=live.get('status','running'),training=live.get('training'));a['runs']={'old_unbiased':a['runs']['old_unbiased']};a['status']='old_arm_completed_new_arm_continues'
files=[p for p in (r/'old_unbiased').rglob('*') if p.is_file() and p.suffix!='.pt']+[p for p in r.glob('old_unbiased_*') if p.is_file()]
payload={str(p.relative_to(r)):p.read_bytes() for p in files};payload['aggregate.json']=json.dumps(a,indent=2).encode();payload['live_budget_snapshot.json']=(r/'budget.json').read_bytes();index={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()};payload['partial_artifact_index.json']=json.dumps(index,indent=2).encode()
archive=r.parent/'old_arm_snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name,data in payload.items():
  info=tarfile.TarInfo('remote_results/'+name);info.size=len(data);t.addfile(info,io.BytesIO(data))
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(index))))'''
    info=json.loads(subprocess.run([*ssh,'python -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60,check=True).stdout);archive=run/'old_arm_snapshot.tar.gz'
    subprocess.run(['scp',*common,'-P',str(cfg['ssh_port']),'root@'+cfg['ssh_host']+':'+info['archive'],str(archive)],capture_output=True,text=True,timeout=180,check=True);assert sha(archive)==info['sha256']
    dest=run/'retrieved';dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t.getmembers():assert (dest/m.name).resolve().is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
        t.extractall(dest)
    result=dest/'remote_results';index=json.loads((result/'partial_artifact_index.json').read_text())
    for name,digest in index.items():assert sha(result/name)==digest,name
    receipt=dict(status='verified_completed_old_arm_only',pod_id=cfg['pod_id'],archive=str(archive),archive_sha256=info['sha256'],results=str(result),verified_files=len(index),observed_unix=time.time(),new_arm='Continues unchanged; this partial receipt does not authorize stopping the pod',checkpoints='Already handled by separate incremental watcher; excluded from this report snapshot')
    (run/'retrieval_receipt.json').write_text(json.dumps(receipt,indent=2));(HERE/'old_arm_retrieval_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
