"""Retrieve the bounded cloud arm; connected API coordinator closes billing."""
import subprocess,json,time,hashlib,tarfile,shutil,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,data):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=read(HERE/'cloud_provisioning.json');root=Path(cfg['local_run']);deadline=cfg['deadline_unix'];common=['-i',cfg['ssh_identity'],'-o','UserKnownHostsFile='+cfg['known_hosts'],'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh','-n',*common,'-p',str(cfg['ssh_port']),'root@'+cfg['ssh_host']];remote=cfg['remote_root'];results=cfg['remote_results'];status=root/'watcher_status.json';incremental=root/'incremental'/'remote_results';incremental.mkdir(parents=True,exist_ok=True);transfers=[]
    def sync_checkpoints():
        code="import pathlib,json; r=pathlib.Path("+repr(results)+"); print(json.dumps([dict(path=str(p.parent.relative_to(r)/v['file']),sha256=v['sha256']) for p in r.rglob('checkpoint_index.jsonl') for v in [json.loads(l) for l in p.read_text().splitlines()]]))"
        listing=subprocess.run([*ssh,'python -c '+__import__('shlex').quote(code)],capture_output=True,text=True,timeout=30,check=True)
        for row in json.loads(listing.stdout):
            name=row['path']
            if not re.fullmatch(r'(old_unbiased|spatial_unbiased)/(profile|training)/checkpoint_[0-9]{6}\.pt',name):raise RuntimeError('Unexpected checkpoint path '+name)
            local=incremental/name;local.parent.mkdir(parents=True,exist_ok=True)
            if local.exists() and sha(local)==row['sha256']:continue
            tick=time.time();temp=local.with_suffix('.download')
            subprocess.run(['scp',*common,'-P',str(cfg['ssh_port']),'root@'+cfg['ssh_host']+':'+results+'/'+name,str(temp)],capture_output=True,text=True,timeout=600,check=True)
            assert sha(temp)==row['sha256'];temp.replace(local);transfers.append(dict(file=name,bytes=local.stat().st_size,seconds=time.time()-tick,sha256=row['sha256']))
            write(root/'incremental_transfers.json',transfers)
    while time.time()<deadline+600:
        try:
            sync_checkpoints()
            check=subprocess.run([*ssh,'test -f '+results+'/exit.json && test -f '+results+'/artifact_index.json && cat '+results+'/exit.json'],capture_output=True,text=True,timeout=30)
            if check.returncode==0:
                exitinfo=json.loads(check.stdout);write(status,dict(status='retrieving',exit=exitinfo,updated_unix=time.time()))
                pack=subprocess.run([*ssh,'cd '+remote+' && tar --exclude="*.pt" -czf unbiased_results.tar.gz remote_results && sha256sum unbiased_results.tar.gz'],capture_output=True,text=True,timeout=180,check=True);digest=pack.stdout.split()[0];archive=root/'unbiased_results.tar.gz'
                subprocess.run(['scp',*common,'-P',str(cfg['ssh_port']),'root@'+cfg['ssh_host']+':'+remote+'/unbiased_results.tar.gz',str(archive)],capture_output=True,text=True,timeout=600,check=True);assert sha(archive)==digest
                dest=root/'retrieved';dest.mkdir(exist_ok=True)
                with tarfile.open(archive,'r:gz') as tar:
                    for member in tar.getmembers():
                        target=(dest/member.name).resolve()
                        if not target.is_relative_to(dest.resolve()) or member.issym() or member.islnk():raise RuntimeError('Unsafe archive member')
                    tar.extractall(dest)
                fetched=dest/'remote_results';shutil.copytree(incremental,fetched,dirs_exist_ok=True);index=read(fetched/'artifact_index.json')
                for name,expected in index.items():
                    target=(fetched/name).resolve();assert target.is_relative_to(fetched.resolve()) and sha(target)==expected,name
                receipt=dict(status='verified',pod_id=cfg['pod_id'],archive=str(archive),archive_sha256=digest,archive_excludes_checkpoints=True,checkpoints_restored_from_verified_incremental_directory=str(incremental),verified_files=len(index),results=str(fetched),exit=exitinfo,verified_unix=time.time(),cleanup='Ready for connected API stop and delete; process exit alone does not stop billing')
                write(root/'retrieval_receipt.json',receipt);write(HERE/'retrieval_receipt.json',receipt);write(status,receipt);print(json.dumps(receipt),flush=True)
                import sys
                try:subprocess.run([sys.executable,'-B',str(HERE/'report.py')],timeout=180,check=False)
                except Exception as e:write(root/'report_error.json',dict(error=repr(e)))
                return
            write(status,dict(status='waiting',checked_unix=time.time(),returncode=check.returncode,stderr=check.stderr[-300:]))
        except Exception as e:write(status,dict(status='retrying',error=repr(e),updated_unix=time.time()));print(repr(e),flush=True)
        time.sleep(45)
    write(status,dict(status='deadline_retrieval_pending',updated_unix=time.time(),action='Coordinator must stop pod; preserve disk until retrieval'))
if __name__=='__main__':main()
