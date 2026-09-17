"""Unattended SSH completion retrieval. Connected API heartbeat owns pod billing."""
import subprocess,json,time,hashlib,tarfile,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,data):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    cfg=read(HERE/'cloud_provisioning.json');root=Path(cfg['local_run']);deadline=cfg['deadline_unix']
    common=['-i',cfg['ssh_identity'],'-o','UserKnownHostsFile='+cfg['known_hosts'],'-o','StrictHostKeyChecking=accept-new','-o','BatchMode=yes','-o','ConnectTimeout=15']
    ssh=['ssh',*common,'-p',str(cfg['ssh_port']),'root@'+cfg['ssh_host']]
    remote=cfg['remote_root'];remote_results=cfg['remote_results'];status=root/'watcher_status.json'
    while time.time()<deadline+600:
        try:
            check=subprocess.run([*ssh,'test -f '+remote_results+'/exit.json && cat '+remote_results+'/exit.json'],capture_output=True,text=True,timeout=30)
            if check.returncode==0:
                exitinfo=json.loads(check.stdout);write(status,dict(status='retrieving',exit=exitinfo,updated_unix=time.time()))
                pack=subprocess.run([*ssh,'cd '+remote+' && tar -czf attention_results.tar.gz remote_results && sha256sum attention_results.tar.gz'],capture_output=True,text=True,timeout=180,check=True)
                digest=pack.stdout.split()[0];archive=root/'attention_results.tar.gz'
                subprocess.run(['scp',*common,'-P',str(cfg['ssh_port']),'root@'+cfg['ssh_host']+':'+remote+'/attention_results.tar.gz',str(archive)],capture_output=True,text=True,timeout=240,check=True)
                assert sha(archive)==digest,'Archive checksum mismatch'
                dest=root/'retrieved';dest.mkdir(exist_ok=True)
                with tarfile.open(archive,'r:gz') as tar:
                    for member in tar.getmembers():
                        target=(dest/member.name).resolve()
                        if not target.is_relative_to(dest.resolve()) or member.issym() or member.islnk():raise RuntimeError('Unsafe archive member')
                    tar.extractall(dest)
                fetched=dest/'remote_results';index=read(fetched/'artifact_index.json')
                for name,expected in index.items():
                    target=(fetched/name).resolve()
                    assert target.is_relative_to(fetched.resolve()) and sha(target)==expected,name
                receipt=dict(status='verified',pod_id=cfg['pod_id'],archive=str(archive),archive_sha256=digest,verified_files=len(index),results=str(fetched),exit=exitinfo,verified_unix=time.time(),cleanup='Ready for coordinator connected-API stop and delete')
                write(root/'retrieval_receipt.json',receipt);write(HERE/'retrieval_receipt.json',receipt);write(status,receipt)
                print(json.dumps(receipt),flush=True)
                if exitinfo.get('status')=='completed':
                    analysis=subprocess.run([sys.executable,'-B',str(HERE/'analyze.py')],capture_output=True,text=True,timeout=120)
                    (root/'analysis.log').write_text(analysis.stdout+'\n'+analysis.stderr)
                    write(root/'analysis_status.json',dict(returncode=analysis.returncode,finished_unix=time.time()))
                return
            write(status,dict(status='waiting',checked_unix=time.time(),returncode=check.returncode,stderr=check.stderr[-300:]))
        except Exception as e:
            write(status,dict(status='retrying',error=repr(e),updated_unix=time.time()));print(repr(e),flush=True)
        time.sleep(45)
    write(status,dict(status='deadline_retrieval_pending',updated_unix=time.time(),action='Coordinator must stop pod and preserve volume until results recovered'))
if __name__=='__main__':run()
