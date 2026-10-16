"""Stage verified preserved state and start dependency setup, without training."""
import hashlib,json,sys,time,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'sshdeps'))
import paramiko

def main():
    dep=json.loads((BASE/'deployment.json').read_text());d=dep['ssh']
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(d['host'],port=d['port'],username=d['user'],key_filename=str(Path.home()/'.ssh/id_ed25519'),timeout=25)
    c.save_host_keys(str(BASE/'known_hosts'))
    def execute(script):
        i,o,e=c.exec_command('python3 -');i.write(script);i.channel.shutdown_write()
        for line in o:print(line.strip(),flush=True)
        err=e.read().decode();code=o.channel.recv_exit_status()
        if err:print(err,flush=True)
        assert code==0
    # Fresh metadata must be in the control bundle.
    root=Path.cwd();parent=root/'SecondPass/StabilizedAttentionSuite/LocalRuntime/continuation02/best.pt'
    with tarfile.open(BASE/'control_bundle.tar.gz','w:gz') as t:
        for f in list(BASE.glob('*.py'))+[BASE/'deployment.json',parent]:t.add(f,arcname='vawm/'+f.relative_to(root).as_posix())
    s=c.open_sftp();s.put(str(BASE/'control_bundle.tar.gz'),'/workspace/control_bundle.tar.gz')
    execute("""import tarfile,pathlib,json,os,time,requests,subprocess
with tarfile.open('/workspace/control_bundle.tar.gz') as t:t.extractall('/workspace')
run=pathlib.Path('/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/continuation02');dep=json.loads((run/'deployment.json').read_text())
env=dict(os.environ)
for item in pathlib.Path('/proc/1/environ').read_bytes().split(b'\\0'):
 if b'=' in item:
  k,v=item.split(b'=',1);env.setdefault(k.decode(),v.decode())
key=env.get('RUNPOD_API_KEY');assert key,'Provider-issued pod key missing'
r=requests.post('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key},json={'query':'query($input: PodFilter) { pod(input: $input) { id desiredStatus } }','variables':{'input':{'podId':dep['pod_id']}}},timeout=30)
assert r.status_code==200 and (r.json().get('data') or {}).get('pod',{}).get('id')==dep['pod_id'],'Pod identity check failed'
p=pathlib.Path('/root/.doom-pod-key');p.write_text(key);p.chmod(0o600)
(run/'preflight_verified.json').write_text(json.dumps(dict(pod_id=dep['pod_id'],pod_scoped_auth=True,identity_verified_at=time.time())))
with (run/'guard.log').open('w') as f:
 p=subprocess.Popen(['python3','-u',str(run/'cloud_control.py'),'guard'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('Guard started',p.pid,flush=True)
setup='python3 -m venv --system-site-packages /workspace/doom-venv && /workspace/doom-venv/bin/pip install vizdoom==1.2.3 numpy==1.26.4 scipy==1.14.1 pillow==10.4.0 gymnasium==0.29.1 pytest==8.4.2'
with (run/'setup.log').open('w') as f:
 p=subprocess.Popen(['bash','-lc',setup],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('Dependency setup started',p.pid,flush=True)
""")
    archive=BASE.parent/'continuation01/recovered/continuation_recovery.tar.gz';h=hashlib.sha256(archive.read_bytes()).hexdigest();last=[time.time()]
    def progress(n,total):
        if time.time()-last[0]>15 or n==total:print(f'Upload {n/1e6:.1f}/{total/1e6:.1f} MB',flush=True);last[0]=time.time()
    s.put(str(archive),'/workspace/continuation_recovery.tar.gz',callback=progress)
    execute("""import pathlib,hashlib,json,tarfile
p=pathlib.Path('/workspace/continuation_recovery.tar.gz');assert hashlib.sha256(p.read_bytes()).hexdigest()=="""+repr(h)+"""
with tarfile.open(p) as t:t.extractall('/workspace')
root=pathlib.Path('/workspace');manifest=json.loads((root/'continuation_recovery_manifest.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
run=root/'vawm/SecondPass/DeadlyCorridor/CloudRuntime/continuation02'
receipt=dict(archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files_verified=len(manifest),all_hashes_match=True)
(run/'upload_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
""")
    remote='/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/continuation02/'
    for name in ['preflight_verified.json','guard.json','upload_verified.json']:s.get(remote+name,str(BASE/name))
    s.close();c.close()

if __name__=='__main__':main()
