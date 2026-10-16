"""Upload source only, verify identity, arm the finite guard, and install runtime."""
import hashlib,json,sys,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/sshdeps'))
import paramiko

def main():
    dep=json.loads((BASE/'deployment.json').read_text());d=dep['ssh']
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(d['host'],port=d['port'],username='root',key_filename=str(Path.home()/'.ssh/id_ed25519'),timeout=25)
    client.save_host_keys(str(BASE/'known_hosts'))
    previous=json.loads((ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/run01/source_manifest.json').read_text())
    files=[ROOT/f for f in previous if not f.endswith('.pt')]
    files+=list((ROOT/'SecondPass/DeadlyCorridorKills').glob('*.py'))+list(BASE.glob('*.py'))+[BASE/'deployment.json']
    manifest={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (BASE/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(BASE/'source_bundle.tar.gz','w:gz') as archive:
        for f in files:archive.add(f,arcname='vawm/'+f.relative_to(ROOT).as_posix())
        archive.add(BASE/'source_manifest.json',arcname='source_manifest.json')
    sftp=client.open_sftp();sftp.put(str(BASE/'source_bundle.tar.gz'),'/workspace/source_bundle.tar.gz')
    script='''import tarfile,pathlib,json,os,time,requests,subprocess,hashlib
with tarfile.open('/workspace/source_bundle.tar.gz') as t:t.extractall('/workspace')
root=pathlib.Path('/workspace/vawm');run=root/'SecondPass/DeadlyCorridorKills/CloudRuntime/run01'
manifest=json.loads(pathlib.Path('/workspace/source_manifest.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
(run/'upload_verified.json').write_text(json.dumps(dict(all_hashes_match=True,files_verified=len(manifest))))
dep=json.loads((run/'deployment.json').read_text());env=dict(os.environ)
for item in pathlib.Path('/proc/1/environ').read_bytes().split(b'\\0'):
 if b'=' in item:
  k,v=item.split(b'=',1);env.setdefault(k.decode(),v.decode())
key=env.get('RUNPOD_API_KEY');assert key,'Provider-issued pod key missing'
r=requests.post('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key},json={'query':'query($input: PodFilter) { pod(input: $input) { id desiredStatus } }','variables':{'input':{'podId':dep['pod_id']}}},timeout=30)
assert r.status_code==200 and (r.json().get('data') or {}).get('pod',{}).get('id')==dep['pod_id']
p=pathlib.Path('/root/.doom-pod-key');p.write_text(key);p.chmod(0o600)
(run/'preflight_verified.json').write_text(json.dumps(dict(pod_id=dep['pod_id'],pod_scoped_auth=True,time=time.time())))
with (run/'guard.log').open('w') as f:
 p=subprocess.Popen(['python3','-u',str(run/'cloud_control.py'),'guard'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('Guard started',p.pid,flush=True)
setup='python3 -m venv --system-site-packages /opt/doom-venv && /opt/doom-venv/bin/pip install vizdoom==1.2.3 numpy==1.26.4 scipy==1.14.1 pillow==10.4.0 gymnasium==0.29.1 pytest==8.4.2'
with (run/'setup.log').open('w') as f:
 p=subprocess.Popen(['bash','-lc',setup],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('Setup started',p.pid,flush=True)
'''
    i,o,e=client.exec_command('python3 -');i.write(script);i.channel.shutdown_write()
    print(o.read().decode());print(e.read().decode());assert o.channel.recv_exit_status()==0
    remote='/workspace/vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01/'
    for name in ['preflight_verified.json','upload_verified.json']:sftp.get(remote+name,str(BASE/name))
    sftp.close();client.close()

if __name__=='__main__':main()
