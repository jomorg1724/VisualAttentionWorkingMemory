"""Download and hash-verify the stopped run through a temporary recovery pod."""
import hashlib,json,sys,time,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.cwd()/'SecondPass/DeadlyCorridor/CloudRuntime/sshdeps'))
import paramiko

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    dest=BASE/'recovered';dest.mkdir(exist_ok=True)
    c=paramiko.SSHClient();c.load_host_keys(str(BASE/'known_hosts'))
    endpoint=json.loads((BASE/'deployment.json').read_text())['ssh']
    c.connect(endpoint['host'],port=endpoint['port'],username='root',key_filename=str(Path.home()/'.ssh/id_ed25519'),timeout=20)
    c.save_host_keys(str(dest/'recovery_known_hosts'))
    script="""import pathlib,json,hashlib,tarfile
root=pathlib.Path('/workspace');run=root/'vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01'
print('EXIT', (run/'training/exit.json').read_text(),flush=True)
print('SUPERVISOR', (run/'supervisor_exit.json').read_text(),flush=True)
assert json.loads((run/'supervisor_exit.json').read_text())['exit_code']==0
worker=json.loads((run/'worker.json').read_text());assert not pathlib.Path('/proc/'+str(worker['pid'])).exists()
source=json.loads((root/'source_manifest.json').read_text())
files=list(run.rglob('*'))+[root/'source_manifest.json']+[root/'vawm'/x for x in source]
files=sorted(set(x for x in files if x.is_file() and '__pycache__' not in x.parts))
manifest={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
(root/'continuation_recovery_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'continuation_recovery.tar.gz','w:gz',compresslevel=1) as t:
 for f in files+[root/'continuation_recovery_manifest.json']:t.add(f,arcname=str(f.relative_to(root)),recursive=False)
p=root/'continuation_recovery.tar.gz';h=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'continuation_recovery_archive.json').write_text(json.dumps(dict(sha256=h,bytes=p.stat().st_size)))
print('ARCHIVE',h,p.stat().st_size,flush=True)
"""
    i,o,e=c.exec_command('python3 -');i.write(script);i.channel.shutdown_write()
    for line in o:print(line.strip(),flush=True)
    err=e.read().decode();code=o.channel.recv_exit_status()
    if err:print(err,flush=True)
    assert code==0
    s=c.open_sftp();s.get('/workspace/continuation_recovery_archive.json',str(dest/'archive_remote.json'))
    meta=json.loads((dest/'archive_remote.json').read_text());last=[time.time()]
    def progress(n,total):
        if time.time()-last[0]>15 or n==total:print(f'Download {n/1e6:.1f}/{total/1e6:.1f} MB',flush=True);last[0]=time.time()
    archive=dest/'continuation_recovery.tar.gz';s.get('/workspace/continuation_recovery.tar.gz',str(archive),callback=progress)
    s.close();c.close();assert digest(archive)==meta['sha256']
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            target=(dest/m.name).resolve();assert target.is_relative_to(dest.resolve()) and m.isfile(),m.name
        t.extractall(dest)
    manifest=json.loads((dest/'continuation_recovery_manifest.json').read_text())
    for name,h in manifest.items():assert digest(dest/name)==h,name
    receipt=dict(archive=meta,files_verified=len(manifest),all_hashes_match=True,completed=time.time(),original_pod='yq0g7zledvfwuk',volume='7gz6ogdywq')
    (dest/'download_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
