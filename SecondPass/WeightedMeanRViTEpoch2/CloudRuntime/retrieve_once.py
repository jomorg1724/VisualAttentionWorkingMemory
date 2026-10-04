"""Verified automatic retrieval while GPU is alive; never restarts a stopped pod."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
from deploy import ROOT, REMOTE, api, remote_python, ssh_command, save, verify_checkpoint_file


def validate_items(items):
    names=[x['relative_path'] for x in items]
    if not names or len(names)!=len(set(names)): raise ValueError('Empty/duplicate manifest')
    for item in items:
        p=PurePosixPath(item['relative_path'])
        if p.is_absolute() or '..' in p.parts or not p.parts or any(x in str(p) for x in ('\n','\r','\\')): raise ValueError('Unsafe artifact path')
        if not isinstance(item['bytes'],int) or item['bytes']<=0 or not re.fullmatch('[0-9a-f]{64}',item['sha256']): raise ValueError('Bad artifact identity')


def verify_local(dest, items):
    validate_items(items)
    for item in items:
        local=dest/item['relative_path']
        if local.is_symlink() or not local.resolve().is_relative_to(dest.resolve()) or local.stat().st_size!=item['bytes'] or hashlib.sha256(local.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('Artifact digest mismatch')


def retrieve():
    ident=json.loads((ROOT/'pod.json').read_text())['id']
    if api('/v2/pods/'+ident)['status']!='RUNNING': raise RuntimeError('Stopped disk; no GPU restart performed')
    pod=api('/pods/'+ident)
    save('ssh_endpoint.json',dict(ip=pod['publicIp'],port=pod['portMappings']['22'],pod=ident))
    data=json.loads(remote_python('''import json,hashlib,sys
from pathlib import Path
root=Path(ROOT_VALUE);r=root/'run';out={}
sys.path.insert(0,str(root));from pod_guard import verified_finalization
for n in ('cloud_completion.json','latest_checkpoint.json'):
 p=r/n
 if p.exists():out[n]=json.loads(p.read_text())
out['finalized']=verified_finalization(r)
if out['finalized']:out['completion_sha256']=hashlib.sha256((r/'cloud_completion.json').read_bytes()).hexdigest()
# Mutable status metadata is mirrored as a best-effort snapshot alongside the
# immutable checkpoint. Only immutable checkpoint bytes establish progress.
out['metadata']=[p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log','.md')]
print(json.dumps(out))
'''.replace('ROOT_VALUE',repr(REMOTE)),timeout=30))
    complete=data.get('cloud_completion.json') if data['finalized'] else None
    cp=data.get('latest_checkpoint.json')
    if complete:
        items=complete['artifact_manifest']
        paths=[x['relative_path'] for x in items]+['cloud_completion.json']
    else:
        if not cp or cp['step']<=0: return dict(complete=False,awaiting_checkpoint=True)
        path=PurePosixPath(cp['path']);remote_run=PurePosixPath(REMOTE)/'run'
        if not path.is_relative_to(remote_run) or '..' in path.parts: raise ValueError('Wrong checkpoint source path')
        items=[dict(relative_path=path.relative_to(remote_run).as_posix(),bytes=cp['bytes'],sha256=cp['sha256'])]
        # Metadata file paths receive the same containment checks.
        metadata=data.get('metadata',[])
        validate_items([dict(relative_path=p,bytes=1,sha256='0'*64) for p in metadata])
        paths=list(dict.fromkeys([items[0]['relative_path'],*metadata]))
    validate_items(items)
    dest=ROOT/'artifacts';dest.mkdir(exist_ok=True)
    listing=ROOT/'retrieval_paths.txt';listing.write_text('\n'.join(paths)+'\n')
    ssh=ssh_command()
    result=subprocess.run(['rsync','-rt','--partial','--timeout=45','--files-from='+str(listing),'-e',shlex.join(ssh[:-1]),ssh[-1]+':'+REMOTE+'/run/',str(dest)+'/'],capture_output=True,text=True,timeout=100)
    if result.returncode: raise RuntimeError('Transfer interrupted; disk retained; no retrieval claim')
    verify_local(dest,items)
    if complete and json.loads((dest/'cloud_completion.json').read_text())!=complete: raise ValueError('Completion manifest changed during retrieval')
    receipt=dict(verified_at=time.time(),complete=bool(complete),checkpoint=cp,files_verified=len(items),directory=str(dest),pod=ident)
    if complete:
        receipt['completion_sha256']=data['completion_sha256']
        if hashlib.sha256((dest/'cloud_completion.json').read_bytes()).hexdigest()!=receipt['completion_sha256']: raise ValueError('Completion byte digest mismatch')
    if complete:
        report=json.loads((dest/'report.json').read_text());rows={row['relative_path']:row for row in items}
        receipt['cpu_checkpoint_reload']={}
        for role,report_key in (('best','selected_step'),('latest','terminal_step')):
            name=role+'.pt'
            receipt['cpu_checkpoint_reload'][role]=verify_checkpoint_file(dest/name,dict(rows[name],step=report[report_key]))
    save('retrieval_verified.json' if complete else 'early_checkpoint_verified.json',receipt)
    if complete:
        # Guard can stop immediately only after complete local digest verification.
        remote_python('import sys;from pathlib import Path;sys.path.insert(0,'+repr(REMOTE)+');from pod_guard import atomic_json;atomic_json(Path('+repr(REMOTE+'/run/retrieval_verified.json')+'),'+repr(receipt)+')',timeout=30)
    return receipt


def main(): print(json.dumps(retrieve(),indent=2))
if __name__=='__main__': main()
