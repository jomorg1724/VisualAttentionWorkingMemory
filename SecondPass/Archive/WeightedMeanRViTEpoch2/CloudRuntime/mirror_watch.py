"""Detached laptop mirror and off-pod status poll; never provisions/restarts."""
import json
import hashlib
import os
from pathlib import Path
import time
from deploy import ROOT, api, save, diagnostic
import monitor
import retrieve_once


def watch():
    if (ROOT/'cleanup_verified.json').exists() or (ROOT/'mirror_result.json').exists(): return
    claim=ROOT/'mirror_claim.json'
    if claim.exists():
        previous=json.loads(claim.read_text())
        try: os.kill(previous['pid'],0);return
        except ProcessLookupError: pass
    save('mirror_claim.json',dict(pid=os.getpid(),observed=time.time(),no_restart=True))
    pod=json.loads((ROOT/'pod.json').read_text());deadline=pod['hard_deadline']
    failures=0
    while time.time()<deadline+120:
        try:
            lifecycle=api('/v2/pods/'+pod['id'])['status']
            if lifecycle!='RUNNING':
                if lifecycle in ('STOPPED','EXITED') and (ROOT/'retrieval_verified.json').exists():
                    receipt=json.loads((ROOT/'retrieval_verified.json').read_text())
                    cleanup=cleanup_verified_ephemeral(receipt)
                    save('mirror_result.json',dict(status='complete_verified_deleted',observed=time.time(),receipt=receipt,cleanup=cleanup))
                    return
                monitor.status()
                save('mirror_result.json',dict(status='stopped',lifecycle=lifecycle,observed=time.time(),no_restart=True))
                return
            # Provider-backed private status survives local disconnection.
            status=monitor.status()
            save('mirror_status.json',dict(observed=time.time(),remote_status_available='remote' in status))
            receipt=retrieve_once.retrieve()
            failures=0
            if receipt.get('complete'):
                cleanup=cleanup_verified_ephemeral(receipt)
                save('mirror_result.json',dict(status='complete_verified_deleted',observed=time.time(),receipt=receipt,cleanup=cleanup))
                # The guard stops; polling is bounded independently of its network.
                return
        except Exception as exc:
            failures+=1
            save('mirror_last_error.json',dict(**diagnostic(exc),observed=time.time(),consecutive_failures=failures))
        time.sleep(15)
    save('mirror_result.json',dict(status='deadline_elapsed',observed=time.time(),no_restart=True))



def cleanup_verified_ephemeral(receipt):
    """Delete only this attempt after complete hashes and stopped readback."""
    from deploy import REMOTE, list_pods
    pod=json.loads((ROOT/'pod.json').read_text())
    if receipt.get('complete') is not True or receipt.get('pod')!=pod['id'] or pod.get('remote_root')!=REMOTE:
        raise ValueError('Cleanup requires complete retrieval for this exact attempt')
    dest=ROOT/'artifacts';completion=dest/'cloud_completion.json'
    if hashlib.sha256(completion.read_bytes()).hexdigest()!=receipt.get('completion_sha256'):
        raise ValueError('Cleanup completion identity mismatch')
    record=json.loads(completion.read_text())
    retrieve_once.verify_local(dest,record['artifact_manifest'])
    checks=receipt.get('cpu_checkpoint_reload',{})
    if set(checks)!={'best','latest'} or not all(checks[role].get('verified') is True for role in checks):
        raise ValueError('Both final checkpoints require CPU reload evidence')
    for role,row in checks.items():
        path=Path(row['path'])
        if not path.resolve().is_relative_to(dest.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Final CPU reload identity mismatch')
    # Guard handles completion stop; never delete a running pod or change its env.
    deadline=time.time()+90
    while time.time()<deadline:
        current=api('/v2/pods/'+pod['id'])
        if current.get('id')!=pod['id']: raise ValueError('Cleanup provider identity mismatch')
        if current['status'] in ('EXITED','STOPPED'):
            save('cleanup_stop_verified.json',dict(pod=pod['id'],status=current['status'],observed=time.time()))
            api('/v2/pods/'+pod['id'],'DELETE')
            for _ in range(6):
                if not any(item['id']==pod['id'] for item in list_pods()):
                    result=dict(pod=pod['id'],deleted=True,artifacts_verified=True,observed=time.time())
                    save('cleanup_verified.json',result);return result
                time.sleep(5)
            raise RuntimeError('Ephemeral deletion readback unresolved')
        time.sleep(5)
    raise RuntimeError('Completion stop readback unresolved; no deletion performed')

if __name__=='__main__': watch()
