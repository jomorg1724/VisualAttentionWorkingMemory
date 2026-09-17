"""Local watcher for the accumulator pod: pull results every 10 minutes, then stop and delete the pod when both lanes
finish or the deadline passes. Retrieval before deletion is verified; on a failed final retrieval the pod is left running
and the receipt says so (delete it with WorkingMemory/cloud_shutdown.py --force after inspecting).

Usage (PowerShell, detached): python WorkingMemory/PlainBaseline/cloud/watch.py <cloud_provisioning.json>
"""
import json,shlex,subprocess,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from WorkingMemory.cloud_shutdown import api_key,provider_request,read,sha,ssh_parts,write

def pull(config,tag):
    root=Path(config['local_run']);common,dest,ssh=ssh_parts(config);remote=config['remote_results'];parent,name=remote.rsplit('/',1)
    remote_archive=f'/workspace/{name}_{tag}.tar.gz'
    pack='cd '+shlex.quote(parent)+" && tar --exclude='*.pt' -czf "+shlex.quote(remote_archive)+' '+shlex.quote(name)+' && sha256sum '+shlex.quote(remote_archive)
    digest=subprocess.run([*ssh,pack],capture_output=True,text=True,timeout=900,check=True).stdout.split()[0]
    local=root/f'{name}_{tag}.tar.gz';subprocess.run(['scp',*common,'-P',str(config['ssh_port']),dest+':'+remote_archive,str(local)],capture_output=True,text=True,timeout=3600,check=True)
    if sha(local)!=digest:raise RuntimeError('archive hash mismatch')
    pulled=root/'pulled';pulled.mkdir(exist_ok=True)
    with tarfile.open(local,'r:gz') as t:
        for m in t.getmembers():
            (pulled/m.name).resolve().relative_to(pulled.resolve())
            if m.issym() or m.islnk():raise RuntimeError('refusing archive link')
        t.extractall(pulled)
    local.unlink();return pulled/name

def pull_checkpoints(config,results):
    """Fetch terminal.pt of the cued and delayC stages of every lane with sha256 verification."""
    common,dest,ssh=ssh_parts(config);remote=config['remote_results'];fetched=[]
    listing=subprocess.run([*ssh,'cd '+shlex.quote(remote)+" && find . -path '*/cued/terminal.pt' -o -path '*/delayC/terminal.pt' | xargs -r sha256sum"],capture_output=True,text=True,timeout=600)
    for line in listing.stdout.splitlines():
        digest,rel=line.split();rel=rel.lstrip('./');target=results/rel;target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*common,'-P',str(config['ssh_port']),dest+':'+remote+'/'+rel,str(target)],capture_output=True,text=True,timeout=1800,check=True)
        if sha(target)!=digest:raise RuntimeError('checkpoint hash mismatch '+rel)
        fetched.append(rel)
    return fetched

def lanes_done(results,config):
    states={}
    for lane in config['lanes']:
        r=results/lane/'program_receipt.json'
        states[lane]=json.loads(r.read_text()).get('status') if r.exists() else None
    return states,all(s in ('completed','deadline') for s in states.values())

def shutdown(config,receipt):
    key=api_key();pid=config['pod_id']
    stop=provider_request('POST',f'https://rest.runpod.io/v1/pods/{pid}/stop',key);receipt['stop']=stop['status']
    time.sleep(10);dele=provider_request('DELETE','https://rest.runpod.io/v1/pods/'+pid,key);receipt['delete']=dele['status']
    time.sleep(5);look=provider_request('GET','https://rest.runpod.io/v1/pods/'+pid,key);receipt['lookup_after_delete']=look['status']
    receipt['pod_gone']=(look['status']==404)

def main(config_path):
    config=read(config_path);root=Path(config['local_run']);log=open(root/'watch.log','a');deadline=float(config['deadline_unix'])
    def say(*a):print(time.strftime('%H:%M:%S'),*a,file=log,flush=True)
    say('watcher started for pod',config['pod_id']);receipt=dict(kind='watcher',pod_id=config['pod_id'],pulls=[],started_unix=time.time())
    while True:
        try:
            results=pull(config,'pull');states,done=lanes_done(results,config);receipt['pulls'].append(dict(unix=time.time(),states=states));say('pulled',states)
        except Exception as e:
            say('pull failed',repr(e));done=False;results=root/'pulled'/config['remote_results'].rsplit('/',1)[1]
        if done or time.time()>deadline+config.get('retrieval_grace_seconds',900):
            say('final retrieval; done=',done)
            try:
                results=pull(config,'final');receipt['checkpoints']=pull_checkpoints(config,results);receipt['final_states']=lanes_done(results,config)[0];receipt['final_retrieval']='verified'
            except Exception as e:
                receipt['final_retrieval']='FAILED '+repr(e);say('FINAL RETRIEVAL FAILED; pod left running',repr(e));write(root/'watch_receipt.json',receipt);return
            shutdown(config,receipt);receipt['completed_unix']=time.time();write(root/'watch_receipt.json',receipt);config['status']='stopped_and_deleted' if receipt.get('pod_gone') else 'delete_unverified';write(config_path,config)
            say('shutdown',receipt.get('stop'),receipt.get('delete'),receipt.get('lookup_after_delete'));return
        time.sleep(600)

if __name__=='__main__':main(sys.argv[1])
