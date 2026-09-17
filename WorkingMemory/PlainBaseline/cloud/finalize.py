"""One-shot finalisation of the accumulator pod: stop remote work, verified pull of results and key checkpoints,
then stop, delete and confirm the pod is gone. Refuses to delete if the final retrieval fails.

Usage (PowerShell): python WorkingMemory/PlainBaseline/cloud/finalize.py <cloud_provisioning.json>
"""
import json,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from WorkingMemory.cloud_shutdown import read,ssh_parts,write
from WorkingMemory.PlainBaseline.cloud.watch import pull,pull_checkpoints,lanes_done,shutdown

def main(config_path):
    config=read(config_path);root=Path(config['local_run']);common,dest,ssh=ssh_parts(config)
    receipt=dict(kind='finalize',pod_id=config['pod_id'],started_unix=time.time(),steps=[])
    stop_remote="pkill -TERM -f '[P]lainBaseline.program|[P]lainBaseline.baseline' || true; sleep 5; pkill -KILL -f '[P]lainBaseline.program|[P]lainBaseline.baseline' || true; ps -eo pid,etime,cmd | grep -E '[P]lainBaseline' || true; nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    stopped=subprocess.run([*ssh,stop_remote],capture_output=True,text=True,timeout=120);receipt['steps'].append(dict(step='remote_processes_stopped',remaining=stopped.stdout.strip()[-800:]))
    print('remote stopped; remaining:',stopped.stdout.strip()[-300:] or '(none)',flush=True)
    results=pull(config,'final');receipt['steps'].append(dict(step='results_pulled',path=str(results)))
    receipt['checkpoints']=pull_checkpoints(config,results);receipt['steps'].append(dict(step='checkpoints_pulled',count=len(receipt['checkpoints'])))
    states,done=lanes_done(results,config);receipt['final_states']=states;receipt['all_lanes_done']=done;receipt['final_retrieval']='verified'
    print('pulled; lane states',states,'checkpoints',len(receipt['checkpoints']),flush=True)
    shutdown(config,receipt);receipt['completed_unix']=time.time();write(root/'finalize_receipt.json',receipt)
    config['status']='stopped_and_deleted' if receipt.get('pod_gone') else 'delete_unverified';config['finalized_unix']=time.time();write(config_path,config)
    print(json.dumps({k:receipt[k] for k in ('stop','delete','lookup_after_delete','pod_gone','final_states')},indent=1))

if __name__=='__main__':main(sys.argv[1])
