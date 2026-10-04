"""One explicit same-deadline handoff: change launch QoS, never training policy.

Reuses the frozen production loop with a local function scope substituting only
its initializer. The source run remains immutable and is not auto-restarted.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time
import types
from SecondPass.JointTraining.core import atomic_json
from SecondPass.JointTraining.worker import cpu_setup, verify_sources
from . import worker
from .protocol import digest, validate_budget, require_approval, assert_no_worker, launchd_spec, supervise
from .state import load_verified


def validate_resume(source, config, report, now=None):
    now=time.time() if now is None else now
    if source['schema']!=2 or source['state']['config']!=config:
        raise ValueError('Resume must preserve exact training config and deadline')
    state=source['state']
    if not 0<state['step']<config['max_steps'] or now>=config['deadline']:
        raise ValueError('Expired or completed source')
    if report['stop_reason']!='signal' or report['failure'] is not None:
        raise ValueError('Requires clean update-boundary signal stop')
    if report['terminal_step']!=state['step'] or report['terminal_checkpoint']['step']!=state['step']:
        raise ValueError('Terminal progress mismatch')
    if source['scheduler']['updates']!=3393+state['step'] or state['episodes']!=32*state['step']:
        raise ValueError('Sampler/exposure mismatch')
    return source


def repair_spec(label,directory,python,repo):
    spec=launchd_spec(label,directory,python,repo)
    spec['ProcessType']='Interactive'
    spec['ProgramArguments']=[str(python),'-u','-m','SecondPass.SpatialReadout.qos_repair','supervise',str(directory)]
    return spec


def check(directory):
    directory=Path(directory)
    receipt=json.loads((directory/'repair.json').read_text())
    if digest(__file__)!=receipt['repair_source_sha256']:
        raise ValueError('Repair source changed')
    source_dir=Path(receipt['source_directory'])
    config=json.loads((directory/'config.json').read_text())
    if (source_dir/'config.json').read_bytes()!=(directory/'config.json').read_bytes():
        raise ValueError('Config bytes changed')
    validate_budget(json.loads((directory/'budget.json').read_text()),config)
    require_approval(directory,config); verify_sources(config)
    report=json.loads((source_dir/'report.json').read_text())
    outcome=json.loads((source_dir/'production_supervisor_result.json').read_text())
    if outcome['hard_cap_triggered'] or outcome['returncode']!=2:
        raise ValueError('Unexpected source supervisor outcome')
    if report['terminal_checkpoint']!=receipt['source_checkpoint']:
        raise ValueError('Repair source is not exact clean terminal')
    payload=validate_resume(load_verified(receipt['source_checkpoint']),config,report)
    return config,receipt,payload


def run(directory):
    directory=Path(directory); config,receipt,payload=check(directory)
    assert_no_worker()
    with Path(config['worker_lock']).open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=payload['state']['step']
        class ResumedSession(worker.TrainingSession):
            def train_update(self,task,cell):
                row=super().train_update(task,cell)
                if self.state['step']==start+1:
                    self.checkpoint('first_resumed_update.pt')
                return row
        scope=dict(worker.run.__globals__,
            verify_parent=lambda pointer:(payload,receipt['source_checkpoint']),
            migrate=lambda source,parent_receipt:source,
            TrainingSession=ResumedSession)
        fn=types.FunctionType(worker.run.__code__,scope,worker.run.__name__,worker.run.__defaults__,worker.run.__closure__)
        return fn(directory)


def launch(directory):
    directory=Path(directory); config,receipt,payload=check(directory)
    assert_no_worker()
    if (directory/'launch_receipt.json').exists(): raise ValueError('One-shot repair already launched')
    label='org.vawm.spatialreadout.qos-repair-'+digest(directory/'repair.json')[:12]
    domain=f'gui/{os.getuid()}'
    spec=repair_spec(label,directory,sys.executable,config['runtime_root'])
    plist=directory/'repair.launchd.plist'
    with plist.open('xb') as f: plistlib.dump(spec,f)
    subprocess.run(['/bin/launchctl','bootstrap',domain,str(plist)],check=True)
    observed=subprocess.run(['/bin/launchctl','print',domain+'/'+label],capture_output=True,text=True,check=True)
    result=dict(job_handle=domain+'/'+label,launchctl=observed.stdout,deadline=config['deadline'],source_step=payload['state']['step'])
    atomic_json(directory/'launch_receipt.json',result)
    return result


def main():
    cpu_setup()
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','launch','supervise']);p.add_argument('directory')
    args=p.parse_args(); directory=Path(args.directory)
    if args.mode=='launch': print(json.dumps(launch(directory),indent=2)); return
    if args.mode=='run': raise SystemExit(run(directory))
    config,receipt,payload=check(directory);assert_no_worker()
    with (directory/'repair_activated.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),epoch=time.time(),deadline=config['deadline']),f)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        result=supervise([sys.executable,'-u','-m','SecondPass.SpatialReadout.qos_repair','run',str(directory)],config['deadline'],directory)
    finally:
        awake.terminate();awake.wait(timeout=5)
    raise SystemExit(result['returncode'])


if __name__=='__main__': main()
