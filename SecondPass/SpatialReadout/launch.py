"""CPU prepare -> disposable bounded profile -> pin -> reviewed launchd one-shot.

Production is never started by profiling/configuring and cannot run without
REVIEW_APPROVED.json bound to the exact config and source hashes.
"""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from SecondPass.JointTraining.core import atomic_json
from SecondPass.JointTraining.worker import cpu_setup, verify_sources, utc
from SecondPass.TaskSuite.suite import CATALOG
from .protocol import (digest,new_budget,validate_budget,plan,launchd_spec,supervise,
                       assert_no_worker,require_approval,PROTOCOL,FINAL_TEST_NAMESPACE)
from .state import verify_parent,migrate,save_payload,portable_source_config

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_PARENT=ROOT/'SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/latest_checkpoint.json'


def prepare(directory,parent_pointer=DEFAULT_PARENT):
    directory=Path(directory).resolve()
    if directory.exists(): raise FileExistsError('Refusing existing destination')
    source,receipt=verify_parent(parent_pointer)
    guards=assert_no_worker()
    hashes=dict(portable_source_config(source['state']['config'])['source_hashes'])
    hashes.update({str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')})
    dependency=ROOT/'SecondPass/JointTraining/continuation_v3.py'
    hashes[str(dependency)]=digest(dependency)
    brief=Path(__file__).with_name('BRIEF.md'); hashes[str(brief)]=digest(brief)
    directory.mkdir(parents=True)
    migrated=migrate(source,receipt)
    migration=save_payload(directory/'preflight_migration.pt',migrated)
    result=dict(cpu_only=True,parent=receipt,parent_pointer=str(Path(parent_pointer).resolve()),runtime_root=str(ROOT),
        worker_lock=str(Path.home()/'VAWMRuntime/.spatial_optimizer.lock'),
        parent_step=source['state']['step'],parent_episodes=source['state']['episodes'],
        migration=migration,migration_mapping=migrated['migration'],source_hashes=hashes,
        source_manifest_sha256=source['state']['config']['source_manifest_sha256'],worker_guard=guards,
        prepared_utc=utc(),protocol=PROTOCOL,parameter_count=sum(v.numel() for v in migrated['model'].values()))
    atomic_json(directory/'preparation.json',result)
    for path,sha in hashes.items():
        destination=directory/'locked_source'/Path(path).relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(Path(path).read_bytes())
        if digest(destination)!=sha: raise ValueError('Source snapshot mismatch')
    return result


def profile(directory):
    directory=Path(directory)
    if (directory/'budget.json').exists(): raise RuntimeError('Profile cap already started; cannot restart or renew')
    preparation=json.loads((directory/'preparation.json').read_text())
    assert_no_worker()
    verify_sources(dict(preparation,catalog=CATALOG))
    # Durable cap receipt precedes subprocess startup and ANY MPS initialization.
    budget=new_budget(time.time()); atomic_json(directory/'budget.json',budget); os.chmod(directory/'budget.json',0o444)
    config=dict(preparation,**budget,catalog=CATALOG,device='mps',effective_batch=32,microbatch=4,eval_microbatch=4,
        disposable=True,cpu_only=False,production_updates=0,all_profile_state_discarded=True)
    atomic_json(directory/'profile_config.json',config)
    result=supervise([sys.executable,'-u','-m','SecondPass.SpatialReadout.worker','profile',str(directory)],
        min(budget['deadline'],budget['cap_started']+1800),directory,'profile')
    if result['returncode']!=0: raise RuntimeError('Profile failed; inspect profile supervisor/log artifacts, cap NOT renewed')
    return result


def configure(directory):
    directory=Path(directory)
    if (directory/'config.json').exists(): raise RuntimeError('Configuration already pinned; no overwrite')
    preparation=json.loads((directory/'preparation.json').read_text())
    profile_data=json.loads((directory/'profile.json').read_text())
    budget=json.loads((directory/'budget.json').read_text())
    validate_budget(budget,budget)
    if not profile_data['complete'] or not profile_data['mps_rng_replay']: raise ValueError('Profile not complete')
    source,receipt=verify_parent(preparation['parent_pointer'])
    allocation=plan(profile_data['rows'],source['scheduler'],remaining=budget['deadline']-time.time()-300.)
    config=dict(preparation,**budget,**allocation,device='mps',cpu_only=False,catalog=CATALOG,
        architecture=dict(name='SpatialReadout',encoder='unchanged CNN plus three spatial KDA modules',stack=3,center=True,
            spatial_input='Conv2d(160,64,1)',recurrence='ConvGRU64 kernel3 state64x7x7',compression='final state flatten3136 -> Linear256 -> ReLU -> inherited head',
            gate_convention='write/reset sigmoid; candidate tanh([input,reset*previous]); (1-write)*previous+write*candidate',
            initialization='PyTorch default Conv/Linear weights; recurrent biases zero; isolated CPU seed94592763'),
        optimizer=dict(name='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,groups=1,clipping=None,all_trainable=True),
        precision='float32',bptt='full_no_detach',cpu_threads=2,
        selection='maximum lexicographic (equal-task mean AUC, mean chance-normalized BA), eligible cell means, earlier ties; no inherited ranking',
        final_test_seed_namespace=FINAL_TEST_NAMESPACE,optional_baseline=False,
        parent_optimizer_updates=3393,parent_episodes=108576,additional_target_requested=2145,
        final_test_source_identities_unchanged=True,profile_state_carried=False,review_time_reserve_seconds=300,
        profile_sha256=digest(directory/'profile.json'),budget_sha256=digest(directory/'budget.json'),pinned_utc=utc())
    verify_sources(config)
    atomic_json(directory/'config.json',config); os.chmod(directory/'config.json',0o444)
    summary=dict(review_ready=True,production_launched=False,run_directory=str(directory),
        config=str(directory/'config.json'),config_sha256=digest(directory/'config.json'),source_hashes=config['source_hashes'],
        protocol=str(Path(__file__).with_name('BRIEF.md')),profile=str(directory/'profile.json'),
        preparation=str(directory/'preparation.json'),deadline=budget['deadline'],deadline_utc=utc(budget['deadline']),
        projected_optimizer_hours=config['projected_optimizer_hours'],additional_updates=config['max_steps'],
        required_review_receipt=dict(path=str(directory/'REVIEW_APPROVED.json'),approved=True,reviewer='parent-independent-read-only',
            config_sha256=digest(directory/'config.json'),source_hashes=config['source_hashes']),
        launch_command=[sys.executable,'-m','SecondPass.SpatialReadout.launch','launch',str(directory)])
    atomic_json(directory/'REVIEW_READY.json',summary)
    atomic_json(Path(__file__).with_name('REVIEW_READY.json'),summary)
    return summary


def checked_config(directory):
    config=json.loads((directory/'config.json').read_text()); budget=json.loads((directory/'budget.json').read_text())
    validate_budget(budget,config); require_approval(directory,config); verify_sources(config)
    if digest(directory/'budget.json')!=config['budget_sha256'] or digest(directory/'profile.json')!=config['profile_sha256']:
        raise ValueError('Pinned budget/profile changed')
    if (directory/'migration.pt').exists(): raise RuntimeError('Production already started; no resume authorized')
    if time.time()+config['estimated_total_remaining_seconds']>config['deadline']:
        raise RuntimeError('Review/setup consumed feasibility slack; do not silently reduce target or renew cap')
    return config


def launch(directory):
    directory=Path(directory).resolve(); config=checked_config(directory)
    guard=assert_no_worker()
    domain=f'gui/{os.getuid()}'
    subprocess.run(['/bin/launchctl','print',domain],stdout=subprocess.DEVNULL,check=True)
    label='org.vawm.spatialreadout.'+directory.parent.name.replace('_','-')+'-'+digest(directory/'config.json')[:12]
    plist=directory/'production.launchd.plist'
    if (directory/'launch_attempt.json').exists(): raise RuntimeError('One-shot launch already attempted; no automatic restart')
    spec=launchd_spec(label,directory,sys.executable,Path(config['runtime_root']))
    with plist.open('xb') as f: plistlib.dump(spec,f); f.flush(); os.fsync(f.fileno())
    atomic_json(directory/'launch_attempt.json',dict(label=label,domain=domain,plist=str(plist),
        utc=utc(),guard=guard,config_sha256=digest(directory/'config.json'),deadline=config['deadline'],owner='macOS user launchd; not Hermes'))
    bootstrap=subprocess.run(['/bin/launchctl','bootstrap',domain,str(plist)],text=True,capture_output=True)
    target=domain+'/'+label
    status=subprocess.run(['/bin/launchctl','print',target],text=True,capture_output=True)
    receipt=dict(label=label,job_handle=target,plist=str(plist),bootstrap_returncode=bootstrap.returncode,
        bootstrap_stdout=bootstrap.stdout,bootstrap_stderr=bootstrap.stderr,launchctl_print_returncode=status.returncode,
        launchctl_print=status.stdout,deadline=config['deadline'],deadline_utc=utc(config['deadline']),
        external_owner='launchd',KeepAlive=False,restarts_authorized=False,
        note='Launchd readback alone is not optimizer progress; verify progress.jsonl and independently load latest checkpoint')
    atomic_json(directory/'launch_receipt.json',receipt)
    if bootstrap.returncode or status.returncode: raise RuntimeError('launchd bootstrap/readback failed; see launch_receipt.json')
    return receipt


def production_supervise(directory):
    directory=Path(directory).resolve(); config=checked_config(directory)
    if not (directory/'launch_attempt.json').exists(): raise RuntimeError('Durable launchd handoff missing')
    attempt=json.loads((directory/'launch_attempt.json').read_text())
    if attempt['config_sha256']!=digest(directory/'config.json'): raise ValueError('Launch attempt config changed')
    assert_no_worker()
    # Exclusive one-shot activation, regardless of launchctl kickstart/reloads.
    with (directory/'production_activated.json').open('x') as f:
        json.dump(dict(utc=utc(),supervisor_pid=os.getpid(),parent_pid=os.getppid(),deadline=config['deadline']),f)
        f.flush(); os.fsync(f.fileno())
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        result=supervise([sys.executable,'-u','-m','SecondPass.SpatialReadout.worker','run',str(directory)],config['deadline'],directory)
    finally:
        awake.terminate(); awake.wait(timeout=5)
    if not (directory/'report.json').exists():
        latest=json.loads((directory/'latest_checkpoint.json').read_text()) if (directory/'latest_checkpoint.json').exists() else None
        atomic_json(directory/'report.json',dict(protocol=PROTOCOL,stop_reason='worker_exit_without_report',supervisor=result,
            latest_checkpoint=latest,final_coverage_complete=False,production_completed=False))
        (directory/'REPORT.md').write_text('# Interrupted spatial-readout run\n\nNo completed final coverage. See report.json, failure logs and partial evaluation files.\n')
    return result


def main():
    cpu_setup()
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','profile','configure','launch','supervise']); parser.add_argument('directory')
    parser.add_argument('--source',default=str(DEFAULT_PARENT)); args=parser.parse_args(); directory=Path(args.directory).resolve()
    functions={'profile':profile,'configure':configure,'launch':launch,'supervise':production_supervise}
    result=prepare(directory,args.source) if args.mode=='prepare' else functions[args.mode](directory)
    print(json.dumps(result,indent=2),flush=True)
    if args.mode=='supervise': raise SystemExit(result['returncode'])


if __name__=='__main__': main()
