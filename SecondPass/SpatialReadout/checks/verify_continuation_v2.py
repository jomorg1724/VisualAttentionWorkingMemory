"""Read-only CPU verification of real continuation state and full-cycle speed."""
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from SecondPass.JointTraining.core import tree_equal, BalancedScheduler, atomic_json
from SecondPass.JointTraining.worker import cpu_setup, utc, verify_sources
from SecondPass.SpatialReadout.state import load_verified
from SecondPass.SpatialReadout.continuation_v2 import SOURCE_SHA, START, TARGET, TASKS
from SecondPass.SpatialReadout.protocol import digest


def verify(directory):
    p=Path(directory)
    config=json.loads((p/'config.json').read_text())
    lineage=json.loads((p/'migration_integrity.json').read_text())
    before=load_verified(lineage['source']);migration=load_verified(lineage['migration'])
    assert lineage['source']['sha256']==SOURCE_SHA
    preserved={k:tree_equal(before[k],migration[k]) for k in ('model','optimizer','optimizer_names','scheduler','stream','rng','migration')}
    preserved.update({'state.'+k:tree_equal(v,migration['state'][k]) for k,v in before['state'].items() if k not in ('config','deadline','elapsed_cap_seconds')})
    assert all(preserved.values()),preserved
    checkpoints=[json.loads(l) for l in (p/'checkpoints.jsonl').read_text().splitlines()]
    first_receipt=next(r for r in checkpoints if Path(r['path']).name=='first_resumed_update.pt')
    cycle_receipt=next(r for r in checkpoints if r['step']==START+13)
    first=load_verified(first_receipt);cycle=load_verified(cycle_receipt)
    rows=[json.loads(l) for l in (p/'progress.jsonl').read_text().splitlines()]
    window=rows[:13]
    assert [r['step'] for r in window]==list(range(START+1,START+14))
    assert len({r['task'] for r in window})==13
    expected=BalancedScheduler(0);expected.load_state_dict(before['scheduler'])
    assert [expected.next() for _ in range(13)]==[(r['task'],r['cell']) for r in window]
    assert tree_equal(expected.state_dict(),cycle['scheduler'])
    assert first['state']['step']==START+1 and cycle['state']['step']==START+13
    assert first['state']['episodes']==before['state']['episodes']+32
    assert cycle['state']['episodes']==before['state']['episodes']+13*32
    assert cycle['state']['optimizer_seconds']>first['state']['optimizer_seconds']>before['state']['optimizer_seconds']
    assert abs(cycle['state']['optimizer_seconds']-before['state']['optimizer_seconds']-sum(r['seconds'] for r in window))<1e-6
    changed=[n for n in first['model'] if not torch.equal(first['model'][n],before['model'][n])]
    conv=[n for n in changed if n.startswith('spatial_gru.')]
    assert len(conv)==4
    advanced={}
    active_task=window[0]['task']
    for i,name in enumerate(first['optimizer_names']):
        old=before['optimizer']['state'][i];new=first['optimizer']['state'][i]
        active=not name.startswith('heads.') or name.startswith('heads.'+active_task+'.')
        assert float(new['step'])==float(old['step'])+int(active),(name,old['step'],new['step'])
        if active:
            assert not tree_equal(new,old)
            advanced[name]=[float(old['step']),float(new['step'])]
        else:assert tree_equal(new,old)
    assert first['state']['selection_history']==before['state']['selection_history']
    costs={(r['task'],r['cell']):r['seconds'] for r in config['cell_costs']}
    expected_seconds=sum(costs[r['task'],r['cell']] for r in window)
    actual_seconds=sum(r['seconds'] for r in window)
    speed=actual_seconds/expected_seconds
    assert speed<1.25,('Material speed discrepancy',speed)
    sup=json.loads((p/'production_supervisor.json').read_text())
    activation=json.loads((p/'production_activated.json').read_text())
    job=json.loads((p/'launch_receipt.json').read_text())['job_handle']
    assert activation['parent_pid']==1 and sup['deadline']==config['deadline']
    os_state=subprocess.check_output(['ps','-p',str(sup['pid'])+','+str(sup['worker_pid']),'-o','pid=,ppid=,ni=,pri=,%cpu=,etime=,command='],text=True)
    job_state=subprocess.check_output(['/bin/launchctl','print',job],text=True)
    assert 'spawn type = interactive' in job_state and 'state = running' in job_state
    verify_sources(config)
    result=dict(verified=True,utc=utc(),run_directory=str(p),job_handle=job,
        supervisor_pid=sup['pid'],worker_pid=sup['worker_pid'],os_state=os_state,
        start_utc=activation['started_utc'],deadline_utc=activation['deadline_utc'],deadline=config['deadline'],
        source_sha256=SOURCE_SHA,config_sha256=digest(p/'config.json'),preserved=preserved,
        changed_first_update_tensors=len(changed),changed_convgru_tensors=conv,advanced_adam=advanced,
        first_checkpoint=first_receipt,cycle_checkpoint=cycle_receipt,
        first_cycle_updates=13,cycle_cumulative_step=cycle['state']['step'],cycle_cumulative_episodes=cycle['state']['episodes'],
        latest_persisted_step=rows[-1]['step'],latest_progress_utc=rows[-1]['utc'],
        measured_cycle_seconds=actual_seconds,matched_repaired_estimate_seconds=expected_seconds,observed_to_repaired_ratio=speed,
        cycle_rows=[{k:r[k] for k in ('step','task','cell','loss','seconds')} for r in window],
        projected_optimizer_hours=config['projected_optimizer_hours'],projected_total_hours=config['estimated_total_remaining_seconds']/3600,
        slower_150pct_total_hours=config['slower_150pct_training_seconds']/3600,
        additional_target=5070,cumulative_target=TARGET,additional_episodes=162240,
        validation_steps=config['validation_steps'],final_test_seed_namespace=config['final_test_seed_namespace'],
        no_restart=True,prior_sources_unchanged=True)
    atomic_json(p/'continuation_verified.json',result)
    return result


if __name__=='__main__':
    cpu_setup()
    print(json.dumps(verify(sys.argv[1]),indent=2))
