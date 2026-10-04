"""Finite-budget local supervisor and recipe planner; no auto-restart."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .core import TASKS, atomic_json


def supervise(command, deadline, directory):
    directory = Path(directory)
    process = subprocess.Popen(command, start_new_session=True)
    atomic_json(directory/'supervisor.json', dict(pid=os.getpid(), worker_pid=process.pid, deadline=deadline, command=command))
    hard = False
    try:
        process.wait(timeout=max(0., deadline-time.time()))
    except subprocess.TimeoutExpired:
        hard = True
        # The exact child process group only. No broad process-name kill.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    result = dict(returncode=process.returncode, hard_cap_triggered=hard, finished=time.time(),
                  supervisor_pid=os.getpid(), worker_pid=process.pid, deadline=deadline)
    atomic_json(directory/'supervisor_result.json', result)
    return result


def frames(task, condition):
    if TASKS[task]['group'] == 'sensory': return 2
    if task == 'orientation_ring': return 4
    from WorkingMemory.SpatialTaskBattery.stimuli import frame_count
    return frame_count(task, condition['kwargs'])


def plan(profile, remaining):
    costs = {r['task']:r for r in profile['rows']}
    def eval_cost(n, krauzlis):
        return 1.7*sum(costs[t]['eval_seconds']/costs[t]['eval_n']*
            (krauzlis if t=='krauzlis_cued_motion' else n)*frames(t,c)/costs[t]['sequence_frames']
            for t,spec in TASKS.items() for c in spec['conditions'])
    choice = None
    for val_n,test_n in ((64,256),(32,128),(32,64),(16,32)):
        val_s=eval_cost(val_n,100); test_s=eval_cost(test_n,200)
        reserve=2*test_s+240
        available=remaining-reserve-4*val_s-180
        for effective in (64,32,16):
            cycle = 1.5*sum(costs[t]['seconds']*(effective/64)*
                (sum(frames(t,c) for c in spec['conditions'])/len(spec['conditions']))/costs[t]['sequence_frames']
                for t,spec in TASKS.items())+3
            cycles = max(0, int(available/cycle)//12*12)
            if cycles >= 12:
                choice=dict(effective_batch=effective, microbatch=4, eval_microbatch=4,
                    max_steps=cycles*13, planned_cycles=cycles, val_n=val_n, val_krauzlis_n=100,
                    test_n=test_n, test_krauzlis_n=200, validation_steps=[(cycles*k//4)*13 for k in (1,2,3,4)],
                    final_reserve_seconds=reserve, estimated_validation_seconds=val_s,
                    estimated_one_test_seconds=test_s, estimated_cycle_seconds=cycle,
                    update_estimate_seconds=2*max(r['seconds'] for r in costs.values())*effective/64,
                    estimated_total_seconds=cycles*cycle+4*val_s+reserve+180, checkpoint_every=13)
                break
        if choice: break
    if choice is None: raise RuntimeError('No complete 12-cycle joint exposure plus final evaluation fits remaining fixed cap')
    choice['planned_exposure'] = {t:dict(updates=choice['planned_cycles'], episodes=choice['planned_cycles']*choice['effective_batch'],
        cell_episodes={c['id']:choice['planned_cycles']//len(spec['conditions'])*choice['effective_batch'] for c in spec['conditions']})
        for t,spec in TASKS.items()}
    return choice


def source_hashes():
    paths = [Path(__file__).with_name(p) for p in ('core.py','worker.py','launch.py')]
    paths += [Path(p) for p in ('SecondPass/TaskSuite/suite.py','SecondPass/TaskSuite/catalog.json',
        'WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py',
        'PreAttentiveVision/neuroscience_stimuli.py','PreAttentiveVision/natural_stimuli.py',
        'WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/PlainBaseline/variants.py')]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=['profile','configure','run']); p.add_argument('directory')
    args=p.parse_args(); directory=Path(args.directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    budget_path=directory/'budget.json'
    if args.mode=='profile':
        if budget_path.exists(): raise RuntimeError('Budget already started; never renew it')
        started=time.time()
        budget=dict(cap_started=started, deadline=started+14400, wall_cap_seconds=14400,
                    note='Starts before first accelerator profile; includes setup after profile, production, validation and finalization')
        atomic_json(budget_path,budget)
        result=supervise([sys.executable,'-m','SecondPass.JointTraining.worker','profile',str(directory)],budget['deadline'],directory)
        raise SystemExit(result['returncode'])
    budget=json.loads(budget_path.read_text())
    if args.mode=='configure':
        from SecondPass.TaskSuite.suite import CATALOG
        from PreAttentiveVision.natural_stimuli import DATA_ROOT, sha256
        profile=json.loads((directory/'profile.json').read_text())
        config=dict(**budget, **plan(profile,budget['deadline']-time.time()),
            device='mps', seed=94182763, scheduler_seed=94182991,
            authorization='Explicit user authorization for local fresh-weight joint KDA on all 13 tasks; historical assembly flag is not run authorization.',
            architecture=dict(class_name='AccumulatorBaseline', stack=3, center=True, accumulator='kda', heads=13),
            optimizer=dict(name='Adam',lr=1e-4,eps=1e-8,betas=[.9,.999],weight_decay=0,parameter_groups=1,clipping=None),
            precision='float32', bptt='full', cpu_threads=2, initial_checkpoint=None,
            task_policy='shuffled equal 13-task cycles; independently shuffled balanced condition cycles; task-local streams',
            loss='one task/cell update, mean CE over effective batch; no auxiliary labels',
            profile_lineage='Disposable longest-condition profile updates; incompatible with balanced condition scheduler. All production weights/Adam/streams reinitialized.',
            selection='maximum lexicographic (minimum chance-normalized task BA, equal-task mean AUC), equal cell means excluding N0; earlier ties',
            catalog=CATALOG, source_manifest_sha256=sha256(Path(DATA_ROOT)/'manifest.json'), source_hashes=source_hashes())
        atomic_json(directory/'config.json',config)
        # Portable evidence snapshot: the live source hashes above are checked by worker.
        source=directory/'locked_source'; source.mkdir(exist_ok=True)
        for path in config['source_hashes']:
            rel=Path(path).relative_to(Path.cwd()); dest=source/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(Path(path).read_bytes())
        os.chmod(directory/'config.json',0o444)
        print(json.dumps({k:v for k,v in config.items() if k not in ('catalog','source_hashes')},indent=2))
    else:
        config=json.loads((directory/'config.json').read_text())
        if time.time()>=budget['deadline']: raise RuntimeError('Original wall cap already expired')
        if config['deadline']!=budget['deadline']: raise RuntimeError('Deadline changed')
        command=['/usr/bin/caffeinate','-i',sys.executable,'-m','SecondPass.JointTraining.worker','run',str(directory)]
        result=supervise(command,budget['deadline'],directory)
        print(json.dumps(result),flush=True)
        raise SystemExit(result['returncode'])


if __name__=='__main__': main()
