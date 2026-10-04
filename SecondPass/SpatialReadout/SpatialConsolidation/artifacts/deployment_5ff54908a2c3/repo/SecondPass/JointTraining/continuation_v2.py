"""Explicit v2 allocation amendment; preserve v1 state and its original cap."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

from .core import restore_checkpoint

PROTOCOL = 'continuation_v2'
FINAL_TEST_NAMESPACE = 94292763

from SecondPass.TaskSuite.suite import SuiteStream, TASKS
from .core import BalancedScheduler


class FinalTestStream(SuiteStream):
    """Fresh draws, unchanged official test identities and native renderers."""
    def __init__(self, split='test'):
        if split != 'test':
            raise ValueError('Fresh namespace is final-test-only')
        super().__init__('test')

    def stream_seed(self, task, cell):
        index, _ = self._cell(task, cell)
        return FINAL_TEST_NAMESPACE * 100000 + TASKS[task]['stream_id'] * 1000 + index


def plan_continuation(old, state, scheduler_state, progress, validation, now=None):
    """Use measured update/cell timing only, not validation or test scores."""
    import statistics
    now = time.time() if now is None else now
    per_cell = {}
    for row in progress:
        per_cell.setdefault((row['task'], row['cell']), []).append(row['seconds'])
    # Equal-task/within-task-condition expected cost, not the current partial cycle.
    cell_costs = {}
    for task, spec in TASKS.items():
        task_values = [r['seconds'] for r in progress if r['task'] == task]
        if not task_values:
            raise ValueError('Missing live timing for task ' + task)
        for cell in spec['conditions']:
            values = per_cell.get((task, cell['id']), task_values)
            cell_costs[(task, cell['id'])] = statistics.mean(values)
    equal_task_seconds = statistics.mean([
        statistics.mean([cell_costs[(task, c['id'])] for c in spec['conditions']])
        for task, spec in TASKS.items()])
    expected = {(t,c['id']) for t,s in TASKS.items() for c in s['conditions']}
    if {(r['task'],r['cell']) for r in validation['cells']} != expected:
        raise ValueError('Cost projection requires every validation cell')
    def evaluation_cost(n, krauzlis_n):
        return 1.25 * sum(r['seconds'] / r['n'] *
                          (krauzlis_n if r['task']=='krauzlis_cued_motion' else n)
                          for r in validation['cells'])
    val_cost = evaluation_cost(old['val_n'], old['val_krauzlis_n'])
    test_cost = evaluation_cost(128, 200)
    # Reserve one new full-cell validation, both final checkpoints, and save/report.
    reserve = val_cost + 2 * test_cost + 240.
    margin = 420.
    estimate = equal_task_seconds * 1.20
    available = old['deadline'] - now - reserve - margin
    steps = int((state['step'] + available / estimate) // 13) * 13
    if steps <= state['step'] or available <= 0:
        raise ValueError('Insufficient original budget for continuation and full final coverage')
    scheduler = BalancedScheduler(0); scheduler.load_state_dict(scheduler_state)
    exposure = copy.deepcopy(state['exposure'])
    for _ in range(state['step'], steps):
        task, cell = scheduler.next()
        exposure[task]['updates'] += 1
        exposure[task]['episodes'] += old['effective_batch']
        exposure[task]['cells'][cell] += old['effective_batch']
    planned = {t:dict(updates=e['updates'], episodes=e['episodes'], cell_episodes=e['cells'])
               for t,e in exposure.items()}
    return dict(old, protocol_version=PROTOCOL, max_steps=steps, planned_cycles=steps//13,
                planned_exposure=planned, validation_steps=[steps], test_n=128, test_krauzlis_n=200,
                final_reserve_seconds=reserve, estimated_validation_seconds=val_cost,
                estimated_one_test_seconds=test_cost, estimated_cycle_seconds=estimate*13,
                update_estimate_seconds=max(r['seconds'] for r in progress)*1.25,
                estimated_total_seconds=(steps-state['step'])*estimate+reserve+margin,
                final_test_seed_namespace=FINAL_TEST_NAMESPACE,
                amendment=dict(reason='Initial 156-update plan overallocated evaluation; user retained original four-hour cap.',
                    pinned_epoch=now, carried_step=state['step'], measured_equal_task_seconds=equal_task_seconds,
                    train_safety_factor=1.20, eval_safety_factor=1.25, extra_margin_seconds=margin,
                    new_validation_looks=1, final_ordinary_n_before=old.get('test_n',256),
                    final_ordinary_n_after=128, test_selection_prohibited=True,
                    prior_test_draws_possibly_forwarded=True,
                    prior_test_metrics_inspected=False,
                    source_identity_test_split_unchanged=True))


def restore_for_continuation(receipt, model, optimizer, scheduler, stream, config, now=None):
    """Validate before loading; only the documented allocation fields may change."""
    import torch
    now = time.time() if now is None else now
    path = Path(receipt['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Source checkpoint digest mismatch')
    saved = torch.load(path, map_location='cpu')
    prior = saved['state']['config']
    mutable = {'max_steps', 'planned_cycles', 'planned_exposure', 'validation_steps',
               'test_n', 'test_krauzlis_n', 'final_reserve_seconds',
               'estimated_validation_seconds', 'estimated_one_test_seconds',
               'estimated_cycle_seconds', 'estimated_total_seconds',
               'update_estimate_seconds', 'protocol_version', 'amendment',
               'final_test_seed_namespace', 'continuation_source_hashes'}
    for key in prior.keys() | config.keys():
        if key not in mutable and prior.get(key) != config.get(key):
            raise ValueError('Unauthorized protocol change: ' + key)
    if config['deadline'] != saved['state']['deadline']:
        raise ValueError('Deadline mismatch')
    if now >= config['deadline']:
        raise ValueError('Original wall cap expired')
    if saved['state']['step'] >= config['max_steps']:
        raise ValueError('Continuation horizon already complete')
    state = copy.deepcopy(restore_checkpoint(path, model, optimizer, scheduler, stream, config['device']))
    state['config'] = copy.deepcopy(config)
    state['continuation'] = dict(protocol=PROTOCOL, source=receipt, carried_step=state['step'])
    return state


def evaluate_final(model, config, device, output, deadline):
    """Identical cell scorer; fresh final-only stream, no pooled concealment."""
    import torch
    from . import worker as w
    from .core import atomic_json, score_cell, summarize
    stream = FinalTestStream(); model.eval(); started = time.perf_counter(); rows = []
    complete = True
    with torch.no_grad():
        for task, cell in w.all_cells():
            count = config['test_krauzlis_n'] if task == 'krauzlis_cued_motion' else config['test_n']
            labels = []; probabilities = []; metadata = []; cell_start = time.perf_counter()
            for i in range(0, count, config['eval_microbatch']):
                if time.time() >= deadline:
                    complete = False; break
                images, y, meta = stream.batch(min(config['eval_microbatch'], count-i), task, cell)
                logits = model(images.to(device), task)
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError('Nonfinite final logits')
                probabilities.extend(logits.softmax(1).cpu().numpy().tolist())
                labels.extend(y.tolist()); metadata.extend(meta)
            if not complete:
                break
            w.sync(device)
            row = score_cell(task, cell, labels, probabilities, metadata)
            row['seconds'] = time.perf_counter()-cell_start; rows.append(row)
            atomic_json(str(output)+'.partial.json', dict(split='test', seed_namespace=FINAL_TEST_NAMESPACE,
                        complete_cells=len(rows), expected_cells=len(w.all_cells()), cells=rows))
    result = dict(split='test', seed_namespace=FINAL_TEST_NAMESPACE, n=config['test_n'],
                  krauzlis_n=config['test_krauzlis_n'], cells=rows, expected_cells=len(w.all_cells()),
                  complete_cells=len(rows), complete=complete and len(rows)==len(w.all_cells()),
                  seconds=time.perf_counter()-started, summary=summarize(rows), fixed_draws=True)
    atomic_json(output, result)
    return result


def run(directory, config, receipt):
    import json
    import os
    import signal
    import traceback
    import torch
    from . import worker as w
    from .core import atomic_json, append_jsonl, save_checkpoint, tree_equal, may_update
    directory = Path(directory)
    if (directory/'migration.pt').exists():
        raise RuntimeError('Refusing automatic restart or overwrite of continuation')
    w.verify_sources(config)
    for source, digest in config.get('continuation_source_hashes', {}).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen continuation source changed: '+source)
    device = config['device']
    model = w.AccumulatorBaseline(w.task_classes(), stack=3, center=True, accumulator='kda').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)
    scheduler = BalancedScheduler(config['scheduler_seed']); stream = SuiteStream('train')
    state = restore_for_continuation(receipt, model, optimizer, scheduler, stream, config)
    carried = state['step']; latest = None; stopped = [False]; handlers = {}

    def status(phase, **extra):
        atomic_json(directory/'live_status.json', dict(phase=phase, pid=os.getpid(), utc=w.utc(),
            step=state['step'], episodes=state['episodes'], frames=state['frames'],
            optimizer_seconds=state['optimizer_seconds'], deadline_epoch=config['deadline'],
            deadline=w.utc(config['deadline']), remaining_seconds=max(0.,config['deadline']-time.time()),
            latest_checkpoint=latest, best_step=state['best_step'], carried_step=carried,
            pinned_target=config['max_steps'], **extra))

    def checkpoint(filename):
        nonlocal latest
        state['elapsed_cap_seconds'] = time.time()-config['cap_started']
        latest = save_checkpoint(directory/filename, model, optimizer, scheduler, stream, state, device)
        atomic_json(directory/'latest_checkpoint.json', latest)
        append_jsonl(directory/'checkpoints.jsonl', dict(utc=w.utc(), **latest))
        return latest

    checkpoint('migration.pt')
    original = torch.load(receipt['path'], map_location='cpu')
    migrated = torch.load(directory/'migration.pt', map_location='cpu')
    exact = all(tree_equal(original[key], migrated[key]) for key in ('model','optimizer','scheduler','stream','rng'))
    if not exact:
        raise RuntimeError('Migration changed model/Adam/scheduler/stream/RNG')
    atomic_json(directory/'resume_integrity.json', dict(exact_nonallocation_state=True,
        source=receipt, migration=latest, carried_step=carried, deadline=config['deadline'],
        carried_episodes=state['episodes'], carried_optimizer_seconds=state['optimizer_seconds'],
        carried_selection_history=state['selection_history'], verified_utc=w.utc()))
    del original, migrated
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    status('training', initialization='exact_full_state_continuation')
    failure = None; reason = 'planned_complete_cycles'; terminal_test = None; selected_test = None
    try:
        while state['step'] < config['max_steps']:
            if not may_update(now=time.time(), deadline=config['deadline'], reserve=config['final_reserve_seconds'],
                              estimate=config['update_estimate_seconds'], step=state['step'],
                              max_steps=config['max_steps'], stopped=stopped[0]):
                reason = 'signal' if stopped[0] else 'wall_budget_reserve'; break
            task, cell = scheduler.next()
            row = w.update(model, optimizer, stream, task, cell, config['effective_batch'], config['microbatch'], device)
            state['step'] += 1; state['episodes'] += row['episodes']; state['frames'] += row['frames']
            state['optimizer_seconds'] += row['seconds']
            e = state['exposure'][task]
            e['updates'] += 1; e['episodes'] += row['episodes']; e['frames'] += row['frames']; e['cells'][cell] += row['episodes']
            row.update(utc=w.utc(), step=state['step'], cumulative_episodes=state['episodes'],
                cumulative_frames=state['frames'], optimizer_seconds=state['optimizer_seconds'],
                elapsed_cap_seconds=time.time()-config['cap_started'], per_task_exposure=state['exposure'],
                clipping=None, protocol=PROTOCOL, carried_step=carried,
                latest_validation=state['selection_history'][-1] if state['selection_history'] else None)
            append_jsonl(directory/'progress.jsonl', row)
            print(json.dumps({k:row[k] for k in ('utc','step','task','cell','loss','seconds','cumulative_episodes')}), flush=True)
            if state['step'] == carried+1 or state['step'] % config['checkpoint_every'] == 0:
                checkpoint(f"checkpoint_{state['step']:06d}.pt")
            status('training', last_update={k:row[k] for k in ('task','cell','loss','grad_norm','seconds')})
        terminal = checkpoint('terminal.pt')
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        atomic_json(directory/'failure.json', failure)
        reason = 'nonfinite_safe_stop' if isinstance(exc, FloatingPointError) else 'worker_error'
        state = restore_checkpoint(latest['path'], model, optimizer, scheduler, stream, device)
        terminal = checkpoint('terminal_recovered.pt')
    try:
        w.verify_sources(config)
        # One additional look: at the pinned horizon, or earlier only on budget/signal termination.
        # Preserve all completed v1 looks and the same validation-only selection rule.
        if failure is None and time.time() < config['deadline']-2*config['estimated_one_test_seconds']-120:
            status('validation')
            validation = w.evaluate(model, 'val', config['val_n'], config['val_krauzlis_n'],
                config['eval_microbatch'], device, directory/f"validation_{state['step']:06d}.json",
                deadline=config['deadline']-2*config['estimated_one_test_seconds']-90)
            key = validation['summary']['selection_key'] if validation['complete'] else None
            state['selection_history'].append(dict(step=state['step'], key=key, complete=validation['complete'], protocol=PROTOCOL))
            filename = f"validation_checkpoint_{state['step']:06d}.pt"
            if key is not None and (state['best_key'] is None or tuple(key)>tuple(state['best_key'])):
                state['best_key'] = key; state['best_step'] = state['step']; state['best_checkpoint'] = str(directory/filename)
            checkpoint(filename)
        if time.time() < config['deadline']-60:
            status('final_test_terminal')
            # Reserve a separate selected pass even when the two checkpoints may coincide.
            selected_reserve = config['estimated_one_test_seconds'] if state['best_step'] != state['step'] and state['best_checkpoint'] else 0
            terminal_test = evaluate_final(model, config, device, directory/'test_terminal.json',
                                           config['deadline']-selected_reserve-60)
            if state['best_step'] == state['step']:
                selected_test = dict(reused_terminal=True, selected_step=state['best_step'], results=terminal_test)
                atomic_json(directory/'test_selected.json', selected_test)
            elif state['best_checkpoint'] and time.time() < config['deadline']-60:
                saved = torch.load(state['best_checkpoint'], map_location='cpu'); model.load_state_dict(saved['model']); del saved
                status('final_test_selected')
                selected_test = evaluate_final(model, config, device, directory/'test_selected.json', config['deadline']-45)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        atomic_json(directory/'finalization_failure.json', failure)
    selected_results = selected_test.get('results', selected_test) if selected_test else None
    coverage_complete = bool(terminal_test and terminal_test['complete'] and selected_results and selected_results['complete'])
    report = dict(protocol=PROTOCOL, stop_reason=reason, failure=failure, config=config,
        carried_step=carried, terminal_step=state['step'], selected_step=state['best_step'],
        selection_history=state['selection_history'], terminal_checkpoint=terminal,
        episodes=state['episodes'], frames=state['frames'], optimizer_seconds=state['optimizer_seconds'],
        exposure=state['exposure'], terminal_test=terminal_test, selected_test=selected_test,
        final_coverage_complete=coverage_complete, finished_utc=w.utc(),
        elapsed_cap_seconds=time.time()-config['cap_started'],
        limitations=['Original evaluation-heavy allocation amended explicitly, not an independent new run.',
          'More substantive acquisition, not sufficient convergence or overnight-equivalent exposure.',
          'One seed and one architecture; no source-independent intervals or paired-delay claim.',
          'Old final-test draws may have started during handover; no test scores used in planning or selection.',
          'Fresh final-only seed namespace retains original official test source identities.',
          'N0 specificity/FPR and Krauzlis event/side denominators reported separately per cell.'])
    atomic_json(directory/'report.json', report)
    (directory/'REPORT.md').write_text(f"# Four-hour KDA continuation v2\n\nTerminal step {state['step']}; selected {state['best_step']}. "
        f"Cumulative episodes {state['episodes']}. Stop: {reason}. Complete selected/terminal coverage: {coverage_complete}.\n\n"
        + '\n'.join('- '+s for s in report['limitations']) + '\n\nCell results: `test_terminal.json`, `test_selected.json`; full provenance: `report.json`.\n')
    status('finished', report=str(directory/'report.json'), stop_reason=reason, failure=failure, final_coverage_complete=coverage_complete)
    for sig, handler in handlers.items():
        signal.signal(sig, handler)
    return report


def is_joint_worker(command):
    """Match the executable and interpreter argv, never echoed child commands."""
    tokens = command.split()
    if not tokens or not Path(tokens[0]).name.lower().startswith('python'):
        return False
    index = 1
    while index < len(tokens) and tokens[index] in {'-u','-B','-O','-OO','-I','-E','-s','-S','-b','-bb','-q'}:
        index += 1
    if tokens[index:index+1] != ['-m']:
        return False
    return tokens[index+1:index+3] in [
        ['SecondPass.JointTraining.worker','run'],
        ['SecondPass.JointTraining.continuation_v2','worker']]


def main():
    import argparse
    import fcntl
    import json
    import os
    import subprocess
    import sys
    from . import worker as w
    from .launch import supervise
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['run','worker']); p.add_argument('directory')
    args = p.parse_args(); directory = Path(args.directory).resolve()
    config = json.loads((directory/'config.json').read_text())
    if time.time() >= config['deadline']:
        raise ValueError('Original cap expired; refusing supervisor/worker launch')
    if (directory/'migration.pt').exists():
        raise RuntimeError('Continuation already started; no automatic restart')
    migration = json.loads((directory/'migration.json').read_text())
    if config['deadline'] != migration['original_deadline'] or config['cap_started'] != migration['original_cap_started']:
        raise ValueError('Original budget changed')
    # Only inspect process identity; never kill another worker here.
    for line in subprocess.check_output(['ps','-axo','pid=,command='], text=True).splitlines():
        if not line.strip():
            continue
        pid, command = line.strip().split(None, 1)
        if int(pid) == os.getpid():
            continue
        if is_joint_worker(command):
            raise RuntimeError('Another joint-training worker is still alive: '+pid)
    if args.mode == 'run':
        result = supervise(['/usr/bin/caffeinate','-i',sys.executable,'-u','-m',
            'SecondPass.JointTraining.continuation_v2','worker',str(directory)], config['deadline'], directory)
        print(json.dumps(result), flush=True)
        raise SystemExit(result['returncode'])
    with (directory.parent/'.joint_training_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        w.cpu_setup()
        report = run(directory, config, migration['source_checkpoint'])
        raise SystemExit(0 if report['failure'] is None and report['final_coverage_complete'] else 2)


if __name__ == '__main__':
    main()
