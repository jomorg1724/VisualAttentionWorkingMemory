"""Versioned, unchanged-architecture continuation of FreshRun (never a fresh start)."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch
from SecondPass.SpatialReadout.FreshRun import worker as fresh
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, append_jsonl, cpu_tree, tree_equal
from SecondPass.TaskSuite.suite import TASKS

cloud = fresh.cloud
PROTOCOL = 'final_spatial_convgru_continuation_v1'
ADDITIONAL = 104000
VALIDATION_EVERY = 10400
CHECKPOINT_EVERY = 260
FINAL_NAMESPACE = 98792763
WORKER_LOCK = fresh.ROOT.parent/'convgru_fresh_worker.lock'


class EvaluationStream(fresh.FreshStream):
    final_namespace = FINAL_NAMESPACE

    def stream_seed(self, task, cell):
        if self.split != 'test':
            return super().stream_seed(task, cell)
        index, _ = self._cell(task, cell)
        return self.final_namespace*100000 + TASKS[task]['stream_id']*1000 + index


def evaluate(model, split, n, krauzlis_n, microbatch, device, output, deadline, cells=None):
    # Fresh.evaluate clones the INNER evaluator; inject its actual stream global.
    return cloud.clone(fresh.evaluate, FreshStream=EvaluationStream, FINAL_NAMESPACE=FINAL_NAMESPACE)(
        model, split, n, krauzlis_n, microbatch, device, output, deadline, cells=cells)


def validate_budget(budget, config, now=None):
    now = time.time() if now is None else now
    if any(config.get(k) != v for k, v in budget.items()):
        raise ValueError('Absolute transition budget changed')
    if not all(math.isfinite(budget[k]) for k in ('cap_started','hard_deadline','deadline',
            'wall_cap_seconds','retrieval_reserve_seconds','max_usd','additional_updates')):
        raise ValueError('Nonfinite budget')
    if (budget['wall_cap_seconds'] != 129600 or budget['hard_deadline']-budget['cap_started'] != 129600
            or budget['retrieval_reserve_seconds'] < 600
            or budget['hard_deadline']-budget['deadline'] != budget['retrieval_reserve_seconds']):
        raise ValueError('Require immutable 36h cap and separate retrieval reserve >=600s')
    if not budget.get('origin') or not budget['cap_started'] <= now < budget['deadline']:
        raise ValueError('Missing explicit origin or cap expired/not started')
    if budget['max_usd'] != 20 or budget['additional_updates'] != ADDITIONAL:
        raise ValueError('Require authorized $20 /104000 additional contract')
    # The parent's provider-side guard owns monetary enforcement; never invent a rate.
    if 'hourly_rate_usd' in budget and (budget['hourly_rate_usd'] <= 0 or
            36*budget['hourly_rate_usd']+budget.get('fixed_cost_reserve_usd', 0) > 20):
        raise ValueError('Recorded rate does not fit the monetary cap')


def measured_plan(source, rows, evaluations, budget, now=None):
    now = time.time() if now is None else now
    cells = set(cloud.original.all_cells())
    samples = {key: [] for key in cells}
    for row in rows:
        key = row['task'], row['cell']
        if key not in cells or row['episodes'] != 32 or not math.isfinite(row['seconds']) or row['seconds'] <= 0:
            raise ValueError('Invalid saved production timing')
        samples[key].append(row['seconds'])
    if any(not values for values in samples.values()):
        raise ValueError('Require saved production timings for every task/cell')
    costs = {key: sum(v)/len(v) for key,v in samples.items()}
    eval_samples = {key: [] for key in cells}
    for result in evaluations:
        if not result['complete'] or len(result['cells']) != 35 or {(r['task'],r['cell']) for r in result['cells']} != cells:
            raise ValueError('Require completed35-cell historical evaluation timings')
        for row in result['cells']:
            value = row['seconds']/row['n']
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid evaluation timing')
            eval_samples[row['task'],row['cell']].append(value)
    if any(not v for v in eval_samples.values()):
        raise ValueError('No completed evaluation timings')
    # Slower recorded per-episode cost across all supplied complete looks.
    ec = {key:max(v) for key,v in eval_samples.items()}
    val = 1.35*sum(ec[t,c]*(100 if t=='krauzlis_cued_motion' else 64) for t,c in cells)
    test = 1.35*sum(ec[t,c]*(200 if t=='krauzlis_cued_motion' else 128) for t,c in cells)
    scheduler = BalancedScheduler(0)
    scheduler.load_state_dict(source['scheduler'])
    exposure = {t:dict(updates=0, episodes=0, cells={c['id']:0 for c in spec['conditions']}) for t,spec in TASKS.items()}
    training = 0.
    for _ in range(ADDITIONAL):
        task, cell = scheduler.next()
        training += costs[task,cell]
        exposure[task]['updates'] += 1
        exposure[task]['episodes'] += 32
        exposure[task]['cells'][cell] += 32
    overhead = 10*val + 2*test + 900
    total = 1.2*training + overhead
    if total >= budget['deadline']-now:
        raise RuntimeError(f'104000 additional updates do not fit: need {total:.1f}s, remaining {budget["deadline"]-now:.1f}s; no reduction')
    step = source['state']['step']
    return dict(source_step=step, additional_target=ADDITIONAL, max_steps=step+ADDITIONAL,
                validation_steps=[step+i for i in range(VALIDATION_EVERY, ADDITIONAL+1, VALIDATION_EVERY)],
                validation_additional_steps=list(range(VALIDATION_EVERY, ADDITIONAL+1, VALIDATION_EVERY)),
                planned_exposure=exposure, total_additional_episodes=ADDITIONAL*32,
                estimated_optimizer_seconds=training, estimated_total_remaining_seconds=total,
                optimizer_margin=1.2, estimated_validation_seconds=val, estimated_one_test_seconds=test,
                slower_150pct_seconds=1.5*training+overhead, final_reserve_seconds=2*test+180,
                update_estimate_seconds=1.2*max(costs.values()),
                production_cell_seconds=[dict(task=t,cell=c,seconds=costs[t,c],samples=len(samples[t,c])) for t,c in sorted(cells)],
                checkpoint_every=CHECKPOINT_EVERY, checkpoint_policy='immutable cadence260 plus first1/13/validation/terminal; delete nothing',
                method='All saved production rows: cell means; exact inherited scheduler104000; optimizer1.20; max historical eval seconds/episode1.35; 10val+2test+900s; separate retrieval reserve; no profile/no reduction')


def rng_state(device):
    result = dict(cpu=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if str(device).startswith('cuda'):
        result['cuda'] = torch.cuda.get_rng_state_all()
    return cpu_tree(result)


def restore_rng(saved, device):
    torch.set_rng_state(saved['cpu'])
    np.random.set_state(saved['numpy'])
    random.setstate(saved['python'])
    if str(device).startswith('cuda'):
        if len(saved.get('cuda', [])) != torch.cuda.device_count():
            raise ValueError('CUDA RNG topology mismatch; no reseeding allowed')
        torch.cuda.set_rng_state_all(saved['cuda'])


def provenance(config, source):
    return dict(protocol=PROTOCOL, initialization='unchanged-architecture continuation',
                architecture=fresh.VERSION, source_checkpoint=source,
                inherited_weights=True, inherited_optimizer=True, inherited_rng=True,
                inherited_stream=True, inherited_scheduler=True, profile_state_inherited=False,
                train_namespace=fresh.TRAIN_NAMESPACE, validation_namespace=fresh.VAL_NAMESPACE,
                final_namespace=config.get('final_namespace', FINAL_NAMESPACE),
                cap_started=config['cap_started'], budget_origin=config.get('origin'))


class Session(fresh.Session):
    def __init__(self, directory, config, payload, receipt):
        super().__init__(directory, config)
        self.source_receipt = copy.deepcopy(receipt)
        self.source_step = payload['state']['step']
        self.source_seconds = payload['state']['optimizer_seconds']
        if payload['schema'] != 3 or payload['optimizer_names'] != [n for n, _ in self.model.named_parameters()]:
            raise ValueError('Not the identical schema3 architecture/named optimizer')
        self.model.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.scheduler.load_state_dict(payload['scheduler'])
        self.stream.load_state_dict(payload['stream'])
        self.state = cpu_tree(payload['state'])
        # The inherited constructor and native stream constructors may consume RNG.
        # Restore it LAST, and verify before making any state/config amendments.
        restore_rng(payload['rng'], self.device)
        self._verify_exact(payload)

    @property
    def additional_step(self):
        return self.state['step'] - self.source_step

    def _verify_exact(self, payload):
        observed = dict(model=cpu_tree(self.model.state_dict()), optimizer=cpu_tree(self.optimizer.state_dict()),
                        scheduler=self.scheduler.state_dict(), stream=self.stream.state_dict(),
                        state=self.state, rng=rng_state(self.device))
        for key, value in observed.items():
            expected = payload[key]
            if key == 'rng' and not str(self.device).startswith('cuda'):
                expected = {k: v for k, v in expected.items() if k != 'cuda'}
            if not tree_equal(value, expected):
                raise ValueError('Exact restore failed: '+key)

    def verify_resume(self, payload, receipt):
        self._verify_exact(payload)
        scheduler = BalancedScheduler(0)
        scheduler.load_state_dict(payload['scheduler'])
        reference_stream = fresh.FreshStream('train')
        reference_stream.load_state_dict(payload['stream'])
        try:
            expected_pair = scheduler.next()
            observed_pair = self.scheduler.next()
            if expected_pair != observed_pair:
                raise ValueError('Next scheduler draw mismatch')
            restore_rng(payload['rng'], self.device)
            expected = reference_stream.batch(self.config['microbatch'], *expected_pair)
            expected_rng = rng_state(self.device)
            restore_rng(payload['rng'], self.device)
            observed = self.stream.batch(self.config['microbatch'], *observed_pair)
            if not tree_equal(expected, observed) or not tree_equal(expected_rng, rng_state(self.device)):
                raise ValueError('Next native draw/RNG mismatch')
            result = dict(verified=True, source=receipt, source_step=self.source_step,
                          optimizer_named_state_equal=True, model_equal=True, state_equal=True,
                          scheduler_queues_equal=True, stream_equal=True, rng_equal=True,
                          cuda_rng_restored=str(self.device).startswith('cuda'),
                          next_scheduler_equal=True, next_native_draw_equal=True,
                          next_task=expected_pair[0], next_cell=expected_pair[1],
                          probe_episodes=self.config['microbatch'], probe_committed=False,
                          cap_started=self.config['cap_started'], origin=self.config.get('origin'))
        finally:
            self.scheduler.load_state_dict(payload['scheduler'])
            self.stream.load_state_dict(payload['stream'])
            restore_rng(payload['rng'], self.device)
        self._verify_exact(payload)
        atomic_json(self.directory/'resume_integrity.json', result)
        return result

    def checkpoint(self, filename):
        self.state['continuation'] = dict(source_step=self.source_step, additional_step=self.additional_step,
                                          additional_target=ADDITIONAL, cumulative_target=self.source_step+ADDITIONAL,
                                          additional_optimizer_seconds=self.state['optimizer_seconds']-self.source_seconds,
                                          cap_started=self.config['cap_started'])
        checkpoint = cloud.clone(fresh.Session.checkpoint,
                                 provenance=lambda: provenance(self.config, self.source_receipt),
                                 verify_progress=lambda _: None)
        receipt = checkpoint(self, filename)
        if self.additional_step in (1, 13) or filename == 'terminal.pt':
            verify_progress(self.directory, receipt)
        return receipt

    def train_update(self, task, cell):
        def append(path, row):
            row.update(initialization='unchanged-architecture continuation',
                       source_step=self.source_step, cumulative_step=row['step'],
                       additional_step=row['step']-self.source_step, additional_target=ADDITIONAL,
                       additional_optimizer_seconds=row['optimizer_seconds']-self.source_seconds)
            append_jsonl(path, row)
        return cloud.clone(fresh.Session.train_update, append_jsonl=append)(self, task, cell)

    def status(self, phase, **extra):
        atomic_json(self.directory/'live_status.json', dict(
            phase=phase, pid=os.getpid(), utc=cloud.original.utc(), protocol=PROTOCOL,
            architecture=fresh.VERSION, initialization='unchanged-architecture continuation',
            source_step=self.source_step, step=self.state['step'], cumulative_step=self.state['step'],
            additional_step=self.additional_step, additional_target=ADDITIONAL,
            episodes=self.state['episodes'], optimizer_seconds=self.state['optimizer_seconds'],
            additional_optimizer_seconds=self.state['optimizer_seconds']-self.source_seconds,
            deadline=self.config['deadline'], latest_checkpoint=self.latest,
            best_step=self.state['best_step'], **extra))


def validation_due(step, config):
    # Scheduled-only; no early-cap extra look and no inherited-history-length gate.
    return step in config['validation_steps']


def verify_progress(directory, receipt=None):
    directory = Path(directory)
    parent_receipt = json.loads((directory/'source_checkpoint.json').read_text())
    source = cloud.load_verified(parent_receipt)
    receipt = receipt or json.loads((directory/'latest_checkpoint.json').read_text())
    saved = cloud.load_verified(receipt)
    old, state = source['state'], saved['state']
    added = state['step']-old['step']
    if added < 0 or saved['schema'] != 3 or receipt['step'] != state['step']:
        raise ValueError('Invalid persisted continuation cursor')
    if saved['optimizer_names'] != source['optimizer_names'] or saved['scheduler']['updates'] != state['step']:
        raise ValueError('Persisted optimizer names/scheduler mismatch')
    if state['episodes'] != old['episodes']+32*added or state['optimizer_seconds'] < old['optimizer_seconds']:
        raise ValueError('Exposure or optimizer clock not preserved')
    if state['selection_history'][:len(old['selection_history'])] != old['selection_history']:
        raise ValueError('Parent selection history changed')
    if saved['optimizer']['param_groups'] != source['optimizer']['param_groups']:
        raise ValueError('Adam recipe/order changed')
    changed = []
    named_steps = {}
    ids = source['optimizer']['param_groups'][0]['params']
    for name, index in zip(saved['optimizer_names'], ids):
        before = source['optimizer']['state'].get(index)
        after = saved['optimizer']['state'].get(index)
        task = name.split('.')[1] if name.startswith('heads.') else None
        delta = state['exposure'][task]['updates']-old['exposure'][task]['updates'] if task else added
        expected = (float(before['step']) if before else 0) + delta
        if after is None or float(after['step']) != expected:
            raise ValueError('Persisted named Adam advancement mismatch: '+name)
        if not all(torch.isfinite(v).all() for v in after.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite named Adam: '+name)
        if not torch.equal(source['model'][name], saved['model'][name]): changed.append(name)
        if delta == 0 and (not tree_equal(before,after) or name in changed):
            raise ValueError('Inactive parameter/state changed: '+name)
        named_steps[name] = dict(source=float(before['step']) if before else 0, saved=float(after['step']), added=delta)
    if added:
        for prefix in ('blocks.','acc.','spatial_input.','spatial_gru.','readout.','heads.'):
            if not any(n.startswith(prefix) for n in changed):
                raise ValueError('No saved learned progress in '+prefix)
        if state['optimizer_seconds'] <= old['optimizer_seconds'] or tree_equal(source['stream'],saved['stream']):
            raise ValueError('Persisted optimizer time/native stream did not advance')
    result = dict(verified=True, source=parent_receipt, checkpoint=receipt,
                  source_step=old['step'], cumulative_step=state['step'], additional_step=added,
                  optimizer_seconds=state['optimizer_seconds'], changed_parameters=len(changed), named_adam=named_steps)
    atomic_json(directory/'persisted_progress_verification.json', result)
    if added in (1,13): atomic_json(directory/f'persisted_progress_{added:06d}.json', result)
    return result


def read_source(receipt):
    # This version is authorized for exactly this stopped fresh run, not another model.
    if (receipt['sha256'] != '7b87e8f21e4a972cbb78f8d657786e38c2aab29bf313329167575f9f564d5ff1'
            or receipt['step'] != 7739 or receipt['bytes'] != 16742235):
        raise ValueError('Not the authorized terminal7739 receipt')
    source = cloud.load_verified(receipt)
    if (source['schema'] != 3 or source['provenance'] != fresh.provenance()
            or source['state']['step'] != receipt['step'] or source['scheduler']['updates'] != receipt['step']
            or source['state']['episodes'] != 32*receipt['step'] or len(source['rng'].get('cuda', [])) != 1):
        raise ValueError('Wrong source lineage, exposure, schema or CUDA topology')
    parent = Path(receipt['path']).parent
    report = json.loads((parent/'report.json').read_text())
    if report['stop_reason'] != 'signal' or report['terminal_checkpoint']['sha256'] != receipt['sha256']:
        raise ValueError('Missing matching update-boundary signal-stop report')
    best = source['state']['best_checkpoint']
    selected = None
    if best is not None:
        records = [json.loads(line) for line in (parent/'checkpoints.jsonl').read_text().splitlines() if line.strip()]
        matches = [r for r in records if r['path'] == best and r['step'] == source['state']['best_step']]
        if len(matches) != 1: raise ValueError('Missing/ambiguous carried-best verified receipt')
        selected = matches[0]
        winner = cloud.load_verified(selected)
        if winner['provenance'] != fresh.provenance() or winner['state']['step'] != source['state']['best_step']:
            raise ValueError('Wrong carried best lineage')
    return source, selected


def config_for(source, receipt, selected, budget):
    config = copy.deepcopy(source['state']['config'])
    cloud.original.verify_sources(config)
    expected = dict(effective_batch=32, microbatch=4, eval_microbatch=4, all_trainable=True,
                    lr=1e-4, clipping=None, bptt='full', precision='fp32', tf32=False)
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Original scientific recipe differs')
    for key in ('target_requested','total_episodes','updates_per_task','exposure_reduced'):
        config.pop(key,None)
    config.update(budget)
    config.update(protocol=PROTOCOL, device='cuda', checkpoint_every=CHECKPOINT_EVERY,
                  final_namespace=FINAL_NAMESPACE, source_checkpoint=receipt, parent_best_receipt=selected,
                  worker_lock=str(WORKER_LOCK), initialization=provenance(budget, receipt),
                  baseline='carried validation winner; unchanged objective', cpu_threads=2)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.md'): config['source_hashes'][str(path.resolve())] = cloud.digest(path)
    return config


def required_disk_bytes(receipt):
    # 400 periodic +10 validation + first1/13, initial, parent, best, terminal,
    # completion copies, plus growth/JSON/temporary-file headroom. Delete nothing.
    return math.ceil(receipt['bytes']*(ADDITIONAL//CHECKPOINT_EVERY+20)*1.08)+512*1024**2


def copy_verified(receipt, destination):
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    cloud.load_verified(receipt)
    shutil.copy2(receipt['path'], destination)
    copied = dict(receipt, path=str(destination.resolve()))
    cloud.load_verified(copied)
    return copied


def run(directory):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    budget = json.loads((directory/'budget.json').read_text())
    validate_budget(budget, config)
    cloud.original.verify_sources(config)
    if config['additional_target'] != ADDITIONAL or config['max_steps'] != 7739+ADDITIONAL:
        raise ValueError('Pinned allocation changed')
    if (directory/'initial.pt').exists(): raise RuntimeError('One-shot continuation; no automatic restart')
    source_receipt = json.loads((directory/'source_checkpoint.json').read_text())
    source = cloud.load_verified(source_receipt)
    session = Session(directory, config, source, source_receipt)
    session.verify_resume(source, source_receipt)
    session.state['source_config'] = copy.deepcopy(session.state['config'])
    session.state['config'] = config
    session.checkpoint('initial.pt')
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    failure = None
    reason = 'planned_additional_updates_complete'
    terminal_test = selected_test = terminal = None
    timing_rows = []
    def validate():
        session.status('validation')
        result = evaluate(session.model, 'val', 64, 100, 4, session.device,
                          directory/f"validation_{session.state['step']:06d}.json",
                          config['deadline']-config['final_reserve_seconds'])
        key = fresh.selection_key(result)
        s = session.state
        s['selection_history'].append(dict(step=s['step'], additional_step=session.additional_step,
                                           key=key, complete=result['complete']))
        name = f"validation_checkpoint_{s['step']:06d}.pt"
        if key is not None and (s['best_key'] is None or tuple(key)>tuple(s['best_key'])):
            s.update(best_key=key, best_step=s['step'], best_checkpoint=str(directory/name))
        session.checkpoint(name)
        if key is None: raise RuntimeError('Incomplete scheduled35-cell validation')
    try:
        while session.additional_step < ADDITIONAL:
            if stopped[0]: reason = 'signal'; break
            if time.time()+config['final_reserve_seconds']+config['update_estimate_seconds'] >= config['deadline']:
                reason = 'wall_budget_reserve'; break
            row = session.train_update(*session.scheduler.next())
            if session.additional_step <= 13: timing_rows.append(row)
            added = session.additional_step
            if added in (1,13) or added % CHECKPOINT_EVERY == 0:
                session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
            if added == 13:
                ec = {(r['task'],r['cell']):r['seconds'] for r in config['production_cell_seconds']}
                observed = sum(r['seconds'] for r in timing_rows)
                expected = sum(ec[r['task'],r['cell']] for r in timing_rows)
                atomic_json(directory/'first_cycle_timing.json', dict(additional_updates=13,
                    matched_saved_production_ratio=observed/expected, margin=1.2,
                    material_slowdown=observed > 1.2*expected, retained_optimizer_progress=True))
            session.status('training', last_update={k:row[k] for k in ('task','cell','loss','seconds')})
            if validation_due(session.state['step'], config) and not stopped[0]: validate()
        terminal = session.checkpoint('terminal.pt')
        if not stopped[0] and session.additional_step > 0 and time.time() < config['deadline']-60:
            session.status('final_test_terminal')
            terminal_test = evaluate(session.model, 'test', 128, 200, 4, session.device,
                                    directory/'test_terminal.json', config['deadline']-config['estimated_one_test_seconds']-60)
            best = session.state['best_checkpoint']
            if best is not None and time.time() < config['deadline']-45:
                if best == source['state']['best_checkpoint']:
                    best_receipt = json.loads((directory/'parent_best_checkpoint.json').read_text())
                else:
                    records = [json.loads(x) for x in (directory/'checkpoints.jsonl').read_text().splitlines()]
                    best_receipt = next(r for r in records if r['path'] == best)
                winner = cloud.load_verified(best_receipt)
                if session.state['best_step'] == session.state['step']:
                    if not tree_equal(winner['model'],cpu_tree(session.model.state_dict())):
                        raise RuntimeError('Same-step model mismatch')
                    selected_test = dict(reused_terminal=True, identical_model_verified=True, results=terminal_test)
                    atomic_json(directory/'test_selected.json', selected_test)
                else:
                    session.model.load_state_dict(winner['model'])
                    session.status('final_test_selected')
                    selected_test = evaluate(session.model, 'test', 128, 200, 4, session.device,
                                             directory/'test_selected.json', config['deadline']-45)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        reason = 'worker_error'
        atomic_json(directory/'failure.json', failure)
        # Never checkpoint a partially mutated optimizer or a final-test selected model.
        if terminal is None: terminal = session.latest
    selected_result = selected_test.get('results',selected_test) if selected_test else None
    def complete35(result):
        return bool(result and result['complete'] and len(result['cells']) == 35
                    and {(r['task'],r['cell']) for r in result['cells']} == set(cloud.original.all_cells())
                    and all(r['n'] == (200 if r['task']=='krauzlis_cued_motion' else 128) for r in result['cells']))
    complete = complete35(terminal_test) and complete35(selected_result)
    saved = cloud.load_verified(terminal)
    state = saved['state']
    report = dict(protocol=PROTOCOL, architecture=fresh.VERSION, initialization=provenance(config,source_receipt),
                  stop_reason=reason, failure=failure, terminal_checkpoint=terminal,
                  source_step=session.source_step, terminal_step=state['step'], cumulative_step=state['step'],
                  additional_step=state['step']-session.source_step, additional_target=ADDITIONAL,
                  selected_step=state['best_step'], selection_history=state['selection_history'],
                  episodes=state['episodes'], exposure=state['exposure'], optimizer_seconds=state['optimizer_seconds'],
                  additional_optimizer_seconds=state['optimizer_seconds']-session.source_seconds,
                  terminal_test=terminal_test, selected_test=selected_test, final_coverage_complete=complete,
                  config=config, finished_utc=cloud.original.utc())
    atomic_json(directory/'report.json', report)
    verify_progress(directory, terminal)
    session.status('finished', stop_reason=reason, failure=failure, final_coverage_complete=complete)
    return 0 if not failure and complete and report['additional_step'] == ADDITIONAL else 2


def completion(directory, status, error=None):
    directory = Path(directory).resolve()
    # Retain the existing artifact/mirror contract, correcting fresh-only paths/counters.
    def publication_load(receipt):
        saved = cloud.load_verified(receipt)
        cp = directory/'config.json'
        if cp.exists():
            config = json.loads(cp.read_text())
            parent = config.get('parent_best_receipt')
            if parent and saved['state']['best_checkpoint'] == parent['path']:
                copied = json.loads((directory/'parent_best_checkpoint.json').read_text())
                cloud.load_verified(copied)
                saved['state']['best_checkpoint'] = copied['path']
        return saved
    cloud.clone(fresh.completion, TARGET=ADDITIONAL, load_verified=publication_load)(directory, status, error)
    path = directory/'cloud_completion.json'
    value = json.loads(path.read_text())
    report_path = directory/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    value.update(protocol=PROTOCOL, source_step=7739, target_additional_updates=ADDITIONAL,
                 target_cumulative_updates=7739+ADDITIONAL, pinned_updates=ADDITIONAL,
                 actual_updates=report.get('additional_step',0),
                 actual_additional_updates=report.get('additional_step',0),
                 actual_cumulative_updates=report.get('terminal_step',7739))
    atomic_json(path,value)


def supervisor(directory, budget, receipt):
    validate_budget(budget,budget)
    directory.mkdir(parents=True,exist_ok=True)
    # Exclusive activation prevents a duplicate supervisor as well as duplicate workers.
    with (directory/'activation.json').open('x') as f:
        json.dump(dict(supervisor_pid=os.getpid(), source=receipt, **budget),f,indent=2)
        f.flush(); os.fsync(f.fileno())
    atomic_json(directory/'budget.json',budget)
    outcomes = []
    try:
        source, selected = read_source(receipt)
        parent = Path(receipt['path']).parent
        rows = [json.loads(x) for x in (parent/'progress.jsonl').read_text().splitlines() if x.strip()]
        rows = [r for r in rows if r['step'] <= receipt['step']]
        if [r['step'] for r in rows] != list(range(1,receipt['step']+1)):
            raise ValueError('Production timings must cover every persisted source update once')
        evaluation_paths = sorted(parent.glob('validation_*.json'))
        evaluation_paths += [p for p in (parent/'test_terminal.json',parent/'test_selected.json') if p.exists()]
        evaluations = []
        evidence = []
        for path in evaluation_paths:
            result = json.loads(path.read_text())
            if result.get('reused_terminal'): continue
            if result.get('complete'):
                if result.get('namespace') == FINAL_NAMESPACE: raise ValueError('Final namespace already used')
                evaluations.append(result)
                evidence.append(dict(path=str(path),sha256=cloud.digest(path)))
        plan = measured_plan(source, rows, evaluations, budget)
        plan['timing_evidence'] = dict(progress=dict(path=str(parent/'progress.jsonl'), sha256=cloud.digest(parent/'progress.jsonl')), evaluations=evidence)
        required = required_disk_bytes(receipt)
        free = shutil.disk_usage(directory).free
        if free < required: raise RuntimeError(f'Insufficient free disk: {free} < {required}; delete nothing')
        plan['disk_preflight'] = dict(free_bytes=free, required_new_bytes=required, no_deletion=True,
                                      quota_check_owner='parent: filesystem free may exceed provider volume quota')
        atomic_json(directory/'allocation.json',plan)
        config = config_for(source, receipt, selected, budget)
        config.update(plan)
        copied = copy_verified(receipt, directory/'parent_terminal.pt')
        atomic_json(directory/'source_checkpoint.json', copied)
        if selected:
            best_copy = copy_verified(selected, directory/'parent_best.pt')
            atomic_json(directory/'parent_best_checkpoint.json',best_copy)
        atomic_json(directory/'config.json',config)
        os.chmod(directory/'allocation.json',0o444)
        os.chmod(directory/'config.json',0o444)
        os.chdir(fresh.ROOT)
        command = [sys.executable,'-u','-m','SecondPass.SpatialReadout.ContinuationRun.worker','run',str(directory)]
        outcome = cloud.supervise(command,budget['deadline'],directory,prefix='production')
        outcomes.append(outcome)
        result = dict(status='complete' if outcome['returncode']==0 else 'incomplete', outcomes=outcomes, **budget)
    except Exception as exc:
        result = dict(status='error',error=repr(exc),traceback=traceback.format_exc(),outcomes=outcomes,**budget)
    atomic_json(directory/'supervisor_result.json',result)
    completion(directory,result['status'],result.get('error'))
    return result


def main():
    cloud.original.cpu_setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['supervise','run','verify'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--budget', type=Path)
    parser.add_argument('--source', '--source-receipt', dest='source', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == 'verify':
        print(json.dumps(verify_progress(directory),indent=2)); return
    if args.mode == 'supervise':
        if args.budget is None or args.source is None: parser.error('--budget and --source required')
        result = supervisor(directory,json.loads(args.budget.read_text()),json.loads(args.source.read_text()))
        print(json.dumps(result),flush=True)
        raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require exactly one CUDA GPU; CPU is tests only')
    with WORKER_LOCK.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(run(directory))


if __name__ == '__main__':
    main()
