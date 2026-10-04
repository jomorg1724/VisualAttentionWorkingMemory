"""Single-worker fp32 full-BPTT primitives. No architecture or stimulus changes."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from SecondPass.TaskSuite.suite import CATALOG, TASKS, SuiteStream, task_classes
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline
from .core import (BalancedScheduler, append_jsonl, atomic_json, may_update,
                   save_checkpoint, score_cell, summarize)


def cpu_setup():
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 2: torch.set_num_interop_threads(2)


def sync(device):
    if str(device) == 'mps': torch.mps.synchronize()


def fresh_model(seed, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if str(device) == 'mps': torch.mps.manual_seed(seed)
    model = AccumulatorBaseline(task_classes(), stack=3, center=True, accumulator='kda').to(device)
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)
    assert len(optimizer.param_groups) == 1 and not optimizer.state
    return model, optimizer


def parameter_group(name, task):
    if name.startswith(('blocks.','feat.','proj.')): return 'sensory'
    if name.startswith('acc.'): return 'kda'
    if name.startswith('gru.'): return 'gru'
    if name.startswith('heads.'+task+'.'): return 'active_head'
    return 'inactive_heads'


def update(model, optimizer, stream, task, cell, effective, microbatch, device, diagnostics=False):
    assert effective % microbatch == 0
    model.train(); optimizer.zero_grad(set_to_none=True)
    before = {name:p.detach().cpu().clone() for name,p in model.named_parameters()} if diagnostics else {}
    sync(device); started = time.perf_counter(); loss_sum = 0.; frames = 0
    for _ in range(effective // microbatch):
        images, labels, _ = stream.batch(microbatch, task, cell)
        frames += images.shape[0] * images.shape[1]
        logits = model(images.to(device), task)
        loss = F.cross_entropy(logits, labels.to(device))
        if not bool(torch.isfinite(loss).item()): raise FloatingPointError('nonfinite loss')
        (loss * (microbatch/effective)).backward()
        loss_sum += float(loss.detach().cpu()) * microbatch/effective
        del images, labels, logits, loss
    # Never clip. Check all active gradients BEFORE Adam mutates any parameter.
    group_norms = {}
    for name,p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach()
        if not bool(torch.isfinite(g).all().item()): raise FloatingPointError('nonfinite gradient: '+name)
        group = parameter_group(name, task)
        group_norms[group] = group_norms.get(group, 0.) + float(g.square().sum().cpu())
    norm = math.sqrt(sum(group_norms.values()))
    if not math.isfinite(norm): raise FloatingPointError('nonfinite gradient norm')
    optimizer.step(); sync(device)
    for name,p in model.named_parameters():
        if not bool(torch.isfinite(p).all().item()): raise FloatingPointError('nonfinite parameter after Adam: '+name)
    elapsed = time.perf_counter()-started
    out = dict(task=task, cell=cell, loss=loss_sum, grad_norm=norm, episodes=effective,
               frames=int(frames), seconds=elapsed, episodes_per_second=effective/elapsed)
    if diagnostics:
        groups = {g:dict(grad_norm=math.sqrt(group_norms.get(g, 0.)), update_norm=0., trainable_parameters=0)
                  for g in ('sensory','kda','gru','active_head','inactive_heads')}
        for name,p in model.named_parameters():
            group = parameter_group(name, task)
            groups[group]['trainable_parameters'] += p.numel()
            groups[group]['update_norm'] += float((p.detach().cpu()-before[name]).square().sum())
        for v in groups.values(): v['update_norm'] = math.sqrt(v['update_norm'])
        out['diagnostics'] = groups
    return out


def all_cells():
    return [(t,c['id']) for t,spec in TASKS.items() for c in spec['conditions']]


@torch.no_grad()
def evaluate(model, split, n, krauzlis_n, microbatch, device, output, cells=None, deadline=float('inf')):
    # Construct anew each look: fixed, task/cell-local validation/test draws.
    stream = SuiteStream(split); model.eval(); started = time.perf_counter(); rows = []
    cells = all_cells() if cells is None else cells
    complete = True
    for task,cell in cells:
        count = krauzlis_n if task == 'krauzlis_cued_motion' else n
        labels = []; probabilities = []; metadata = []; cell_started = time.perf_counter()
        for i in range(0, count, microbatch):
            if time.time() >= deadline:
                complete = False; break
            images, y, meta = stream.batch(min(microbatch, count-i), task, cell)
            logits = model(images.to(device), task)
            if not bool(torch.isfinite(logits).all().item()): raise FloatingPointError('nonfinite evaluation logits')
            probabilities.extend(logits.softmax(1).cpu().numpy().tolist())
            labels.extend(y.tolist()); metadata.extend(meta)
        if not complete: break
        sync(device)
        row = score_cell(task, cell, labels, probabilities, metadata)
        row['seconds'] = time.perf_counter()-cell_started
        rows.append(row)
        atomic_json(str(output)+'.partial.json', dict(split=split, complete_cells=len(rows), expected_cells=len(cells), cells=rows))
    out = dict(split=split, n=n, krauzlis_n=krauzlis_n, cells=rows, complete=complete and len(rows)==len(cells),
               complete_cells=len(rows), expected_cells=len(cells), seconds=time.perf_counter()-started,
               summary=summarize(rows), fixed_draws=True)
    atomic_json(output, out)
    return out


def utc(timestamp=None):
    return dt.datetime.fromtimestamp(time.time() if timestamp is None else timestamp, dt.timezone.utc).isoformat()


def verify_sources(config):
    from PreAttentiveVision.natural_stimuli import DATA_ROOT, sha256
    if CATALOG != config['catalog']: raise ValueError('Frozen catalog changed')
    if sha256(Path(DATA_ROOT)/'manifest.json') != config['source_manifest_sha256']:
        raise ValueError('BSDS500 identity manifest changed')
    for path, expected in config['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen production source changed: '+path)


def run(directory, config):
    from .core import restore_checkpoint
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if (directory/'checkpoint_000000.pt').exists():
        raise RuntimeError('Refusing automatic restart or overwrite of existing production run')
    verify_sources(config)
    torch.set_num_threads(2)
    device = config['device']; model, optimizer = fresh_model(config['seed'], device)
    scheduler = BalancedScheduler(config['scheduler_seed']); stream = SuiteStream('train')
    state = dict(step=0, episodes=0, frames=0, optimizer_seconds=0., deadline=config['deadline'],
        elapsed_cap_seconds=time.time()-config['cap_started'], config=config, selection_history=[],
        best_checkpoint=None, best_key=None, best_step=None,
        exposure={t:dict(updates=0, episodes=0, frames=0, cells={c['id']:0 for c in spec['conditions']}) for t,spec in TASKS.items()})
    stopped = [False]; old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    latest = None

    def status(phase, **extra):
        atomic_json(directory/'live_status.json', dict(phase=phase, pid=os.getpid(), utc=utc(),
            step=state['step'], episodes=state['episodes'], frames=state['frames'],
            optimizer_seconds=state['optimizer_seconds'], deadline=utc(config['deadline']),
            deadline_epoch=config['deadline'], remaining_seconds=max(0., config['deadline']-time.time()),
            latest_checkpoint=latest, best_step=state['best_step'], **extra))

    def checkpoint(filename):
        nonlocal latest
        state['elapsed_cap_seconds'] = time.time()-config['cap_started']
        receipt = save_checkpoint(directory/filename, model, optimizer, scheduler, stream, state, device)
        latest = receipt
        atomic_json(directory/'latest_checkpoint.json', receipt)
        append_jsonl(directory/'checkpoints.jsonl', dict(utc=utc(), **receipt))
        return receipt

    checkpoint('checkpoint_000000.pt')
    status('training', initialization='fresh_random_all_parameters', optimizer_initial_state='empty')
    stop_reason = 'planned_complete_cycles'; failure = None; terminal_test = None; selected_test = None
    try:
        while state['step'] < config['max_steps']:
            if not may_update(now=time.time(), deadline=config['deadline'], reserve=config['final_reserve_seconds'],
                              estimate=config['update_estimate_seconds'], step=state['step'], max_steps=config['max_steps'], stopped=stopped[0]):
                stop_reason = 'signal' if stopped[0] else 'wall_budget_reserve'; break
            task,cell = scheduler.next()
            row = update(model, optimizer, stream, task, cell, config['effective_batch'], config['microbatch'], device,
                         diagnostics=state['step']==0)
            state['step'] += 1; state['episodes'] += row['episodes']; state['frames'] += row['frames']
            state['optimizer_seconds'] += row['seconds']
            exposure = state['exposure'][task]
            exposure['updates'] += 1; exposure['episodes'] += row['episodes']; exposure['frames'] += row['frames']
            exposure['cells'][cell] += row['episodes']
            row.update(utc=utc(), step=state['step'], cumulative_episodes=state['episodes'], cumulative_frames=state['frames'],
                       optimizer_seconds=state['optimizer_seconds'], elapsed_cap_seconds=time.time()-config['cap_started'],
                       per_task_exposure=state['exposure'], clipping=None, latest_validation=state['selection_history'][-1] if state['selection_history'] else None)
            append_jsonl(directory/'progress.jsonl', row)
            print(json.dumps({k:row[k] for k in ('utc','step','task','cell','loss','grad_norm','seconds','cumulative_episodes')}), flush=True)
            if state['step'] in (1,2) or state['step'] % config['checkpoint_every'] == 0:
                checkpoint(f"checkpoint_{state['step']:06d}.pt")
            status('training', last_update={k:row[k] for k in ('task','cell','loss','grad_norm','seconds')})
            if state['step'] in config['validation_steps'] and not stopped[0]:
                verify_sources(config)
                status('validation')
                validation = evaluate(model, 'val', config['val_n'], config['val_krauzlis_n'], config['eval_microbatch'], device,
                    directory/f"validation_{state['step']:06d}.json", deadline=config['deadline']-config['final_reserve_seconds'])
                key = validation['summary']['selection_key'] if validation['complete'] else None
                record = dict(step=state['step'], key=key, complete=validation['complete'])
                state['selection_history'].append(record)
                filename = f"validation_checkpoint_{state['step']:06d}.pt"
                if key is not None and (state['best_key'] is None or tuple(key)>tuple(state['best_key'])):
                    state['best_key'] = key; state['best_step'] = state['step']; state['best_checkpoint'] = str(directory/filename)
                checkpoint(filename)
        # A checkpoint independent of validation always preserves the terminal optimizer state.
        terminal = checkpoint('terminal.pt')
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        atomic_json(directory/'failure.json', failure)
        stop_reason = 'nonfinite_safe_stop' if isinstance(exc, FloatingPointError) else 'worker_error'
        # Do not publish contaminated tensors or an advanced sampler as healthy.
        if latest is None: raise
        state = restore_checkpoint(latest['path'], model, optimizer, scheduler, stream, device)
        terminal = checkpoint('terminal_recovered.pt')
    try:
        verify_sources(config)
        if time.time() < config['deadline']-60 and state['step'] > 0:
            status('final_test_terminal')
            terminal_test = evaluate(model, 'test', config['test_n'], config['test_krauzlis_n'], config['eval_microbatch'], device,
                directory/'test_terminal.json', deadline=config['deadline']-45)
            if state['best_step'] == state['step']:
                selected_test = dict(reused_terminal=True, selected_step=state['best_step'], results=terminal_test)
                atomic_json(directory/'test_selected.json', selected_test)
            elif state['best_checkpoint'] is not None and time.time() < config['deadline']-60:
                selected = torch.load(state['best_checkpoint'], map_location='cpu')
                model.load_state_dict(selected['model'])
                status('final_test_selected')
                selected_test = evaluate(model, 'test', config['test_n'], config['test_krauzlis_n'], config['eval_microbatch'], device,
                    directory/'test_selected.json', deadline=config['deadline']-45)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        atomic_json(directory/'finalization_failure.json', failure)
    report = dict(stop_reason=stop_reason, failure=failure, config=config, terminal_step=state['step'],
        selected_step=state['best_step'], selection_history=state['selection_history'], terminal_checkpoint=terminal,
        exposure=state['exposure'], episodes=state['episodes'], frames=state['frames'], optimizer_seconds=state['optimizer_seconds'],
        terminal_test=terminal_test, selected_test=selected_test, finished_utc=utc(), elapsed_cap_seconds=time.time()-config['cap_started'],
        limitations=['Single bounded fresh-weight run; no second arm or seed replication.',
          'Task identity externally chooses the supervised head; metadata never enters the model.',
          'No source-independent confidence intervals. Official BSDS500 identities shared across photo tasks within split.',
          'Delay cells are independent, not paired. Suite test seeds previously used for stimulus verification, not model selection.',
          'Profile optimizer updates discarded; production started with fresh model, empty Adam and new SuiteStream.',
          'No early-chance stopping; N0 excluded from selection; only planned validation chooses a checkpoint.'])
    atomic_json(directory/'report.json', report)
    text = f"# Bounded fresh-weight KDA joint training\n\nStop: {stop_reason}. Terminal step: {state['step']}; validation-selected step: {state['best_step']}.\n\n"
    text += f"Episodes: {state['episodes']}; frames: {state['frames']}; optimizer seconds: {state['optimizer_seconds']:.2f}.\n\n"
    text += 'Full cell confusion, recall, BA/AUC, event denominators and sampled strata: `report.json`, `test_terminal.json`, `test_selected.json`. Terminal and selected results are distinct unless the same checkpoint.\n\n'
    text += '\n'.join('- '+s for s in report['limitations'])+'\n'
    (directory/'REPORT.md').write_text(text)
    status('finished', stop_reason=stop_reason, report=str(directory/'report.json'), failure=failure)
    for sig, handler in old_handlers.items(): signal.signal(sig, handler)
    print(json.dumps(dict(finished=True, report=str(directory/'report.json'), step=state['step'], stop_reason=stop_reason)), flush=True)
    return report


def profile(directory):
    from .launch import frames
    from .core import restore_checkpoint
    directory=Path(directory); budget=json.loads((directory/'budget.json').read_text())
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; explicit bounded CPU assessment required')
    device='mps'; model,optimizer=fresh_model(94180001,device)
    stream=SuiteStream('train'); scheduler=BalancedScheduler(94180002)
    rows=[]
    for task,spec in TASKS.items():
        if time.time() > budget['cap_started']+1800:
            raise RuntimeError('Profile exceeded its 30-minute sub-budget; production not launched')
        cell=max(spec['conditions'], key=lambda c:frames(task,c))
        row=update(model,optimizer,stream,task,cell['id'],64,4,device,diagnostics=not rows)
        # Forward-only throughput uses validation, never final test, on same longest cell.
        measured=evaluate(model,'val',8,8,4,device,directory/f'profile_eval_{task}.json',
                          cells=[(task,cell['id'])],deadline=budget['deadline']-60)
        row.update(eval_seconds=measured['seconds'],eval_n=8,sequence_frames=frames(task,cell),
                   disposable=True, microbatch=4,effective_batch=64,
                   allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
        rows.append(row); append_jsonl(directory/'profile_progress.jsonl',row)
        atomic_json(directory/'profile.json',dict(rows=rows,complete=False,device=device,**budget))
        print(json.dumps(row),flush=True)
    state=dict(step=len(rows),deadline=budget['deadline'],elapsed_cap_seconds=time.time()-budget['cap_started'],
               selection_history=[],config={'disposable_profile':True,'effective_batch':64,'microbatch':4})
    receipt=save_checkpoint(directory/'disposable_profile.pt',model,optimizer,scheduler,stream,state,device)
    expected=torch.rand(16,device=device).cpu()
    restore_checkpoint(directory/'disposable_profile.pt',model,optimizer,scheduler,stream,device)
    actual=torch.rand(16,device=device).cpu()
    if not torch.equal(expected,actual): raise RuntimeError('MPS RNG checkpoint exact replay failed')
    result=dict(rows=rows,complete=True,device=device,checkpoint=receipt,mps_rng_replay=True,
                torch_version=torch.__version__,numpy_version=np.__version__,**budget)
    atomic_json(directory/'profile.json',result)
    print(json.dumps(dict(profile_complete=True,checkpoint=receipt)),flush=True)


def main():
    cpu_setup()
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['profile','run']);p.add_argument('directory')
    args=p.parse_args();directory=Path(args.directory).resolve()
    if args.mode=='profile': profile(directory)
    else: run(directory,json.loads((directory/'config.json').read_text()))


if __name__=='__main__': main()
