"""Fresh, fully trainable weighted-image transformer change classifier."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
from SecondPass.PixelContrastiveVAE.train import (GPULock, atomic_json, append_json,
    cpu_tree, metrics, replace_with_retry, configure, synchronize)
from . import dataset
from .model import WeightedSumTransformer

POOL_SIZE, POOL_EPOCHS, POOL_UPDATES = 1000, 2, 64
BATCH_SIZE = 32
INIT_SEED, SHUFFLE_SEED = 191211, 191212


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob('*.py')) + [root / path for path in (
        'SecondPass/PixelContrastiveVAE/train.py',
        'SecondPass/VariationalMotionPredictor/dataset.py',
        'SecondPass/PredictiveConsistencyVAE/queue.py')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build(device):
    random.seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(INIT_SEED)
    model = WeightedSumTransformer().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8, weight_decay=0.)


def remaining(deadline, reserve=30):
    if time.time() >= deadline - reserve:
        raise TimeoutError('Finite four-hour local cap reached; no automatic extension')


def update(model, optimizer, rows, indices, device, microbatch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for group in indices.split(microbatch):
        chosen = [rows[int(i)] for i in group]
        first, second = [torch.stack([row[k] for row in chosen]).to(device) for k in (0, 1)]
        labels = torch.tensor([row[2] for row in chosen], dtype=torch.float32, device=device)
        logits = model(first, second).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        fraction = len(group) / len(indices)
        (loss * fraction).backward()
        loss_sum += float(loss.detach()) * fraction
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    gradients = [p.grad for p in model.parameters()]
    if missing or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Missing/nonfinite gradients: ' + ', '.join(missing))
    norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    if not bool(torch.isfinite(norm)) or not math.isfinite(loss_sum):
        raise FloatingPointError('Nonfinite gradient norm or loss')
    optimizer.step()
    return dict(loss=loss_sum, gradient_norm=float(norm))


@torch.no_grad()
def evaluate(model, split, count, device, microbatch, deadline, threshold=None):
    if split == 'test' and threshold is None:
        raise ValueError('Test requires a saved validation threshold')
    model.eval()
    scores, labels, metadata = [], [], []
    offset = 12000000 if split == 'val' else 36000000
    for start in range(0, count, microbatch):
        remaining(deadline)
        rows = [dataset.generate_evaluation(offset + i, split) for i in range(start, min(start + microbatch, count))]
        first, second = [torch.stack([row[k] for row in rows]).to(device) for k in (0, 1)]
        logits = model(first, second).reshape(-1).cpu()
        scores.extend(logits.tolist())
        labels.extend(int(row[2]) for row in rows)
        metadata.extend({k: v for k, v in row[3].items() if k != 'positions_unwrapped_xy'} for row in rows)
    scores, labels = np.asarray(scores), np.asarray(labels)
    summary = metrics(scores, labels, threshold, 1)
    zero = metrics(scores, labels, 0., 1)
    loss = float(F.binary_cross_entropy_with_logits(torch.tensor(scores), torch.tensor(labels, dtype=torch.float64)))
    cells = []
    for cell in range(6):
        indices = [i for i, m in enumerate(metadata) if m['cell'] == cell]
        cells.append(dict(cell=cell, speed=metadata[indices[0]]['speed'], angle=metadata[indices[0]]['angle'],
            n=len(indices), **metrics(scores[indices], labels[indices], summary['threshold'], 1),
            zero_logit_balanced_accuracy=metrics(scores[indices], labels[indices], 0., 1)['balanced_accuracy']))
    return dict(n=count, paired_contexts=len({m['context'] for m in metadata}), loss=loss,
        **summary, zero_logit_balanced_accuracy=zero['balanced_accuracy'], cells=cells,
        predictions=[dict(label=int(y), logit=float(score), metadata=m) for y, score, m in zip(labels, scores, metadata)])


def save(directory, model, optimizer, state, config, shuffle, device, best=False):
    payload = cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), state=state,
        config=config, shuffle_rng=shuffle.get_state(), torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None))
    for name in ('latest', 'best') if best else ('latest',):
        temporary = directory / (name + '.pt.tmp')
        torch.save(payload, temporary)
        replace_with_retry(temporary, directory / (name + '.pt'))


def first_step_proof(run, model, initial):
    saved = torch.load(run / 'latest.pt', map_location='cpu')
    steps = [float(item['step']) for item in saved['optimizer']['state'].values()]
    expected = len(list(model.parameters()))
    finite_weights = all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
    finite_adam = all(bool(torch.isfinite(v).all()) for item in saved['optimizer']['state'].values()
        for v in item.values() if torch.is_tensor(v))
    if len(steps) != expected or set(steps) != {1.} or not finite_weights or not finite_adam:
        raise RuntimeError('CPU-reloaded first-step all-parameter Adam/finite proof failed')
    atomic_json(run / 'production_verified.json', dict(step=1, cpu_reloaded=True,
        optimizer_states=len(steps), expected_active_states=expected, adam_steps=steps,
        all_weights_finite=finite_weights, all_adam_tensors_finite=finite_adam,
        changed_tensors=[k for k, v in saved['model'].items() if not torch.equal(v, initial[k])],
        fresh_constructor=True, checkpoint_inputs=[]))


def rng_hashes(shuffle, device):
    tensor_hash = lambda value: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
    return dict(torch_rng=tensor_hash(torch.get_rng_state()),
        shuffle_rng=tensor_hash(shuffle.get_state()),
        numpy_rng=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        python_rng=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        cuda_rng=[tensor_hash(value) for value in torch.cuda.get_rng_state_all()] if device == 'cuda' else None)


def restore_checkpoint(saved, model, optimizer, shuffle, device):
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    steps = [float(item['step']) for item in optimizer.state.values()]
    state = cpu_tree(saved['state'])
    if len(steps) != len(list(model.parameters())) or set(steps) != {float(state['step'])}:
        raise RuntimeError('Resume requires aligned Adam state for every learned parameter')
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise FloatingPointError('Nonfinite restored weights')
    if not all(bool(torch.isfinite(v).all()) for item in optimizer.state.values()
            for v in item.values() if torch.is_tensor(v)):
        raise FloatingPointError('Nonfinite restored Adam state')
    for group in optimizer.param_groups:
        if group['lr'] != 3e-4 or group['eps'] != 1e-8 or group['weight_decay'] != 0. or tuple(group['betas']) != (.9, .999):
            raise ValueError('Saved optimizer differs from original scientific settings')
    shuffle.set_state(saved['shuffle_rng'])
    torch.set_rng_state(saved['torch_rng'])
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    if device == 'cuda':
        if saved['cuda_rng'] is None or len(saved['cuda_rng']) != torch.cuda.device_count():
            raise RuntimeError('Saved CUDA RNG/device count unavailable or changed')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    order, cursor, epoch = state['order'], state['cursor'], state['epoch']
    if sorted(order) != list(range(POOL_SIZE)) or not 0 <= cursor <= POOL_SIZE or not 0 <= epoch < POOL_EPOCHS:
        raise ValueError('Saved task-local permutation/cursor invalid')
    if cursor != POOL_SIZE and cursor % BATCH_SIZE:
        raise ValueError('Saved cursor is not a completed minibatch boundary')
    updates_per_epoch = math.ceil(POOL_SIZE / BATCH_SIZE)
    expected_step = state['pool_index'] * POOL_UPDATES + epoch * updates_per_epoch + math.ceil(cursor / BATCH_SIZE)
    expected_presentations = state['pool_index'] * POOL_SIZE * POOL_EPOCHS + epoch * POOL_SIZE + cursor
    if state['step'] != expected_step or state['presentations'] != expected_presentations or state['pool_start'] != state['pool_index'] * POOL_SIZE:
        raise ValueError('Saved exposure and sampler position are inconsistent')
    return state


def training_batches(state, shuffle, target):
    """Continue the checkpoint's unused permutation before drawing another.

    A checkpoint at the last tail has cursor==POOL_SIZE. The exhausted epoch
    (and, when applicable, pool) advances without repeating any example.
    """
    restore_order = 'order' in state
    pool_index = state.get('pool_index', 0) if restore_order else 0
    while state['step'] < target:
        start = pool_index * POOL_SIZE
        pool = [dataset.generate(i, 'train') for i in range(start, start + POOL_SIZE)]
        state.update(pool_index=pool_index, pool_start=start, unique_pairs=start + POOL_SIZE)
        first_epoch = state['epoch'] if restore_order else 0
        for epoch in range(first_epoch, POOL_EPOCHS):
            if restore_order:
                order = torch.tensor(state['order'], dtype=torch.long)
                cursor = state['cursor']
                restore_order = False
            else:
                order = torch.randperm(POOL_SIZE, generator=shuffle)
                cursor = 0
                state.update(epoch=epoch, order=order.tolist(), cursor=0)
            for indices in order[cursor:].split(BATCH_SIZE):
                if not len(indices):
                    continue
                yield pool, indices
                if state['step'] >= target:
                    return
        pool_index += 1


def profile(args, run):
    if (run / 'budget.json').exists() or args.profile_json.exists():
        raise RuntimeError('Existing budget/profile; refusing implicit renewal')
    started = time.time()
    deadline = started + args.wall_seconds
    atomic_json(run / 'budget.json', dict(started=started, deadline=deadline,
        wall_seconds=args.wall_seconds, worker_pid=os.getpid(), phase='profile'))
    clock = time.time()
    rows = [dataset.generate(i, 'train') for i in range(POOL_SIZE)]
    generation = time.time() - clock
    model, optimizer = build(args.device)
    update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch)
    synchronize(args.device)
    timings = []
    for _ in range(args.profile_steps):
        clock = time.time()
        update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch)
        synchronize(args.device)
        timings.append(time.time() - clock)
    clock = time.time()
    evaluate(model, 'val', 12, args.device, args.microbatch, deadline)
    synchronize(args.device)
    evaluation = (time.time() - clock) / 12
    reserve = 600. + 1.25 * evaluation * (7 * args.val_count + args.test_count)
    cost = 1.25 * (float(np.median(timings)) + generation / POOL_UPDATES)
    feasible = int(max(0., deadline - time.time() - reserve) / cost) // POOL_UPDATES * POOL_UPDATES
    target = min(args.target_updates, feasible)
    if target < POOL_UPDATES:
        raise RuntimeError('Cannot fit one complete pool plus evaluation in finite cap')
    atomic_json(args.profile_json, dict(started=started, deadline=deadline, target_updates=target,
        requested_target_updates=args.target_updates, update_seconds=float(np.median(timings)),
        pool_generation_seconds=generation, evaluation_seconds_per_pair=evaluation,
        reserve_seconds=reserve, safety_factor=1.25, profile_weights_discarded=True,
        source_hashes=source_hashes(), device=args.device, microbatch=args.microbatch,
        val_count=args.val_count, test_count=args.test_count, wall_seconds=args.wall_seconds,
        torch_version=torch.__version__, speed_scale=dataset.SPEED_SCALE,
        speeds=[speed * dataset.SPEED_SCALE for speed in dataset.SPEEDS]))


def production(args, run):
    is_resume = getattr(args, 'resume', False)
    if (run / 'production_config.json').exists() and not is_resume:
        raise RuntimeError('Existing production; explicit recovery required')
    pinned = json.loads(args.profile_json.read_text(encoding='utf-8'))
    if not is_resume and pinned['source_hashes'] != source_hashes():
        raise RuntimeError('Source changed after profile')
    for key in ('device', 'microbatch', 'val_count', 'test_count'):
        if pinned[key] != getattr(args, key):
            raise ValueError('Profile/production mismatch: ' + key)
    if pinned.get('speed_scale', 1.) != dataset.SPEED_SCALE:
        raise ValueError('Profile/production mismatch: speed_scale')
    if float(pinned['deadline']) - float(pinned['started']) != 14400.:
        raise ValueError('One finite four-hour allocation must cover entire run')
    target, deadline = int(pinned['target_updates']), float(pinned['deadline'])
    recovery = None
    if is_resume:
        if (run / 'completion_verified.json').exists():
            raise RuntimeError('Completed run cannot be resumed automatically')
        recovery = json.loads(args.recovery_budget.read_text(encoding='utf-8'))
        if float(recovery['deadline']) - float(recovery['started']) != 3600. or recovery.get('wall_seconds') != 3600:
            raise ValueError('Explicit recovery must use one finite3600-second allowance')
        if not recovery.get('authorization') or recovery.get('source_hashes') != source_hashes():
            raise ValueError('Recovery authorization/current-source manifest missing or changed')
        # Only the runtime adapter changes. Every original non-adapter source
        # must remain byte-identical to the scientific production snapshot.
        for path, digest in pinned['source_hashes'].items():
            if Path(path).name == 'train.py' and 'WeightedSumTransformer' in path:
                continue
            if source_hashes().get(path) != digest:
                raise ValueError('Original scientific source changed: ' + path)
        deadline = float(recovery['deadline'])
    if target <= 0 or target % POOL_UPDATES:
        raise ValueError('Need prospectively pinned complete pools')
    remaining(deadline, 120)
    model, optimizer = build(args.device)
    initial = cpu_tree(model.state_dict()) if not is_resume else None
    shuffle = torch.Generator().manual_seed(SHUFFLE_SEED)
    config = dict(fresh_whole_model=True, checkpoint_inputs=[], init_seed=INIT_SEED,
        shuffle_seed=SHUFFLE_SEED, target_updates=target, batch_pairs=BATCH_SIZE, microbatch_pairs=args.microbatch,
        pool_size=POOL_SIZE, pool_epochs=POOL_EPOCHS, optimizer='Adam', lr=3e-4, epsilon=1e-8,
        clipping=None, weight_decay=0., precision='fp32', tf32=False, cpu_threads=2,
        weighted_input='(oldest +2*middle +3*latest)/6, single RGB image per three-frame clip',
        architecture='shared one-layer transformer, patch10/d128/heads4/feedforward512/GELU/dropout0/preLN/CLS; ordered CLS concatenation then FFN',
        loss='binary cross entropy with logits only', all_learned_parameters_trainable=True,
        selection='minimum validation BCE', decision='positive-sign validation BA-optimal scalar threshold',
        source_hashes=source_hashes(), dataset_namespaces=dataset.NAMESPACES, dataset_version=dataset.VERSION,
        speed_scale=dataset.SPEED_SCALE, speeds=[speed * dataset.SPEED_SCALE for speed in dataset.SPEEDS],
        device=args.device, pid=os.getpid(), started=pinned['started'], deadline=deadline,
        val_count=args.val_count, test_count=args.test_count)
    if is_resume:
        config = json.loads((run / 'production_config.json').read_text(encoding='utf-8'))
        saved = torch.load(run / 'latest.pt', map_location='cpu')
        if saved['config'] != config or config['source_hashes'] != pinned['source_hashes']:
            raise ValueError('Checkpoint and original production provenance differ')
        if config['pool_size'] != POOL_SIZE or config['pool_epochs'] != POOL_EPOCHS or config['batch_pairs'] != BATCH_SIZE:
            raise ValueError('Sampler implementation differs from original configuration')
        state = restore_checkpoint(saved, model, optimizer, shuffle, args.device)
        if not 0 < state['step'] < target:
            raise ValueError('Resume checkpoint is outside remaining cumulative exposure')
        if state['best_loss'] is not None:
            best = torch.load(run / 'best.pt', map_location='cpu')
            if best['state']['step'] != state['best_step'] or best['state']['threshold'] != state['threshold']:
                raise ValueError('Retained best checkpoint/validation decision inconsistent')
            del best
        checkpoint_hash = hashlib.sha256((run / 'latest.pt').read_bytes()).hexdigest()
        saved_rng = dict(torch_rng=hashlib.sha256(saved['torch_rng'].numpy().tobytes()).hexdigest(),
            shuffle_rng=hashlib.sha256(saved['shuffle_rng'].numpy().tobytes()).hexdigest(),
            numpy_rng=hashlib.sha256(pickle.dumps(saved['numpy_rng'])).hexdigest(),
            python_rng=hashlib.sha256(pickle.dumps(saved['python_rng'])).hexdigest(),
            cuda_rng=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in saved['cuda_rng']] if args.device == 'cuda' else None)
        restored_rng = rng_hashes(shuffle, args.device)
        if restored_rng != saved_rng:
            raise RuntimeError('Exact restored RNG proof failed')
        previous_live = json.loads((run / 'live_status.json').read_text(encoding='utf-8')) if (run / 'live_status.json').exists() else {}
        atomic_json(args.recovery_budget.parent / 'resume_receipt.json', dict(checkpoint_sha256=checkpoint_hash,
            restored_step=state['step'], previous_live_step=previous_live.get('step'),
            rollback_updates=max(0, previous_live.get('step', state['step']) - state['step']),
            retained_target=target, original_config_unchanged=True, exact_rng_restored=True,
            rng_hashes=restored_rng, restored_sampler={k: state[k] for k in ('pool_index', 'pool_start', 'epoch', 'cursor')},
            order_sha256=hashlib.sha256(np.asarray(state['order'], dtype=np.int64).tobytes()).hexdigest(),
            original_deadline=pinned['deadline'], recovery_deadline=deadline, pid=os.getpid(), utc=utc()))
        append_json(run / 'progress.jsonl', dict(event='explicit_resume', utc=utc(),
            restored_step=state['step'], previous_live_step=previous_live.get('step'),
            rollback_updates=max(0, previous_live.get('step', state['step']) - state['step']),
            recovery_budget=str(args.recovery_budget)))
        del saved
    else:
        atomic_json(run / 'production_config.json', config)
        state = dict(step=0, presentations=0, unique_pairs=0, best_step=None, best_loss=None, threshold=None)
    resume_start_step = state['step']
    selected_loaded = False

    def status(phase, **values):
        atomic_json(run / 'live_status.json', dict(phase=phase, pid=os.getpid(), utc=utc(),
            target_updates=target, recovery_attempt=str(args.recovery_budget.parent) if is_resume else None,
            **{k: v for k, v in state.items() if k != 'order'}, **values))

    def validate():
        status('validation')
        result = evaluate(model, 'val', args.val_count, args.device, args.microbatch, deadline)
        result['step'] = state['step']
        atomic_json(run / ('validation_%06d.json' % state['step']), result)
        better = state['best_loss'] is None or result['loss'] < state['best_loss']
        if better:
            state.update(best_step=state['step'], best_loss=result['loss'], threshold=result['threshold'])
        save(run, model, optimizer, state, config, shuffle, args.device, better)

    try:
        validation_steps = {64, 5000, 10000, 15000, 20000, target}
        for pool, indices in training_batches(state, shuffle, target):
            remaining(deadline, 120)
            clock = time.time()
            values = update(model, optimizer, pool, indices, args.device, args.microbatch)
            state['step'] += 1
            state['presentations'] += len(indices)
            state['cursor'] += len(indices)
            compact = {k: v for k, v in state.items() if k != 'order'}
            append_json(run / 'progress.jsonl', dict(event='update', utc=utc(), **compact,
                recovery_attempt=str(args.recovery_budget.parent) if is_resume else None,
                **values, update_seconds=time.time() - clock))
            status('training', **values)
            if state['step'] == 1:
                save(run, model, optimizer, state, config, shuffle, args.device)
                first_step_proof(run, model, initial)
                del initial
            elif is_resume and state['step'] == resume_start_step + 1:
                save(run, model, optimizer, state, config, shuffle, args.device)
                advanced = torch.load(run / 'latest.pt', map_location='cpu')
                adam_steps = [float(item['step']) for item in advanced['optimizer']['state'].values()]
                if len(adam_steps) != len(list(model.parameters())) or set(adam_steps) != {float(state['step'])}:
                    raise RuntimeError('Persisted resumed Adam advancement proof failed')
                atomic_json(args.recovery_budget.parent / 'production_verified.json', dict(
                    cpu_reloaded=True, restored_step=resume_start_step, saved_step=state['step'],
                    all_active_adam_states=len(adam_steps), adam_steps=adam_steps,
                    cumulative_target_unchanged=target, pid=os.getpid(), utc=utc()))
                del advanced
            elif state['step'] % 250 == 0:
                save(run, model, optimizer, state, config, shuffle, args.device)
            if state['step'] in validation_steps:
                validate()
        save(run, model, optimizer, state, config, shuffle, args.device)
        selected = torch.load(run / 'best.pt', map_location='cpu')
        model.load_state_dict(selected['model'])
        selected_loaded = True
        selected_step, threshold = selected['state']['step'], selected['state']['threshold']
        del selected
        status('final_test')
        final = evaluate(model, 'test', args.test_count, args.device, args.microbatch, deadline, threshold)
        final.update(selected_step=selected_step, test_thresholds_fitted=False)
        atomic_json(run / 'final_test.json', final)
        (run / 'FINAL_REPORT.md').write_text('\n'.join(['# Weighted-sum transformer change classifier', '',
            f"Completed {state['step']} updates / {state['presentations']} pair presentations / {state['unique_pairs']} unique movies; selected {selected_step} by validation BCE.", '',
            f"Held-out BA {final['balanced_accuracy']:.6f}; AUC {final['auc']:.6f}; BCE {final['loss']:.6f}.",
            f"Zero-logit-threshold BA {final['zero_logit_balanced_accuracy']:.6f}.",
            f"Test: {final['n']} presentations / {final['paired_contexts']} paired contexts; all six cells in final_test.json.",
            'One shared single-layer transformer encodes each ordered 1:2:3 weighted RGB sum; ordered CLS vectors concatenate into a nonlinear classifier.',
            'Every learned parameter starts fresh and trains jointly on BCE. No VAE, angular objective, contrastive objective, predictive target or pretrained transfer.',
            'Simplified continuous dots, not native Krauzlis or biological validation.', '']), encoding='utf-8')
        receipt = dict(status='complete', stop_reason='planned_complete', completed=utc(),
            elapsed_seconds=time.time() - pinned['started'], updates=state['step'],
            elapsed_seconds_kind='calendar wall time including any shutdown',
            calendar_elapsed_seconds=time.time() - pinned['started'],
            recovery_elapsed_seconds=time.time() - recovery['started'] if is_resume else None,
            presentations=state['presentations'], unique_pairs=state['unique_pairs'],
            selected_step=selected_step, final_n=final['n'], paired_contexts=final['paired_contexts'], no_cloud=True)
        atomic_json(run / 'completion_verified.json', receipt)
        atomic_json(run / 'live_status.json', dict(phase='complete', pid=os.getpid(), **receipt))
    except BaseException as exc:
        if state['step'] and not selected_loaded:
            save(run, model, optimizer, state, config, shuffle, args.device)
        atomic_json(run / 'failure.json', dict(type=type(exc).__name__, message=str(exc),
            state={k: v for k, v in state.items() if k != 'order'}, utc=utc()))
        raise


def main(args):
    dataset.configure_speed_scale(args.speed_scale)
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = GPULock(args.lock_path)
    try:
        configure(args.device)
        if args.profile_only:
            profile(args, run)
        else:
            production(args, run)
    finally:
        lock.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile-json', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recovery-budget', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--speed-scale', type=float, default=1.)
    parser.add_argument('--wall-seconds', type=int, default=14400)
    parser.add_argument('--target-updates', type=int, default=25024)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--val-count', type=int, default=384)
    parser.add_argument('--test-count', type=int, default=3072)
    parser.add_argument('--lock-path', type=Path, default=Path(__file__).resolve().parents[2] / 'LocalRuntime/local_gpu_worker.lock')
    args = parser.parse_args()
    if args.resume and (args.profile_only or args.recovery_budget is None):
        parser.error('Resume requires explicit recovery budget and cannot profile or start fresh')
    if args.recovery_budget is not None and not args.resume:
        parser.error('Recovery budget is only valid with --resume')
    if not math.isfinite(args.speed_scale) or args.speed_scale <= 0:
        parser.error('Speed scale must be finite and positive')
    if args.profile_json is None:
        args.profile_json = args.run / 'profile.json'
    if args.wall_seconds != 14400:
        parser.error('Authorized finite allocation is exactly14400seconds for all work together')
    if args.target_updates % 64 or args.val_count % 12 or args.test_count % 12 or min(args.target_updates, args.microbatch, args.val_count, args.test_count, args.profile_steps) < 1:
        parser.error('Need positive counts, complete64-update pools and balanced12-example evaluation cells')
    return args


if __name__ == '__main__':
    main(parse_args())
