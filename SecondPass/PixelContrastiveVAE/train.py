"""Fresh, bounded local VAE and pixel-distance VAE comparison.

Importing this module never launches work. Both arms use the identical fresh
initialization, stateless stimulus indices, replay pools and shuffled orders.
Only reconstruction, KL and pixel-derived distances enter the training loss.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from . import dataset
from .model import PixelContrastiveVAE, losses

INIT_SEED = 187211
SHUFFLE_SEED = 187212
POOL_SIZE = 1000
POOL_EPOCHS = 2
POOL_UPDATES = 64


def replace_with_retry(source, destination, retries=8):
    """Windows readers/AV can briefly deny replacement of an open file."""
    for attempt in range(retries + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == retries:
                raise
            time.sleep(min(.025 * 2 ** attempt, .4))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    for attempt in range(9):
        try:
            with tmp.open('w', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            return
        except OSError as exc:
            transient = getattr(exc, 'winerror', None) in (5, 32, 33)
            if transient and attempt < 8:
                time.sleep(min(.025 * 2 ** attempt, .4))
                continue
            if path.name != 'live_status.json':
                raise
            # Telemetry is dispensable; checkpoint/protocol/result writes are not.
            print(f'WARNING skipped live-status write after bounded retries: {exc}', file=sys.stderr, flush=True)
            return


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


class GPULock:
    """OS-held exclusive lock; process exit releases it without deleting files."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0)
        if self.file.read(1) == b'':
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.file.close()
            raise RuntimeError('Another local GPU worker owns the experiment lock')

    def close(self):
        self.file.close()


def configure(device):
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')


def seed_fresh(device):
    random.seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(INIT_SEED)


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py')}


def build(device):
    seed_fresh(device)
    model = PixelContrastiveVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
    return model, optimizer


def batches(rows, indices, device, microbatch):
    for chunk in indices.split(microbatch):
        selected = [rows[int(i)] for i in chunk]
        first = torch.stack([r[0] for r in selected]).to(device)
        second = torch.stack([r[1] for r in selected]).to(device)
        yield first, second, len(selected) / len(indices)


def update(model, optimizer, rows, indices, device, microbatch, step, pair_weight):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = {}
    for first, second, fraction in batches(rows, indices, device, microbatch):
        a, b = model(first), model(second)
        result = losses(a, b, first, second, beta=1e-4 * min(1, step / 250), pair_weight=pair_weight)
        (result['loss'] * fraction).backward()
        for key in ('loss', 'recon', 'kl', 'pair_loss', 'full_mse'):
            values[key] = values.get(key, 0.0) + float(result[key].detach()) * fraction
    missing = [name for name, p in model.named_parameters() if p.grad is None
               and not (pair_weight == 0 and name.startswith('projection.'))]
    if missing:
        raise FloatingPointError('Unexpected missing gradients: ' + ', '.join(missing))
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not math.isfinite(values['loss']):
        raise FloatingPointError('Missing gradients or nonfinite loss')
    # One aggregate device synchronization rather than one per parameter.
    finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
    norm_squared = torch.stack([g.detach().square().sum() for g in gradients]).sum()
    if not bool(finite) or not bool(torch.isfinite(norm_squared)):
        raise FloatingPointError('Nonfinite training gradient')
    values['gradient_norm'] = float(norm_squared.sqrt())
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError('Nonfinite updated parameter')
    return values


def metrics(scores, labels, threshold=None, sign=None):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(scores).all() or not (labels == 0).any() or not (labels == 1).any():
        raise ValueError('Distance metrics require finite scores and both labels')
    raw_scores = scores.copy()
    differences = scores[labels == 1, None] - scores[labels == 0][None, :]
    raw_auc = float(((differences > 0) + .5 * (differences == 0)).mean())
    if sign is None:
        # Sign is learned from validation only. Held-out evaluation receives it.
        sign = 1 if raw_auc >= .5 else -1
    if sign not in (-1, 1):
        raise ValueError('Distance decision sign must be -1 or +1')
    scores = scores * sign
    if threshold is None:
        unique = np.unique(scores)
        candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)],
                                     .5 * (unique[:-1] + unique[1:]), [unique[-1]]))
        ba = .5 * ((scores[labels == 1, None] > candidates).mean(0)
                   + (scores[labels == 0, None] <= candidates).mean(0))
        threshold = float(candidates[int(np.argmax(ba))])
    predictions = scores > threshold
    # Exact pairwise rank AUC with tie handling; small evaluation batches only.
    differences = scores[labels == 1, None] - scores[labels == 0][None, :]
    auc = float(((differences > 0) + .5 * (differences == 0)).mean())
    return dict(threshold=float(threshold), sign=int(sign), raw_auc=raw_auc, auc=auc,
                balanced_accuracy=float(.5 * (predictions[labels == 1].mean()
                                             + (~predictions[labels == 0]).mean())),
                same_mean=float(raw_scores[labels == 0].mean()),
                change_mean=float(raw_scores[labels == 1].mean()))


@torch.no_grad()
def evaluate(model, split, count, device, microbatch, deadline, pair_weight, thresholds=None):
    if split != 'val' and thresholds is None:
        raise ValueError('Held-out tests require validation-only thresholds')
    model.eval()
    score_lists = {k: [] for k in ('posterior', 'projection', 'pixels')}
    labels, metadata, means, projections = [], [], [], []
    total = {k: 0. for k in ('loss', 'recon', 'kl', 'pair_loss', 'full_mse')}
    predictions = []
    for start in range(0, count, microbatch):
        if time.time() >= deadline:
            raise TimeoutError('Evaluation deadline reached')
        namespace_offset = 12000000 if split == 'val' else 36000000
        rows = [dataset.generate_evaluation(namespace_offset + i, split)
                for i in range(start, min(start + microbatch, count))]
        first = torch.stack([r[0] for r in rows]).to(device)
        second = torch.stack([r[1] for r in rows]).to(device)
        a, b = model(first), model(second)
        result = losses(a, b, first, second, beta=1e-4, pair_weight=pair_weight)
        for key in total:
            total[key] += float(result[key]) * len(rows) / count
        mu_a, mu_b = a['mu'].flatten(1), b['mu'].flatten(1)
        posterior = 1 - F.cosine_similarity(mu_a, mu_b, dim=1, eps=1e-8)
        projection = 1 - F.cosine_similarity(a['embedding'], b['embedding'], dim=1, eps=1e-8)
        # The pair loss's raw-pixel target is preserved unchanged for reporting.
        pixels = result['pixel_distance'].reshape(-1)
        score_batch = dict(posterior=posterior.cpu().tolist(), projection=projection.cpu().tolist(),
                           pixels=pixels.cpu().tolist())
        means.append(mu_a.cpu().numpy())
        projections.append(a['embedding'].cpu().numpy())
        for key in score_lists:
            score_lists[key].extend(score_batch[key])
        for j, row in enumerate(rows):
            labels.append(int(row[2]))
            compact_metadata = {k: v for k, v in row[3].items() if k != 'positions_unwrapped_xy'}
            metadata.append(compact_metadata)
            predictions.append(dict(label=int(row[2]), metadata=compact_metadata,
                                    **{key: float(score_batch[key][j]) for key in score_batch}))
    metrics_all = {}
    for key, scores in score_lists.items():
        calibration = None if thresholds is None else thresholds[key]
        summary = metrics(scores, labels, None if calibration is None else calibration['threshold'],
                          None if calibration is None else calibration['sign'])
        cells = []
        for speed in (.375, 1., 2.):
            for angle in (26, 28):
                ix = [i for i, m in enumerate(metadata) if float(m['speed']) == speed and int(m['angle']) == angle]
                cell = metrics(np.asarray(scores)[ix], np.asarray(labels)[ix], summary['threshold'], summary['sign'])
                cells.append(dict(speed=speed, angle=angle, n=len(ix), **cell))
        metrics_all[key] = dict(**summary, cells=cells)
    mean_matrix = np.concatenate(means)
    projection_matrix = np.concatenate(projections)
    target = np.asarray(score_lists['pixels'])
    learned = np.asarray(score_lists['projection'])
    correlation = float(np.corrcoef(target, learned)[0, 1]) if target.std() > 0 and learned.std() > 0 else 0.
    return dict(n=count, paired_contexts=len({m['context'] for m in metadata}), **total,
                distances=metrics_all, posterior_variance=float(mean_matrix.var(axis=0).mean()),
                posterior_norm=float(np.linalg.norm(mean_matrix, axis=1).mean()),
                projection_variance=float(projection_matrix.var(axis=0).mean()),
                projection_norm=float(np.linalg.norm(projection_matrix, axis=1).mean()),
                pixel_target_projection_correlation=correlation,
                pixel_target_projection_mse=float(np.square(target - learned).mean()),
                collapsed_pair_loss_baseline=float(.5 * np.square(target).mean()), predictions=predictions)


def profile(args, run):
    if args.profile_json.exists():
        raise RuntimeError('Profile output already exists; refusing to replace the budget origin')
    start = time.time()
    deadline = start + args.wall_seconds
    atomic_json(run / 'budget.json', dict(started=start, deadline=deadline, wall_seconds=args.wall_seconds,
                                         phase='profile', worker_pid=os.getpid()))
    pool_clock = time.time()
    rows = [dataset.generate(i, 'train') for i in range(1000)]
    pool_generation_seconds = time.time() - pool_clock
    times = {}
    for name, weight in (('vae_control', 0.), ('pixel_contrastive', 1.)):
        model, optimizer = build(args.device)
        update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch, 1, weight)
        synchronize(args.device)
        timings = []
        for i in range(args.profile_steps):
            clock = time.time()
            update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch, i + 2, weight)
            synchronize(args.device)
            timings.append(time.time() - clock)
        clock = time.time()
        evaluation = evaluate(model, 'val', 12, args.device, args.microbatch, deadline, weight)
        synchronize(args.device)
        eval_seconds = time.time() - clock
        times[name] = dict(update_seconds=float(np.median(timings)), measured_updates=len(timings),
                           evaluation_seconds_per_trial=eval_seconds / 12,
                           parameters=sum(p.numel() for p in model.parameters()))
        del model, optimizer, evaluation
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    # Conservative complete equal pool exposure; profiling never supplies weights.
    reserved = sum(t['evaluation_seconds_per_trial'] * (args.val_count * 7 + args.test_count)
                   for t in times.values()) * 1.25 + 300
    available = deadline - time.time() - reserved
    seconds_per_pair_update = (sum(t['update_seconds'] for t in times.values())
                               + 2 * pool_generation_seconds / POOL_UPDATES) * 1.25
    feasible = int(max(0, available) / seconds_per_pair_update) // POOL_UPDATES * POOL_UPDATES
    target = min(args.target_updates, feasible)
    if target < POOL_UPDATES:
        raise RuntimeError('Profile cannot fit one complete pool per arm in this cap')
    proof = dict(started=start, deadline=deadline, wall_seconds=args.wall_seconds, timings=times,
                 target_updates=target, requested_target_updates=args.target_updates, reserve_seconds=reserved,
                 safety_factor=1.25, pool_generation_seconds=pool_generation_seconds,
                 profile_weights_discarded=True, source_hashes=source_hashes(),
                 device=args.device, torch_version=torch.__version__, cpu_threads=2)
    atomic_json(args.profile_json, proof)
    print(json.dumps(proof), flush=True)


def restore_checkpoint(model, optimizer, shuffle, checkpoint, device):
    """Restore every mutable training state; regenerate pixels statelessly later."""
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['torch_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])
    shuffle.set_state(checkpoint['shuffle_rng'])
    if device.startswith('cuda'):
        if checkpoint['cuda_rng'] is None:
            raise ValueError('CUDA continuation requires saved CUDA RNG')
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return copy.deepcopy(checkpoint['state'])


def continuation_epochs(state, shuffle, resumed_pool):
    """Yield exact unconsumed old order before drawing any new permutation."""
    first_epoch = state['epoch'] if resumed_pool else 0
    for epoch in range(first_epoch, POOL_EPOCHS):
        if resumed_pool and epoch == first_epoch:
            order = torch.tensor(state['order'], dtype=torch.long)
            cursor = state['cursor']
        else:
            order = torch.randperm(POOL_SIZE, generator=shuffle)
            cursor = 0
        state.update(epoch=epoch, order=order.tolist(), cursor=cursor)
        yield order[cursor:]


def train_arm(args, run, name, weight, deadline):
    arm = run / name
    resumed = args.resume and arm.exists()
    if arm.exists() and not resumed:
        raise RuntimeError('Arm already exists; refusing to overwrite any experiment')
    arm.mkdir(exist_ok=resumed)
    model, optimizer = build(args.device)
    initial = cpu_tree(model.state_dict())
    shuffle = torch.Generator().manual_seed(SHUFFLE_SEED)
    config = dict(arm=name, pair_weight=weight, fresh_whole_model=True, checkpoint_inputs=[],
                  init_seed=INIT_SEED, shuffle_seed=SHUFFLE_SEED, all_learned_weights_trainable=True,
                  optimizer='Adam', lr=1e-4, epsilon=1e-8, weight_decay=0., clipping=None,
                  batch_pairs=32, microbatch_pairs=args.microbatch, pool_size=POOL_SIZE,
                  pool_epochs=POOL_EPOCHS, target_updates=args.target_updates,
                  target_pair_presentations=args.target_updates // POOL_UPDATES * 2000,
                  target_unique_pairs=args.target_updates // POOL_UPDATES * 1000,
                  precision='fp32', tf32=False, cpu_threads=2, device=args.device,
                  beta_max=1e-4, beta_warmup_updates=250, source_hashes=source_hashes(),
                  dataset_namespaces=dataset.NAMESPACES, dataset_version=dataset.VERSION,
                  checkpoint_selection='minimum unsupervised validation loss; no labels or angles',
                  common_primary_distance='cosine dissimilarity of posterior mean vectors',
                  projection_control='fresh projection receives no gradient when pair_weight=0',
                  threshold_selection='validation-only sign from AUC then balanced-accuracy threshold; test sign fixed',
                  hard_deadline=deadline)
    state = dict(step=0, presentations=0, unique_pairs=0, pool_index=-1,
                 best_step=None, best_loss=None, thresholds=None)
    if resumed:
        checkpoint = torch.load(arm / 'latest.pt', map_location='cpu')
        old_config = checkpoint['config']
        for key in ('arm', 'pair_weight', 'init_seed', 'shuffle_seed', 'lr', 'epsilon', 'weight_decay',
                    'clipping', 'microbatch_pairs', 'target_updates', 'hard_deadline', 'dataset_namespaces', 'dataset_version'):
            if old_config[key] != config[key]:
                raise ValueError(f'Resume must preserve {key}: {old_config[key]} != {config[key]}')
        state = restore_checkpoint(model, optimizer, shuffle, checkpoint, args.device)
        config = old_config  # Preserve original scientific config/source provenance.
        expected = len(list(model.parameters())) - (2 if weight == 0 else 0)
        adam_steps = [float(value['step']) for value in checkpoint['optimizer']['state'].values()]
        if len(adam_steps) != expected or any(step != state['step'] for step in adam_steps):
            raise ValueError('Resume checkpoint Adam states do not match saved progress')
        atomic_json(arm / f"resume_verified_{state['step']:06d}.json", dict(
            checkpoint='latest.pt', checkpoint_sha256=hashlib.sha256((arm / 'latest.pt').read_bytes()).hexdigest(),
            cpu_reloaded=True, optimizer_states=len(adam_steps), adam_steps=adam_steps,
            state={k: v for k, v in state.items() if k != 'order'},
            permutation_sha256=hashlib.sha256(np.asarray(state['order'], dtype=np.int64).tobytes()).hexdigest(),
            all_rng_restored=True, source_hashes_original=config['source_hashes'], source_hashes_current=source_hashes(),
            migration='file-write resilience and explicit exact-progress resume only', hard_deadline=deadline))
        del checkpoint, initial
    else:
        atomic_json(arm / 'config.json', config)
    validation_steps = {64, 5000, 10000, 15000, 20000, 25024, args.target_updates}
    validated = {int(p.stem.split('_')[-1]) for p in arm.glob('validation_*.json')}
    selected_weights_loaded = False

    def save(best=False):
        payload = cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               state=state, config=config, torch_rng=torch.get_rng_state(),
                               numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                               shuffle_rng=shuffle.get_state(),
                               cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None))
        tmp = arm / 'latest.pt.tmp'
        torch.save(payload, tmp)
        replace_with_retry(tmp, arm / 'latest.pt')
        if best:
            tmp = arm / 'best.pt.tmp'
            torch.save(payload, tmp)
            replace_with_retry(tmp, arm / 'best.pt')
        atomic_json(arm / 'checkpoint_receipt.json', dict(step=state['step'], optimizer_states=len(optimizer.state)))

    def validate():
        atomic_json(run / 'live_status.json', dict(phase='validation', arm=name, pid=os.getpid(), **state))
        val = evaluate(model, 'val', args.val_count, args.device, args.microbatch, deadline - 30, weight)
        val['step'] = state['step']
        atomic_json(arm / f"validation_{state['step']:06d}.json", val)
        better = state['best_loss'] is None or val['loss'] < state['best_loss']
        if better:
            state.update(best_step=state['step'], best_loss=val['loss'],
                         thresholds={key: dict(threshold=item['threshold'], sign=item['sign'])
                                     for key, item in val['distances'].items()})
        validated.add(state['step'])
        save(best=better)

    try:
        resumed_pool = resumed and state['pool_index'] >= 0
        while state['step'] < args.target_updates:
            if time.time() >= deadline - 120:
                raise TimeoutError('Shared wall cap reached before pinned exposure completed')
            if not resumed_pool:
                state['pool_index'] += 1
            pool_start = state['pool_index'] * POOL_SIZE
            clock = time.time()
            pool = [dataset.generate(i, 'train') for i in range(pool_start, pool_start + POOL_SIZE)]
            state.update(unique_pairs=pool_start + POOL_SIZE, pool_start=pool_start)
            append_json(arm / 'progress.jsonl', dict(event='pool_generated', pool=state['pool_index'],
                                                   seconds=time.time() - clock, unique_pairs=state['unique_pairs']))
            for remaining_order in continuation_epochs(state, shuffle, resumed_pool):
                for indices in remaining_order.split(32):
                    if time.time() >= deadline - 120:
                        raise TimeoutError('Shared wall cap reached during training')
                    clock = time.time()
                    values = update(model, optimizer, pool, indices, args.device, args.microbatch,
                                    state['step'] + 1, weight)
                    state['step'] += 1
                    state['presentations'] += len(indices)
                    state['cursor'] += len(indices)
                    compact_state = {key: value for key, value in state.items() if key != 'order'}
                    append_json(arm / 'progress.jsonl', dict(event='update', **compact_state, **values,
                                                           update_seconds=time.time() - clock))
                    atomic_json(run / 'live_status.json', dict(phase='training', arm=name, pid=os.getpid(),
                                                               utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                                               **compact_state, **values))
                    if state['step'] == 1:
                        save()
                        saved = torch.load(arm / 'latest.pt', map_location='cpu')
                        changed = [key for key, value in saved['model'].items()
                                   if torch.is_tensor(value) and not torch.equal(value, initial[key])]
                        proof = dict(step=1, parameter_tensors=len(list(model.parameters())),
                                     optimizer_states=len(saved['optimizer']['state']), changed_tensors=changed,
                                     adam_steps=[float(v['step']) for v in saved['optimizer']['state'].values()],
                                     fresh_constructor=True, checkpoint_inputs=[], cpu_reloaded=True)
                        expected_states = len(list(model.parameters())) - (2 if weight == 0 else 0)
                        if len(saved['optimizer']['state']) != expected_states or any(v != 1 for v in proof['adam_steps']):
                            raise RuntimeError('Saved first-step Adam proof does not match active parameters')
                        proof['expected_active_adam_states'] = expected_states
                        atomic_json(arm / 'production_verified.json', proof)
                        del saved, initial
                    elif state['step'] % 250 == 0:
                        save()
                    if state['step'] in validation_steps:
                        validate()
            del pool
            resumed_pool = False
        if state['step'] not in validated:
            validate()
        save()
        selected = torch.load(arm / 'best.pt', map_location='cpu')
        model.load_state_dict(selected['model'])
        selected_weights_loaded = True
        selected_step = selected['state']['step']
        thresholds = selected['state']['thresholds']
        del selected
        atomic_json(run / 'live_status.json', dict(phase='final_test', arm=name, pid=os.getpid(), **state))
        final = evaluate(model, 'test', args.test_count, args.device, args.microbatch, deadline - 10, weight, thresholds)
        final.update(selected_step=selected_step, test_thresholds_fitted=False)
        atomic_json(arm / 'final_test.json', final)
        lines = [f'# {name}: completed local pixel VAE comparison', '',
                 f"Trained {state['step']} fresh updates, {state['presentations']} pair presentations, "
                 f"{state['unique_pairs']} unique movies. Unsupervised validation selected update {selected_step}.", '',
                 f"Held-out test: {final['n']} presentations from {final['paired_contexts']} paired contexts. "
                 'Sign and threshold were selected on validation and fixed for test.', '',
                 '| Distance | Raw AUC | Validation sign | Calibrated AUC | Balanced accuracy |',
                 '|---|---:|---:|---:|---:|']
        for key, item in final['distances'].items():
            lines.append(f"| {key} | {item['raw_auc']:.6f} | {item['sign']:+d} | {item['auc']:.6f} | {item['balanced_accuracy']:.6f} |")
        lines += ['', 'Per-speed and angle results, reconstruction, KL, latent variation, raw pixel targets '
                  'and every test score are preserved in final_test.json. All trained targets are pixel-derived; '
                  'direction/angle/label metadata never enters backpropagation. This is the simplified continuous '
                  'movie, not native Krauzlis transfer.', '']
        (arm / 'FINAL_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        receipt = dict(status='complete', stop_reason='planned_complete', arm=name, **state,
                       selected_step=selected_step, final_n=final['n'], paired_contexts=final['paired_contexts'],
                       completed=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(arm / 'completion_verified.json', receipt)
        return receipt
    except BaseException as exc:
        if state['step'] and not selected_weights_loaded:
            save()
        atomic_json(arm / 'failure.json', dict(type=type(exc).__name__, message=str(exc), **state))
        raise
    finally:
        del model, optimizer
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()


def main(args):
    configure(args.device)
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = GPULock(args.lock_path)
    try:
        if args.profile_only:
            profile(args, run)
            return
        if not args.resume and ((run / 'production_config.json').exists() or any((run / name).exists() for name in ('vae_control', 'pixel_contrastive'))):
            raise RuntimeError('Existing production artifacts; no implicit restart or overwrite')
        profile_record = json.loads(args.profile_json.read_text(encoding='utf-8'))
        if args.target_updates > profile_record['target_updates'] or args.target_updates % POOL_UPDATES:
            raise ValueError('Target must be a feasible equal number of complete 64-update pools')
        if not args.resume and profile_record['source_hashes'] != source_hashes():
            raise RuntimeError('Source changed after profile; repeat explicit profile within original cap')
        deadline = float(profile_record['deadline'])
        if time.time() >= deadline - 120:
            raise TimeoutError('Original experiment cap already exhausted')
        if args.resume:
            original = json.loads((run / 'production_config.json').read_text(encoding='utf-8'))
            for key, current in (('target_updates_per_arm', args.target_updates), ('deadline', deadline),
                                 ('val_count', args.val_count), ('test_count', args.test_count)):
                if original[key] != current:
                    raise ValueError(f'Resume must preserve original {key}')
            current_hashes = source_hashes()
            for name, original_hash in original['source_hashes'].items():
                if name != 'train.py' and current_hashes.get(name) != original_hash:
                    raise RuntimeError(f'Unrelated scientific/source change cannot migrate implicitly: {name}')
            atomic_json(run / f'resume_launch_{os.getpid()}.json', dict(pid=os.getpid(), deadline=deadline,
                        target_updates_per_arm=args.target_updates, original_source_hashes=original['source_hashes'],
                        current_source_hashes=current_hashes, original_budget_started=profile_record['started'],
                        migration='bounded Windows file write retries, nonfatal telemetry, exact-progress resume',
                        new_profile=False, new_budget=False, checkpoint_inputs=['pixel_contrastive/latest.pt']))
        else:
            atomic_json(run / 'production_config.json', dict(pid=os.getpid(), target_updates_per_arm=args.target_updates,
                    deadline=deadline, budget_started=profile_record['started'], profile_weights_discarded=True,
                    sequential_arms=['pixel_contrastive', 'vae_control'], source_hashes=source_hashes(),
                    cuda_device=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                    torch_version=torch.__version__, val_count=args.val_count, test_count=args.test_count))
        receipts = []
        for name, weight in (('pixel_contrastive', 1.), ('vae_control', 0.)):
            completion = run / name / 'completion_verified.json'
            if args.resume and completion.exists():
                receipts.append(json.loads(completion.read_text(encoding='utf-8')))
            else:
                receipts.append(train_arm(args, run, name, weight, deadline))
        atomic_json(run / 'completion_verified.json', dict(status='complete', arms=receipts, deadline=deadline,
                    elapsed_seconds=time.time() - profile_record['started'], no_cloud=True))
        atomic_json(run / 'live_status.json', dict(phase='complete', pid=os.getpid(), arms=receipts))
    finally:
        lock.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--target-updates', type=int, default=25024)
    parser.add_argument('--wall-seconds', type=int, default=28800)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--val-count', type=int, default=384)
    parser.add_argument('--test-count', type=int, default=1536)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Explicitly restore existing progress without a new profile/budget')
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--profile-json', type=Path, required=True)
    parser.add_argument('--lock-path', type=Path, default=Path(__file__).resolve().parents[2] / 'LocalRuntime' / 'local_gpu_worker.lock')
    args = parser.parse_args()
    if args.val_count % 12 or args.test_count % 12:
        parser.error('Validation/test counts must preserve the complete six-cell balanced pairs (multiples of 12)')
    if args.resume and args.profile_only:
        parser.error('Resume never profiles or starts a new allowance')
    return args


if __name__ == '__main__':
    main(parse_args())
