"""Sequential weighted-image VAE, then a frozen-latent nonlinear change decoder.

Importing this module is inert. A single prospective local deadline covers both
stages, latent extraction, validation and final held-out evaluation.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from SecondPass.PixelContrastiveVAE.train import (GPULock, atomic_json, append_json,
    cpu_tree, metrics, replace_with_retry, configure, synchronize)
from . import dataset
from .model import WeightedSumVAE, NonlinearChangeDecoder, weighted_sum

POOL_SIZE, POOL_EPOCHS, POOL_UPDATES = 1000, 2, 64
INIT_SEED, SHUFFLE_SEED, HEAD_SEED = 190211, 190212, 190221
CACHE_TRAIN_OFFSET, CACHE_VAL_OFFSET, CACHE_TEST_OFFSET = 80000000, 12000000, 36000000


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob('*.py')) + [
        root / 'SecondPass/PixelContrastiveVAE/train.py',
        root / 'SecondPass/VariationalMotionPredictor/dataset.py',
        root / 'SecondPass/PredictiveConsistencyVAE/queue.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def seed(value, device):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if device == 'cuda':
        torch.cuda.manual_seed_all(value)


def build_vae(device):
    seed(INIT_SEED, device)
    model = WeightedSumVAE().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)


def build_head(device):
    seed(HEAD_SEED, device)
    head = NonlinearChangeDecoder().to(device)
    return head, torch.optim.Adam(head.parameters(), lr=1e-3, eps=1e-8)


def remaining(deadline, reserve=30):
    if time.time() >= deadline - reserve:
        raise TimeoutError('Finite four-hour experiment cap reached; no automatic extension')


def objective(result, images, beta):
    reconstruction = F.mse_loss(result['reconstruction'], images)
    mu, logvar = result['mu'], result['logvar']
    kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
    return dict(loss=reconstruction + beta * kl, reconstruction=reconstruction, kl=kl)


def check_gradients(model):
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if missing or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Missing or nonfinite active gradients: ' + ', '.join(missing))
    norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite gradient norm')
    return float(norm)


def vae_update(model, optimizer, rows, indices, device, microbatch, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = dict(loss=0., reconstruction=0., kl=0.)
    for group in indices.split(microbatch):
        clips = torch.stack([clip for i in group for clip in rows[int(i)][:2]])
        images = weighted_sum(clips).to(device)
        local = objective(model(images), images, 1e-4 * min(1., step / 250.))
        fraction = len(group) / len(indices)
        (local['loss'] * fraction).backward()
        for key in values:
            values[key] += float(local[key].detach()) * fraction
    values['gradient_norm'] = check_gradients(model)
    if not math.isfinite(values['loss']):
        raise FloatingPointError('Nonfinite VAE loss')
    optimizer.step()
    return values


def head_update(head, optimizer, features, labels, indices, device):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    logits = head(features[indices].to(device)).reshape(-1)
    loss = F.binary_cross_entropy_with_logits(logits, labels[indices].to(device))
    loss.backward()
    gradient_norm = check_gradients(head)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite classifier loss')
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm=gradient_norm)


@torch.no_grad()
def evaluate_vae(model, split, count, device, microbatch, deadline, examples=None, mean_image=None):
    model.eval()
    totals = dict(loss=0., reconstruction=0., kl=0., gray_mse=0., target_variance=0., prediction_variance=0.,
        raw_sum_mse=0., support_mse=0., gray_support_mse=0., train_mean_mse=0.)
    offset = 60000000
    for start in range(0, count, microbatch):
        remaining(deadline)
        rows = [dataset.generate_evaluation(offset + i, split) for i in range(start, min(start + microbatch, count))]
        images = weighted_sum(torch.stack([clip for row in rows for clip in row[:2]])).to(device)
        result = model(images)
        local = objective(result, images, 1e-4)
        local['gray_mse'] = (images - .5).square().mean()
        local['raw_sum_mse'] = 36 * local['reconstruction']
        support = (images - .5).abs() > 1. / 255.
        local['support_mse'] = ((result['reconstruction'] - images).square() * support).sum() / support.sum().clamp_min(1)
        local['gray_support_mse'] = ((images - .5).square() * support).sum() / support.sum().clamp_min(1)
        baseline = .5 if mean_image is None else mean_image.to(device)
        local['train_mean_mse'] = (images - baseline).square().mean()
        local['target_variance'] = images.var(dim=(1, 2, 3), unbiased=False).mean()
        local['prediction_variance'] = result['reconstruction'].var(dim=(1, 2, 3), unbiased=False).mean()
        for key in totals:
            totals[key] += float(local[key]) * len(rows) / count
        if examples is not None and start == 0:
            np.savez_compressed(examples, target_sum=(6 * images[:8]).cpu().numpy(),
                reconstruction_sum=(6 * result['reconstruction'][:8]).cpu().numpy(),
                normalized_target=images[:8].cpu().numpy(), normalized_reconstruction=result['reconstruction'][:8].cpu().numpy())
            from PIL import Image
            actual = images[:8].permute(0, 2, 3, 1).cpu().numpy()
            predicted = result['reconstruction'][:8].permute(0, 2, 3, 1).cpu().numpy()
            canvas = np.concatenate([np.concatenate(list(actual), axis=1), np.concatenate(list(predicted), axis=1)], axis=0)
            Image.fromarray((np.clip(canvas, 0., 1.) * 255).astype(np.uint8)).resize((800, 200)).save(Path(examples).with_suffix('.png'))
    return dict(n_pairs=count, reconstruction_images=2 * count, **totals)


@torch.no_grad()
def extract(model, split, count, offset, device, microbatch, deadline, status=None):
    model.eval()
    features, labels, metadata = [], [], []
    for start in range(0, count, microbatch):
        remaining(deadline)
        generator = dataset.generate_classifier if split == 'train' else dataset.generate_evaluation
        rows = [generator(offset + i, split) for i in range(start, min(start + microbatch, count))]
        images = weighted_sum(torch.stack([clip for row in rows for clip in row[:2]])).to(device)
        mu = model.encode(images)[0].reshape(len(rows), 2, -1)
        features.append(torch.cat((mu[:, 0], mu[:, 1]), dim=1).cpu())
        labels.extend(int(row[2]) for row in rows)
        metadata.extend({key: value for key, value in row[3].items() if key != 'positions_unwrapped_xy'} for row in rows)
        if status is not None and start % (microbatch * 128) == 0:
            atomic_json(status, dict(phase='stage2_latent_extraction', split=split,
                extracted=min(start + microbatch, count), target=count, pid=os.getpid(), utc=utc()))
    tensor = torch.cat(features)
    if not bool(torch.isfinite(tensor).all()) or set(labels) != {0, 1}:
        raise FloatingPointError('Invalid latent cache or unbalanced class availability')
    return tensor, torch.tensor(labels, dtype=torch.float32), metadata


@torch.no_grad()
def evaluate_head(head, cache, device, threshold=None):
    head.eval()
    features, labels, metadata = cache
    logits = torch.cat([head(chunk.to(device)).reshape(-1).cpu() for chunk in features.split(256)])
    loss = float(F.binary_cross_entropy_with_logits(logits, labels))
    scores = logits.numpy()
    summary = metrics(scores, labels.numpy(), threshold, 1)
    cells = []
    for cell in range(6):
        indices = [i for i, m in enumerate(metadata) if m['cell'] == cell]
        cells.append(dict(cell=cell, speed=metadata[indices[0]]['speed'], angle=metadata[indices[0]]['angle'],
            n=len(indices), **metrics(scores[indices], labels.numpy()[indices], summary['threshold'], 1)))
    return dict(n=len(labels), paired_contexts=len({m['context'] for m in metadata}), loss=loss,
        **summary, cells=cells, predictions=[dict(label=int(y), logit=float(s), metadata=m)
            for y, s, m in zip(labels, scores, metadata)])


def save_checkpoint(directory, model, optimizer, state, config, shuffle, device, best=False):
    directory.mkdir(parents=True, exist_ok=True)
    payload = cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        state=state, config=config, shuffle_rng=shuffle.get_state(), torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None))
    for name in ('latest', 'best') if best else ('latest',):
        temporary = directory / (name + '.pt.tmp')
        torch.save(payload, temporary)
        replace_with_retry(temporary, directory / (name + '.pt'))


def first_step_proof(directory, model, initial):
    saved = torch.load(directory / 'latest.pt', map_location='cpu')
    steps = [float(v['step']) for v in saved['optimizer']['state'].values()]
    expected = len([p for p in model.parameters() if p.requires_grad])
    finite = all(bool(torch.isfinite(p).all()) for p in saved['model'].values())
    adam_finite = all(bool(torch.isfinite(value).all()) for item in saved['optimizer']['state'].values()
        for value in item.values() if torch.is_tensor(value))
    if len(steps) != expected or set(steps) != {1.} or not finite or not adam_finite:
        raise RuntimeError('Persisted first-step optimizer/finite-parameter proof failed')
    atomic_json(directory / 'production_verified.json', dict(cpu_reloaded=True, step=1,
        expected_active_states=expected, optimizer_states=len(steps), adam_steps=steps,
        all_parameters_finite=finite, all_adam_tensors_finite=adam_finite,
        changed_tensors=[k for k, v in saved['model'].items() if not torch.equal(v, initial[k])],
        fresh_constructor=True, checkpoint_inputs=[]))


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
    model, optimizer = build_vae(args.device)
    vae_update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch, 1)
    synchronize(args.device)
    timings = []
    for index in range(args.profile_steps):
        clock = time.time()
        vae_update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch, index + 2)
        synchronize(args.device)
        timings.append(time.time() - clock)
    clock = time.time()
    evaluate_vae(model, 'val', 12, args.device, args.microbatch, deadline)
    synchronize(args.device)
    evaluation_seconds = (time.time() - clock) / 12
    clock = time.time()
    extract(model, 'train', 96, CACHE_TRAIN_OFFSET, args.device, args.microbatch, deadline)
    synchronize(args.device)
    extraction_seconds = (time.time() - clock) / 96
    head, head_optimizer = build_head(args.device)
    features = torch.randn(512, 256)
    labels = (torch.arange(512) % 2).float()
    head_update(head, head_optimizer, features, labels, torch.arange(128), args.device)
    synchronize(args.device)
    clock = time.time()
    for _ in range(20):
        head_update(head, head_optimizer, features, labels, torch.arange(128), args.device)
    synchronize(args.device)
    head_seconds = (time.time() - clock) / 20
    cache_pairs = args.cache_pairs + args.val_count + args.test_count
    head_reserve = max(1800., 1.5 * (extraction_seconds * cache_pairs + head_seconds * args.head_updates) + 300.)
    reserve = head_reserve + 1.5 * evaluation_seconds * (7 * args.val_count + args.val_count) + 300.
    cost = 1.25 * (float(np.median(timings)) + generation / POOL_UPDATES)
    feasible = int(max(0., deadline - time.time() - reserve) / cost) // POOL_UPDATES * POOL_UPDATES
    target = min(args.target_updates, feasible)
    if target < POOL_UPDATES:
        raise RuntimeError('Cannot fit both complete training stages within finite cap')
    atomic_json(args.profile_json, dict(started=started, deadline=deadline,
        target_updates=target, requested_target_updates=args.target_updates,
        head_updates=args.head_updates, cache_pairs=args.cache_pairs,
        vae_update_seconds=float(np.median(timings)), pool_generation_seconds=generation,
        vae_evaluation_seconds_per_pair=evaluation_seconds, extraction_seconds_per_pair=extraction_seconds,
        classifier_update_seconds=head_seconds, head_reserve_seconds=head_reserve,
        reserve_seconds=reserve, safety_factor=1.25, profile_weights_discarded=True,
        source_hashes=source_hashes(), device=args.device, microbatch=args.microbatch,
        val_count=args.val_count, test_count=args.test_count, wall_seconds=args.wall_seconds,
        torch_version=torch.__version__))


def production(args, run):
    if (run / 'production_config.json').exists():
        raise RuntimeError('Existing production; explicit recovery required')
    pinned = json.loads(args.profile_json.read_text(encoding='utf-8'))
    if source_hashes() != pinned['source_hashes']:
        raise RuntimeError('Source changed after profile')
    for key in ('device', 'microbatch', 'val_count', 'test_count'):
        if pinned[key] != getattr(args, key):
            raise ValueError('Profile/production argument mismatch: ' + key)
    if float(pinned['deadline']) - float(pinned['started']) != 14400.:
        raise ValueError('Scientific profile must retain exactly one four-hour allocation')
    target = int(pinned['target_updates'])
    deadline = float(pinned['deadline'])
    if target % POOL_UPDATES or args.head_updates != pinned['head_updates'] or args.cache_pairs != pinned['cache_pairs']:
        raise ValueError('Production must use prospectively pinned full exposure')
    remaining(deadline, 120)
    config = dict(fresh_vae=True, checkpoint_inputs=[], precision='fp32', tf32=False,
        vae_target_updates=target, head_target_updates=args.head_updates,
        batch_pairs=32, microbatch_pairs=args.microbatch, pool_size=POOL_SIZE, pool_epochs=POOL_EPOCHS,
        weighted_sum='(frame_oldest + 2*frame_middle + 3*frame_latest)/6',
        raw_sum_reconstruction='multiply both prediction and target by6 for original units',
        vae_loss='plain full-image MSE + beta mean-coordinate KL; beta1e-4 warmup250',
        vae_selection='minimum deterministic validation reconstruction MSE',
        classifier_selection='minimum validation BCE', classifier_threshold='validation BA-optimal scalar threshold, fixed positive sign',
        classifier_lr=1e-3, vae_lr=1e-4, clipping=None, classifier_batch=128,
        classifier_cache_pairs=args.cache_pairs, cache_train_offset=CACHE_TRAIN_OFFSET,
        cache_val_offset=CACHE_VAL_OFFSET, cache_test_offset=CACHE_TEST_OFFSET,
        frozen_selected_vae_in_classifier=True, label_supervision_stage1=False,
        source_hashes=source_hashes(), dataset_namespaces=dataset.NAMESPACES,
        dataset_version=dataset.VERSION, pid=os.getpid(), started=pinned['started'], deadline=deadline,
        val_count=args.val_count, test_count=args.test_count, device=args.device)
    atomic_json(run / 'production_config.json', config)
    vae, vae_optimizer = build_vae(args.device)
    initial = cpu_tree(vae.state_dict())
    shuffle = torch.Generator().manual_seed(SHUFFLE_SEED)
    state = dict(stage='vae', step=0, presentations=0, unique_pairs=0, best_step=None, best_loss=None)
    vae_dir, head_dir = run / 'vae', run / 'classifier'
    selected_loaded = False
    head, head_optimizer = None, None
    mean_image = None

    def status(phase, state, **extra):
        atomic_json(run / 'live_status.json', dict(phase=phase, pid=os.getpid(), utc=utc(),
            target_updates=target if state['stage'] == 'vae' else args.head_updates, **state, **extra))

    def validate_vae():
        status('stage1_validation', state)
        result = evaluate_vae(vae, 'val', args.val_count, args.device, args.microbatch, deadline, mean_image=mean_image)
        result['step'] = state['step']
        atomic_json(vae_dir / ('validation_%06d.json' % state['step']), result)
        better = state['best_loss'] is None or result['reconstruction'] < state['best_loss']
        if better:
            state.update(best_step=state['step'], best_loss=result['reconstruction'])
        save_checkpoint(vae_dir, vae, vae_optimizer, state, config, shuffle, args.device, better)

    try:
        validation_steps = {64, 5000, 10000, 15000, 20000, target}
        while state['step'] < target:
            remaining(deadline, 120)
            pool_index = state['step'] // POOL_UPDATES
            start = pool_index * POOL_SIZE
            pool = [dataset.generate(i, 'train') for i in range(start, start + POOL_SIZE)]
            if mean_image is None:
                mean_image = weighted_sum(torch.stack([clip for row in pool[:128] for clip in row[:2]])).mean(0, keepdim=True)
                atomic_json(run / 'train_mean_baseline.json', dict(training_pairs=min(128, len(pool)),
                    diagnostic_only=True, validation_or_test_fitted=False))
            state.update(pool_index=pool_index, pool_start=start, unique_pairs=start + POOL_SIZE)
            for epoch in range(POOL_EPOCHS):
                order = torch.randperm(POOL_SIZE, generator=shuffle)
                state.update(epoch=epoch, order=order.tolist(), cursor=0)
                for indices in order.split(32):
                    remaining(deadline, 120)
                    clock = time.time()
                    values = vae_update(vae, vae_optimizer, pool, indices, args.device, args.microbatch, state['step'] + 1)
                    state['step'] += 1
                    state['presentations'] += len(indices)
                    state['cursor'] += len(indices)
                    compact = {k: v for k, v in state.items() if k != 'order'}
                    append_json(run / 'progress.jsonl', dict(event='update', utc=utc(), **compact, **values, update_seconds=time.time() - clock))
                    status('stage1_training', compact, **values)
                    if state['step'] == 1:
                        save_checkpoint(vae_dir, vae, vae_optimizer, state, config, shuffle, args.device)
                        first_step_proof(vae_dir, vae, initial)
                        del initial
                    elif state['step'] % 250 == 0:
                        save_checkpoint(vae_dir, vae, vae_optimizer, state, config, shuffle, args.device)
                    if state['step'] in validation_steps:
                        validate_vae()
            del pool
        save_checkpoint(vae_dir, vae, vae_optimizer, state, config, shuffle, args.device)
        vae_terminal = dict(state)
        selected = torch.load(vae_dir / 'best.pt', map_location='cpu')
        vae.load_state_dict(selected['model'])
        vae_selected_step = selected['state']['step']
        del selected, vae_optimizer
        selected_loaded = True
        vae.eval().requires_grad_(False)
        vae.zero_grad(set_to_none=True)
        vae_before = cpu_tree(vae.state_dict())
        recon_final = evaluate_vae(vae, 'test', args.val_count, args.device, args.microbatch, deadline,
            run / 'weighted_reconstructions.npz', mean_image)
        recon_final['selected_step'] = vae_selected_step
        atomic_json(run / 'vae_final_test.json', recon_final)
        caches = {}
        for split, count, offset in [('train', args.cache_pairs, CACHE_TRAIN_OFFSET),
                ('val', args.val_count, CACHE_VAL_OFFSET), ('test', args.test_count, CACHE_TEST_OFFSET)]:
            caches[split] = extract(vae, split, count, offset, args.device, args.microbatch,
                deadline, run / 'live_status.json')
        train_features = caches['train'][0]
        feature_mean = train_features.mean(0, keepdim=True)
        feature_scale = train_features.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        atomic_json(run / 'latent_standardization.json', dict(train_only=True,
            mean=feature_mean[0].tolist(), scale=feature_scale[0].tolist(), scale_floor=1e-6))
        for split in caches:
            features, labels, metadata = caches[split]
            caches[split] = ((features - feature_mean) / feature_scale, labels, metadata)
        config['latent_standardization'] = 'train-cache coordinate mean/std only; persisted JSON; std floor1e-6'
        atomic_json(run / 'latent_cache_receipt.json', dict(selected_vae_step=vae_selected_step,
            train_pairs=args.cache_pairs, val_pairs=args.val_count, test_pairs=args.test_count,
            features_shape=list(caches['train'][0].shape), caches_RAM_only=True,
            train_label_counts=torch.bincount(caches['train'][1].long()).tolist(),
            cache_train_offset=CACHE_TRAIN_OFFSET, separate_from_vae_training=True))
        head, head_optimizer = build_head(args.device)
        initial = cpu_tree(head.state_dict())
        shuffle = torch.Generator().manual_seed(HEAD_SEED + 1)
        state = dict(stage='classifier', step=0, presentations=0, unique_pairs=args.cache_pairs,
            best_step=None, best_loss=None, threshold=None, selected_vae_step=vae_selected_step)
        validation_steps = {64, 1000, 2500, 5000, 7500, args.head_updates}
        while state['step'] < args.head_updates:
            order = torch.randperm(args.cache_pairs, generator=shuffle)
            state.update(order=order.tolist(), cursor=0)
            for indices in order.split(128):
                remaining(deadline)
                clock = time.time()
                values = head_update(head, head_optimizer, caches['train'][0], caches['train'][1], indices, args.device)
                state['step'] += 1
                state['presentations'] += len(indices)
                state['cursor'] += len(indices)
                compact = {k: v for k, v in state.items() if k != 'order'}
                append_json(run / 'progress.jsonl', dict(event='update', utc=utc(), **compact, **values, update_seconds=time.time() - clock))
                status('stage2_training', compact, **values)
                if state['step'] == 1:
                    save_checkpoint(head_dir, head, head_optimizer, state, config, shuffle, args.device)
                    first_step_proof(head_dir, head, initial)
                    del initial
                elif state['step'] % 250 == 0:
                    save_checkpoint(head_dir, head, head_optimizer, state, config, shuffle, args.device)
                if state['step'] in validation_steps:
                    result = evaluate_head(head, caches['val'], args.device)
                    result['step'] = state['step']
                    atomic_json(head_dir / ('validation_%06d.json' % state['step']), result)
                    better = state['best_loss'] is None or result['loss'] < state['best_loss']
                    if better:
                        state.update(best_step=state['step'], best_loss=result['loss'], threshold=result['threshold'])
                    save_checkpoint(head_dir, head, head_optimizer, state, config, shuffle, args.device, better)
                if state['step'] >= args.head_updates:
                    break
        save_checkpoint(head_dir, head, head_optimizer, state, config, shuffle, args.device)
        head_terminal = dict(state)
        selected = torch.load(head_dir / 'best.pt', map_location='cpu')
        head.load_state_dict(selected['model'])
        head_selected_step, threshold = selected['state']['step'], selected['state']['threshold']
        del selected
        final = evaluate_head(head, caches['test'], args.device, threshold)
        final.update(selected_vae_step=vae_selected_step, selected_classifier_step=head_selected_step,
            test_thresholds_fitted=False, encoder_frozen=True)
        immutable = all(torch.equal(v.detach().cpu(), vae_before[k]) for k, v in vae.state_dict().items())
        if not immutable or any(p.grad is not None for p in vae.parameters()):
            raise RuntimeError('Frozen VAE changed or received a stage2 gradient')
        final['encoder_immutability_verified'] = immutable
        atomic_json(run / 'final_test.json', final)
        (run / 'FINAL_REPORT.md').write_text('\n'.join(['# Weighted-sum VAE and latent change decoder', '',
            f"VAE: {vae_terminal['step']} updates, {vae_terminal['presentations']} pair presentations, {vae_terminal['unique_pairs']} unique pairs; selected {vae_selected_step}.",
            f"Classifier: {head_terminal['step']} updates, {head_terminal['presentations']} pair presentations on {args.cache_pairs} independent cached training pairs; selected {head_selected_step}.", '',
            f"Held-out change BA {final['balanced_accuracy']:.6f}; AUC {final['auc']:.6f}; BCE {final['loss']:.6f}.",
            f"Test: {final['n']} presentations / {final['paired_contexts']} matched contexts; all six cells in final_test.json.",
            'Three observed frames become (oldest + 2*middle + 3*latest)/6; multiply reconstruction by6 to recover requested raw-sum units.',
            'VAE uses only image reconstruction and KL. Classifier uses labeled concatenated posterior means from the fixed selected VAE.',
            'No prediction, contrastive objective or direction/angle input. Simplified continuous dots; no native Krauzlis or biological-validation claim.', '']), encoding='utf-8')
        receipt = dict(status='complete', stop_reason='planned_complete', completed=utc(),
            elapsed_seconds=time.time() - pinned['started'], vae_updates=vae_terminal['step'],
            classifier_updates=head_terminal['step'], selected_vae_step=vae_selected_step,
            selected_classifier_step=head_selected_step, final_n=final['n'], paired_contexts=final['paired_contexts'],
            encoder_immutability_verified=immutable, no_cloud=True)
        atomic_json(run / 'completion_verified.json', receipt)
        atomic_json(run / 'live_status.json', dict(phase='complete', pid=os.getpid(), **receipt))
    except BaseException as exc:
        if state['step'] and not selected_loaded:
            save_checkpoint(vae_dir, vae, vae_optimizer, state, config, shuffle, args.device)
        elif state['step'] and state['stage'] == 'classifier' and head is not None:
            # Only terminal training weights are saved here, never selected test weights.
            if 'head_terminal' not in locals():
                save_checkpoint(head_dir, head, head_optimizer, state, config, shuffle, args.device)
        atomic_json(run / 'failure.json', dict(type=type(exc).__name__, message=str(exc),
            state={k: v for k, v in state.items() if k != 'order'}, utc=utc()))
        raise


def main(args):
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
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--profile-json', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--wall-seconds', type=int, default=14400)
    parser.add_argument('--target-updates', type=int, default=25024)
    parser.add_argument('--head-updates', type=int, default=10000)
    parser.add_argument('--cache-pairs', type=int, default=64000)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--val-count', type=int, default=384)
    parser.add_argument('--test-count', type=int, default=3072)
    parser.add_argument('--lock-path', type=Path, default=Path(__file__).resolve().parents[2] / 'LocalRuntime/local_gpu_worker.lock')
    args = parser.parse_args()
    if args.profile_json is None:
        args.profile_json = args.run / 'profile.json'
    if args.wall_seconds != 14400:
        parser.error('Authorized finite allocation is exactly14400seconds for both stages together')
    if args.target_updates % 64 or args.val_count % 12 or args.test_count % 12 or args.cache_pairs % 128 or min(args.microbatch, args.head_updates, args.cache_pairs, args.target_updates, args.val_count, args.test_count, args.profile_steps) < 1:
        parser.error('Need complete64-update VAE pools, 12-trial evaluation cells, cache divisible128, positive counts')
    return args


if __name__ == '__main__':
    main(parse_args())
