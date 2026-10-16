"""One fresh predictive/VICReg VAE; bounded, sequential local execution.

The global VICReg gradient cache computes statistics on all 64 clips in an
effective 32-pair update, including the 16-clip partial tails. Import is inert.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from SecondPass.PixelContrastiveVAE.train import (
    GPULock, atomic_json, append_json, cpu_tree, metrics, synchronize, replace_with_retry)
from SecondPass.PixelContrastiveVAE.model import support_balanced_reconstruction
from . import dataset
from .model import PredictiveConsistencyVAE, augment_clips, vicreg_loss, local_losses as model_local_losses
from .queue import source_hashes

INIT_SEED = 189211
SHUFFLE_SEED = 189212
POOL_SIZE, POOL_EPOCHS, POOL_UPDATES = 1000, 2, 64


def configure(device):
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Local CUDA device unavailable')


def build(device):
    random.seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(INIT_SEED)
    model = PredictiveConsistencyVAE().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8,
                                   weight_decay=0.)


def local_losses(result, clips, beta):
    result = model_local_losses(result, clips, beta)
    result['local_loss'] = result.pop('loss')
    return result


def gradient_cache(model, views1, views2, device, microclips):
    """Return exact whole-batch VICReg output gradients; no encoder graph kept.

    Views are materialized once. No dropout/BN exists in this model, therefore
    reencoding the exact same views reproduces the cached deterministic values.
    """
    first, second = [], []
    with torch.no_grad():
        for start in range(0, len(views1), microclips):
            a = model.encode(views1[start:start + microclips].to(device))[0]
            b = model.encode(views2[start:start + microclips].to(device))[0]
            first.append(model.project(a))
            second.append(model.project(b))
    leaf1 = torch.cat(first).detach().requires_grad_(True)
    leaf2 = torch.cat(second).detach().requires_grad_(True)
    global_loss = vicreg_loss(leaf1, leaf2)
    gradients = torch.autograd.grad(global_loss['view_loss'], (leaf1, leaf2))
    return gradients, {key: float(value.detach()) for key, value in global_loss.items()}


def update(model, optimizer, rows, indices, device, microbatch, step, supplied_views=None):
    """One global effective-batch update, with exact cached view gradients."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Interleave each before/after clip; all 64 clips form EACH VICReg view.
    sequences = torch.stack([clip for i in indices for clip in rows[int(i)][:2]])
    clips = sequences[:, :3]
    views1, views2 = supplied_views if supplied_views is not None else (
        augment_clips(clips), augment_clips(clips))
    if views1.shape != clips.shape or views2.shape != clips.shape:
        raise ValueError('View shapes must match the whole effective clip batch')
    microclips = microbatch * 2
    (gradient1, gradient2), values = gradient_cache(model, views1, views2, device, microclips)
    beta = 1e-4 * min(1., step / 250.)
    for start in range(0, len(clips), microclips):
        end = min(start + microclips, len(clips))
        projection1 = model.project(model.encode(views1[start:end].to(device))[0])
        projection2 = model.project(model.encode(views2[start:end].to(device))[0])
        torch.autograd.backward((projection1, projection2),
                                (gradient1[start:end], gradient2[start:end]))
        local_sequences = sequences[start:end].to(device)
        local = local_losses(model(local_sequences[:, :3]), local_sequences, beta)
        fraction = (end - start) / len(clips)
        (local['local_loss'] * fraction).backward()
        for key, value in local.items():
            values[key] = values.get(key, 0.) + float(value.detach()) * fraction
    values['loss'] = values['local_loss'] + values['view_loss']
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    if missing:
        raise FloatingPointError('Missing active parameter gradients: ' + ', '.join(missing))
    gradients = [p.grad for p in model.parameters()]
    finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
    norm2 = torch.stack([g.detach().square().sum() for g in gradients]).sum()
    if not bool(finite) or not bool(torch.isfinite(norm2)) or not math.isfinite(values['loss']):
        raise FloatingPointError('Nonfinite gradient or training objective')
    values['gradient_norm'] = float(norm2.sqrt())
    values['vicreg_clips_per_view'] = len(clips)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError('Nonfinite updated weights')
    return values


def evaluation_indices(split, count):
    # Distinct high offsets; CPU fixtures use small indices only.
    offset = 12000000 if split == 'val' else 36000000
    return range(offset, offset + count)


@torch.no_grad()
def evaluate_objective(model, split, count, device, microbatch, deadline):
    """Matched six-frame constant-motion learning objective, separate draws.

    Each row supplies two causal3=>3 sequences from one movie; metadata/labels never
    enter this objective. VICReg statistics include all2*count observed contexts.
    """
    model.eval()
    total = {key: 0. for key in ('recon', 'prediction', 'kl', 'full_mse', 'prediction_mse')}
    projected1, projected2 = [], []
    forecast_errors, forecast_mse, copy_errors, copy_mse = [], [], [], []
    movie_contexts = set()
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else None
    torch.default_generator.manual_seed(189301)
    try:
        for start in range(0, count, microbatch):
            if time.time() >= deadline:
                raise TimeoutError('Objective evaluation deadline reached')
            indices = range(start, min(start + microbatch, count))
            rows = [dataset.generate_objective_evaluation(i, split) for i in indices]
            movie_contexts.update(row[3]['context'] for row in rows)
            sequences_cpu = torch.stack([clip for row in rows for clip in row[:2]])
            sequences = sequences_cpu.to(device)
            observed, future = sequences[:, :3], sequences[:, 3:6]
            result = model(observed)
            local = local_losses(result, sequences, 1e-4)
            for key in total:
                total[key] += float(local[key]) * len(rows) / count
            copied = observed[:, 2:3].expand(-1, 3, -1, -1, -1)
            forecast_errors.append(torch.stack([support_balanced_reconstruction(
                result['future_prediction'][:, k:k + 1], future[:, k:k + 1]) for k in range(3)]).cpu().numpy())
            forecast_mse.append((result['future_prediction'] - future).square().mean((0, 2, 3, 4)).cpu().numpy())
            copy_errors.append(torch.stack([support_balanced_reconstruction(
                copied[:, k:k + 1], future[:, k:k + 1]) for k in range(3)]).cpu().numpy())
            copy_mse.append((copied - future).square().mean((0, 2, 3, 4)).cpu().numpy())
            # This is a separate CPU augmentation stream, fully restored below.
            for view, destination in ((augment_clips(sequences_cpu[:, :3]), projected1),
                                      (augment_clips(sequences_cpu[:, :3]), projected2)):
                destination.append(model.project(model.encode(view.to(device))[0]).cpu())
    finally:
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    global_views = vicreg_loss(torch.cat(projected1), torch.cat(projected2))
    total.update({key: float(value) for key, value in global_views.items()})
    total['loss'] = total['recon'] + total['prediction'] + 1e-4 * total['kl'] + total['view_loss']
    # All microbatches have the same size except the possible final tail.
    weights = np.asarray([min(microbatch, count - start) for start in range(0, count, microbatch)], dtype=float)
    weighted = lambda rows: np.average(np.stack(rows), axis=0, weights=weights).tolist()
    return dict(movie_presentations=count, paired_nuisance_contexts=len(movie_contexts),
                observed_clip_presentations=2 * count, namespace=dataset.NAMESPACES[split],
                separate_objective_stream=True, metadata_used_in_loss=False,
                context_frames=3, trained_prediction_horizon=3,
                prediction_horizon_errors=weighted(forecast_errors),
                prediction_horizon_mse=weighted(forecast_mse),
                copy_last_horizon_errors=weighted(copy_errors),
                copy_last_horizon_mse=weighted(copy_mse), **total)


@torch.no_grad()
def evaluate(model, split, count, device, microbatch, deadline, thresholds=None, artifact_dir=None):
    if split != 'val' and thresholds is None:
        raise ValueError('Test requires fixed validation sign and threshold')
    model.eval()
    # Objective-selection draws contain three observed plus three stable future
    # frames. Native change draws below contain only the original3+3 frames.
    objective = evaluate_objective(model, split, min(count, 384), device, microbatch, deadline)
    names = ('rollout_surprise', 'first_frame_error', 'rollout_step4_error',
             'rollout_step5_error', 'rollout_full_mse', 'posterior_cosine',
             'projection_cosine', 'copy_last_rollout_error', 'copy_last_rollout_mse')
    score_lists = {name: [] for name in names}
    labels, metadata, predictions, image_samples, means, projections = [], [], [], [], [], []
    indices = list(evaluation_indices(split, count))
    for start in range(0, count, microbatch):
        if time.time() >= deadline:
            raise TimeoutError('Change evaluation deadline reached')
        rows = [dataset.generate_evaluation(i, split) for i in indices[start:start + microbatch]]
        first_cpu = torch.stack([r[0] for r in rows])
        second_cpu = torch.stack([r[1] for r in rows])
        first, second = first_cpu.to(device), second_cpu.to(device)
        mu_a, _ = model.encode(first)
        mu_b, _ = model.encode(second)
        projection_a, projection_b = model.project(mu_a), model.project(mu_b)
        # Three observed frames predict three future frames recursively.
        # B is strictly a target: even B0 never conditions the next forecast.
        rollout = model.rollout(first, steps=3)
        errors = torch.stack([support_balanced_reconstruction(
            rollout[j:j + 1, k:k + 1], second[j:j + 1, k:k + 1])
            for j in range(len(rows)) for k in range(3)]).reshape(len(rows), 3)
        raw_mse = (rollout - second).square().mean(dim=(2, 3, 4))
        copied = first[:, 2:3].expand(-1, 3, -1, -1, -1)
        copy_errors = torch.stack([support_balanced_reconstruction(
            copied[j:j + 1, k:k + 1], second[j:j + 1, k:k + 1])
            for j in range(len(rows)) for k in range(3)]).reshape(len(rows), 3)
        copy_mse = (copied - second).square().mean(dim=(2, 3, 4))
        scores = dict(rollout_surprise=errors[:, 1:3].mean(1), first_frame_error=errors[:, 0],
                      rollout_step4_error=errors[:, 1], rollout_step5_error=errors[:, 2],
                      rollout_full_mse=raw_mse[:, 1:3].mean(1),
                      copy_last_rollout_error=copy_errors[:, 1:3].mean(1),
                      copy_last_rollout_mse=copy_mse[:, 1:3].mean(1),
                      posterior_cosine=1. - F.cosine_similarity(mu_a, mu_b, dim=1, eps=1e-8),
                      projection_cosine=1. - F.cosine_similarity(projection_a, projection_b, dim=1, eps=1e-8))
        values = {key: value.cpu().tolist() for key, value in scores.items()}
        means.append(torch.cat((mu_a, mu_b)).cpu().numpy())
        projections.append(torch.cat((projection_a, projection_b)).cpu().numpy())
        for key in score_lists:
            score_lists[key].extend(values[key])
        for j, row in enumerate(rows):
            compact = {key: value for key, value in row[3].items() if key != 'positions_unwrapped_xy'}
            labels.append(int(row[2]))
            metadata.append(compact)
            predictions.append(dict(label=int(row[2]), metadata=compact,
                                    rollout_frame_errors=errors[j].cpu().tolist(),
                                    rollout_frame_full_mse=raw_mse[j].cpu().tolist(),
                                    copy_last_frame_errors=copy_errors[j].cpu().tolist(),
                                    copy_last_frame_full_mse=copy_mse[j].cpu().tolist(),
                                    **{key: float(values[key][j]) for key in values}))
            if artifact_dir is not None and len(image_samples) < 8:
                actual = torch.cat((first_cpu[j], second_cpu[j]), dim=0)
                forecast = torch.cat((first_cpu[j], rollout[j].cpu()), dim=0)
                image_samples.append((actual, forecast))
    summaries = {}
    for key, scores in score_lists.items():
        calibration = None if thresholds is None else thresholds[key]
        summary = metrics(scores, labels, None if calibration is None else calibration['threshold'],
                          None if calibration is None else calibration['sign'])
        cells = []
        for speed in (.375, 1., 2.):
            for angle in (26, 28):
                ix = [i for i, m in enumerate(metadata) if m['speed'] == speed and m['angle'] == angle]
                cell = metrics(np.asarray(scores)[ix], np.asarray(labels)[ix], summary['threshold'], summary['sign'])
                cells.append(dict(speed=speed, angle=angle, presentations=len(ix), paired_contexts=len(ix) // 2, **cell))
        summaries[key] = dict(**summary, cells=cells)
    mean_matrix = np.concatenate(means)
    projection_matrix = np.concatenate(projections)
    first_errors = np.asarray(score_lists['first_frame_error'])
    first_difference = float(np.abs(first_errors[::2] - first_errors[1::2]).max())
    if first_difference > 1e-6:
        raise RuntimeError('Paired first forecast error differs; inspect rollout/target leakage')
    if artifact_dir is not None:
        from PIL import Image
        sheets = []
        for actual, forecast in image_samples:
            convert = lambda frames: np.concatenate([
                (frame.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                for frame in frames], axis=1)
            sheets.append(np.concatenate((convert(actual), convert(forecast)), axis=0))
        Image.fromarray(np.concatenate(sheets, axis=0)).save(Path(artifact_dir) / 'forecast_examples.png')
        atomic_json(Path(artifact_dir) / 'forecast_examples.json', dict(
            trials=[p['metadata'] for p in predictions[:len(image_samples)]],
            rows='two per trial: actual A0/A1/A2/B0/B1/B2; observed A0/A1/A2 then three autoregressive forecasts',
            selection='first eight held-out test presentations, fixed order'))
    return dict(n=count, paired_contexts=len({m['context'] for m in metadata}),
                loss=objective['loss'], objective=objective,
                primary_score='rollout_surprise', scores=summaries,
                first_frame_pair_max_difference=first_difference,
                posterior_variance=float(mean_matrix.var(axis=0).mean()),
                posterior_norm=float(np.linalg.norm(mean_matrix, axis=1).mean()),
                projection_variance=float(projection_matrix.var(axis=0).mean()),
                projection_std_mean=float(projection_matrix.std(axis=0, ddof=1).mean()),
                score_means={key: float(np.mean(value)) for key, value in score_lists.items()},
                predictions=predictions)


def profile(args, run):
    origin = json.loads((run / 'budget.json').read_text(encoding='utf-8'))
    if args.profile_json.exists() or origin.get('worker_pid') != os.getpid():
        raise RuntimeError('Existing profile/budget origin; no implicit renewal')
    started, deadline = float(origin['started']), float(origin['deadline'])
    clock = time.time()
    rows = [dataset.generate(i, 'train') for i in range(POOL_SIZE)]
    generation = time.time() - clock
    model, optimizer = build(args.device)
    update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch, 1)
    synchronize(args.device)
    timings = []
    for i in range(args.profile_steps):
        clock = time.time()
        update(model, optimizer, rows, torch.arange(32), args.device, args.microbatch, i + 2)
        synchronize(args.device)
        timings.append(time.time() - clock)
    clock = time.time()
    evaluate(model, 'val', 12, args.device, args.microbatch, deadline)
    synchronize(args.device)
    eval_seconds = (time.time() - clock) / 12
    reserve = 1.25 * eval_seconds * (7 * args.val_count + args.test_count) + 600.
    available = max(0., deadline - time.time() - reserve)
    per_update = 1.25 * (float(np.median(timings)) + generation / POOL_UPDATES)
    feasible = int(available / per_update) // POOL_UPDATES * POOL_UPDATES
    target = min(args.target_updates, feasible)
    if target < POOL_UPDATES:
        raise RuntimeError('Cannot fit one full pool plus evaluation in new finite cap')
    atomic_json(args.profile_json, dict(started=started, deadline=deadline,
                wall_seconds=args.wall_seconds, target_updates=target,
                requested_target_updates=args.target_updates, update_seconds=float(np.median(timings)),
                measured_updates=len(timings), pool_generation_seconds=generation,
                evaluation_seconds_per_trial=eval_seconds, reserve_seconds=reserve,
                safety_factor=1.25, profile_weights_discarded=True,
                source_hashes=source_hashes(), device=args.device,
                torch_version=torch.__version__, cpu_threads=2,
                global_vicreg_gradient_cache=True, clips_per_full_view=64))


def production(args, run):
    if (run / 'production_config.json').exists() or (run / 'latest.pt').exists():
        raise RuntimeError('Existing production; explicit recovery required, no overwrite')
    pinned = json.loads(args.profile_json.read_text(encoding='utf-8'))
    if source_hashes() != pinned['source_hashes']:
        raise RuntimeError('Source changed after profile')
    if args.target_updates > pinned['target_updates'] or args.target_updates % POOL_UPDATES:
        raise ValueError('Exposure must be prospectively pinned complete 64-update pools')
    deadline = float(pinned['deadline'])
    if time.time() >= deadline - 120:
        raise TimeoutError('Original cap exhausted before production')
    model, optimizer = build(args.device)
    initial = cpu_tree(model.state_dict())
    shuffle = torch.Generator().manual_seed(SHUFFLE_SEED)
    config = dict(fresh_whole_model=True, checkpoint_inputs=[], init_seed=INIT_SEED,
        shuffle_seed=SHUFFLE_SEED, optimizer='Adam', lr=1e-4, epsilon=1e-8,
        weight_decay=0., clipping=None, precision='fp32', tf32=False, cpu_threads=2,
        device=args.device, cuda_device=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        batch_pairs=32, microbatch_pairs=args.microbatch, pool_size=POOL_SIZE, pool_epochs=POOL_EPOCHS,
        target_updates=args.target_updates, target_unique_pairs=args.target_updates // POOL_UPDATES * POOL_SIZE,
        target_presentations=args.target_updates // POOL_UPDATES * POOL_SIZE * POOL_EPOCHS, all_learned_weights_trainable=True,
        beta_max=1e-4, beta_warmup_updates=250, loss_weights=dict(recon=1., prediction=1., invariance=1., variance=1., covariance=.04),
        global_vicreg_clips_per_view=64, partial_tail_clips_per_view=16,
        checkpoint_selection='minimum unsupervised validation loss',
        decision_calibration='validation sign and scalar threshold; fixed held-out test',
        primary_score='rollout steps4/5 support-balanced forecast surprise',
        no_actual_second_clip_conditioning_in_rollout=True,
        observed_context_frames=3, trained_autoregressive_prediction_frames=3,
        full_temporal_prediction_gradients=True,
        learning_sequences='two six-frame constant-motion segments within continuous twelve-frame movies',
        objective_validation='independent six-frame context/future draws at index+60000000; labels unused',
        fresh_corrected_architecture=True, budget_origin=str(args.budget_origin) if args.budget_origin else None,
        source_hashes=source_hashes(), dataset_namespaces=dataset.NAMESPACES,
        dataset_version=dataset.VERSION, budget_started=pinned['started'], hard_deadline=deadline,
        pid=os.getpid(), val_count=args.val_count, test_count=args.test_count)
    atomic_json(run / 'production_config.json', config)
    state = dict(step=0, presentations=0, unique_pairs=0, pool_index=-1,
                 best_step=None, best_loss=None, thresholds=None)
    validation_steps = {64, 5000, 10000, 15000, 20000, 25024, args.target_updates}
    selected_loaded = False

    def save(best=False):
        payload = cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            state=state, config=config, torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), shuffle_rng=shuffle.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None))
        for name in ('latest', 'best') if best else ('latest',):
            temporary = run / (name + '.pt.tmp')
            torch.save(payload, temporary)
            replace_with_retry(temporary, run / (name + '.pt'))
        atomic_json(run / 'checkpoint_receipt.json', dict(step=state['step'], optimizer_states=len(optimizer.state)))

    def validate():
        atomic_json(run / 'live_status.json', dict(phase='validation', pid=os.getpid(),
                    utc=dt.datetime.now(dt.timezone.utc).isoformat(), **state))
        val = evaluate(model, 'val', args.val_count, args.device, args.microbatch, deadline - 30)
        val['step'] = state['step']
        atomic_json(run / ("validation_%06d.json" % state['step']), val)
        better = state['best_loss'] is None or val['loss'] < state['best_loss']
        if better:
            state.update(best_step=state['step'], best_loss=val['loss'],
                thresholds={key: dict(sign=item['sign'], threshold=item['threshold']) for key, item in val['scores'].items()})
        save(best=better)

    try:
        while state['step'] < args.target_updates:
            if time.time() >= deadline - 120:
                raise TimeoutError('Finite local cap reached before pinned exposure completed')
            state['pool_index'] += 1
            start = state['pool_index'] * POOL_SIZE
            pool = [dataset.generate(i, 'train') for i in range(start, start + POOL_SIZE)]
            state.update(pool_start=start, unique_pairs=start + POOL_SIZE)
            for epoch in range(POOL_EPOCHS):
                order = torch.randperm(POOL_SIZE, generator=shuffle)
                state.update(epoch=epoch, order=order.tolist(), cursor=0)
                for indices in order.split(32):
                    if time.time() >= deadline - 120:
                        raise TimeoutError('Finite local cap reached during production')
                    clock = time.time()
                    values = update(model, optimizer, pool, indices, args.device, args.microbatch, state['step'] + 1)
                    state['step'] += 1
                    state['presentations'] += len(indices)
                    state['cursor'] += len(indices)
                    compact = {key: value for key, value in state.items() if key != 'order'}
                    append_json(run / 'progress.jsonl', dict(event='update', **compact, **values, update_seconds=time.time() - clock))
                    atomic_json(run / 'live_status.json', dict(phase='training', pid=os.getpid(),
                        utc=dt.datetime.now(dt.timezone.utc).isoformat(), **compact, **values))
                    if state['step'] == 1:
                        save()
                        saved = torch.load(run / 'latest.pt', map_location='cpu')
                        changed = [key for key, value in saved['model'].items() if not torch.equal(value, initial[key])]
                        expected = len(list(model.parameters()))
                        steps = [float(v['step']) for v in saved['optimizer']['state'].values()]
                        if len(steps) != expected or any(step != 1. for step in steps):
                            raise RuntimeError('Saved CPU first-step Adam/parameter proof incomplete')
                        atomic_json(run / 'production_verified.json', dict(step=1, cpu_reloaded=True,
                            optimizer_states=len(steps), expected_active_states=expected,
                            adam_steps=steps, changed_tensors=changed,
                            unchanged_tensors=[key for key in saved['model'] if key not in changed],
                            zero_change_note='VICReg common projector bias is shift-invariant; active Adam coverage remains required',
                            fresh_constructor=True, checkpoint_inputs=[]))
                        del saved, initial
                    elif state['step'] % 250 == 0:
                        save()
                    if state['step'] in validation_steps:
                        validate()
            del pool
        save()
        selected = torch.load(run / 'best.pt', map_location='cpu')
        model.load_state_dict(selected['model'])
        selected_loaded = True
        selected_step, thresholds = selected['state']['step'], selected['state']['thresholds']
        del selected
        atomic_json(run / 'live_status.json', dict(phase='final_test', pid=os.getpid(),
                    utc=dt.datetime.now(dt.timezone.utc).isoformat(), **state))
        final = evaluate(model, 'test', args.test_count, args.device, args.microbatch, deadline - 10, thresholds, run)
        final.update(selected_step=selected_step, test_thresholds_fitted=False)
        atomic_json(run / 'final_test.json', final)
        primary = final['scores']['rollout_surprise']
        report = ["# Predictive-consistency VAE: completed local run", "",
            f"{state['step']} fresh updates, {state['presentations']} presentations, {state['unique_pairs']} unique twelve-frame movies.",
            f"Unsupervised validation selected step {selected_step}; test has {final['n']} presentations in {final['paired_contexts']} paired contexts.", "",
            f"Primary forecast surprise: raw AUC {primary['raw_auc']:.6f}; validation sign {primary['sign']:+d}; test AUC {primary['auc']:.6f}; BA {primary['balanced_accuracy']:.6f}.", "",
            "Three observed A frames forecast three future frames recursively; actual B never conditions the rollout.",
            "Training averages losses over all three recursively predicted future frames with full temporal gradients.",
            "The first forecast compares identical paired B0 targets and is a built-in chance control.",
            "All six cells and trial scores are retained in final_test.json; labels only calibrate validation decisions.",
            "No direction or angle target enters learning. These are simplified persistent coherent dots, not native Krauzlis or attention results.", ""]
        (run / 'FINAL_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
        receipt = dict(status='complete', stop_reason='planned_complete', **state,
            selected_step=selected_step, final_n=final['n'], paired_contexts=final['paired_contexts'],
            completed=dt.datetime.now(dt.timezone.utc).isoformat(), no_cloud=True,
            elapsed_seconds=time.time() - pinned['started'])
        atomic_json(run / 'completion_verified.json', receipt)
        atomic_json(run / 'live_status.json', dict(phase='complete', pid=os.getpid(), **receipt))
    except BaseException as exc:
        if state['step'] and not selected_loaded:
            save()
        atomic_json(run / 'failure.json', dict(type=type(exc).__name__, message=str(exc), **state))
        raise
    finally:
        del model, optimizer
        gc.collect()


def main(args):
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    # Lock ownership precedes CUDA availability checks or construction.
    lock = GPULock(args.lock_path)
    try:
        if args.profile_only:
            if args.profile_json.exists() or (run / 'budget.json').exists():
                raise RuntimeError('Existing budget/profile; no implicit renewal')
            if args.budget_origin:
                origin = json.loads(args.budget_origin.read_text(encoding='utf-8'))
                started, deadline = float(origin['started']), float(origin['deadline'])
                if deadline - started > args.wall_seconds + 1e-6:
                    raise ValueError('Budget origin exceeds the authorized four-hour envelope')
            else:
                started = time.time()
                deadline = started + args.wall_seconds
            if time.time() >= deadline - 120:
                raise TimeoutError('Original experiment cap exhausted; correction cannot renew it')
            atomic_json(run / 'budget.json', dict(started=started, deadline=deadline,
                        wall_seconds=deadline - started, phase='profile', worker_pid=os.getpid(),
                        corrective_profile_started=time.time(), no_budget_renewal=True,
                        budget_origin=str(args.budget_origin) if args.budget_origin else None))
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
    parser.add_argument('--profile-json', type=Path, required=True)
    parser.add_argument('--budget-origin', type=Path,
                        help='Reuse original experiment ledger/deadline for a fresh corrected model; never inherit weights')
    parser.add_argument('--target-updates', type=int, default=25024)
    parser.add_argument('--wall-seconds', type=int, default=14400)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--val-count', type=int, default=384)
    parser.add_argument('--test-count', type=int, default=3072)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--lock-path', type=Path, default=Path(__file__).resolve().parents[2] / 'LocalRuntime' / 'local_gpu_worker.lock')
    args = parser.parse_args()
    if args.val_count % 12 or args.test_count % 12 or args.target_updates % 64 or args.microbatch < 1:
        parser.error('Require positive microbatch, complete64-update pools, and balanced12-example evaluation cells')
    return args


if __name__ == '__main__':
    main(parse_args())
