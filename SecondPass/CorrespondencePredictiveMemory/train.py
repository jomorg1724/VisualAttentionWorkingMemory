"""Finite fresh correspondence, prediction and spatial-memory training."""
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
from torch.nn import functional as F
from SecondPass.PixelContrastiveVAE.train import (
    GPULock, atomic_json, append_json, cpu_tree, metrics, replace_with_retry,
    configure, synchronize)
from . import dataset
from .model import CorrespondencePredictiveMemory
from .loss import objective

ROOT = Path(__file__).resolve().parents[2]
POOL_SIZE, POOL_EPOCHS, POOL_UPDATES, BATCH = 1000, 2, 64, 32
INIT_SEED, SHUFFLE_SEED = 194211, 194212
TRANSFER_COUNT = 768


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / p for p in (
        'SecondPass/PixelContrastiveVAE/train.py',
        'SecondPass/VariationalMotionPredictor/dataset.py',
        'SecondPass/PredictiveConsistencyVAE/queue.py',
        'SecondPass/OutcomeContrastiveRViT/model.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build(device):
    random.seed(INIT_SEED); np.random.seed(INIT_SEED); torch.manual_seed(INIT_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(INIT_SEED)
    model = CorrespondencePredictiveMemory().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8, weight_decay=0.)


def remaining(deadline, reserve=120):
    if time.time() >= deadline - reserve:
        raise TimeoutError('Finite local allowance exhausted; no automatic extension')


def balanced_order(shuffle):
    """Exactly16/class for full batches and4/class in the final8-item tail."""
    zero = 2 * torch.randperm(POOL_SIZE // 2, generator=shuffle)
    one = 2 * torch.randperm(POOL_SIZE // 2, generator=shuffle) + 1
    batches = []
    for start in range(0, POOL_SIZE // 2, BATCH // 2):
        batch = torch.cat((zero[start:start+BATCH//2], one[start:start+BATCH//2]))
        batches.append(batch[torch.randperm(len(batch), generator=shuffle)])
    return torch.cat(batches)


def update(model, optimizer, rows, indices, device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    chosen = [rows[int(i)] for i in indices]
    paired = torch.stack([r[0] for r in chosen]).to(device)
    frames = torch.cat((paired[:,0], paired[:,1]))
    labels = torch.tensor([r[1] for r in chosen], device=device)
    logits, cls, details = model(frames, return_details=True)
    loss, parts = objective(logits, details, labels)
    loss.backward()
    missing = [n for n,p in model.named_parameters() if p.grad is None]
    if missing or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Missing/nonfinite gradients: '+', '.join(missing))
    norm = torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite objective/gradient norm')
    optimizer.step()
    choices = logits.detach().argmax(-1); target = labels.repeat(2)
    return dict(loss=float(loss.detach()), decision_loss=float(parts['decision'].detach()),
        prediction_loss=float(parts['prediction'].detach()),
        representation_loss=float(parts['representation'].detach()),
        invariance=float(parts['invariance'].detach()), variance_penalty=float(parts['variance'].detach()),
        covariance_penalty=float(parts['covariance'].detach()), transform_std=float(parts['transform_std'].detach()),
        temperature=float(details['temperature'].detach()), gradient_norm=float(norm),
        accuracy=float((choices==target).float().mean()), batch_size=len(indices), movie_presentations=2*len(indices),
        cls_std=float(cls.detach().std(0,unbiased=False).mean()),
        TP=int(((choices==1)&(target==1)).sum()), TN=int(((choices==0)&(target==0)).sum()),
        FP=int(((choices==1)&(target==0)).sum()), FN=int(((choices==0)&(target==1)).sum()))


def summary(scores, labels):
    result = metrics(scores, labels, 0., 1)
    choices = scores > 0  # argmax([logit0,logit1]) chooses0 at exact ties
    result['balanced_accuracy'] = .5 * (float((~choices[labels == 0]).mean()) + float(choices[labels == 1].mean()))
    result.update(TP=int(((labels == 1) & choices).sum()), TN=int(((labels == 0) & ~choices).sum()),
                  FP=int(((labels == 0) & choices).sum()), FN=int(((labels == 1) & ~choices).sum()))
    return result


@torch.no_grad()
def evaluate(model, split, count, device, deadline, batch=8, domain="standard"):
    model.eval(); scores, labels, metadata, cls_vectors = [], [], [], []
    offset = 12000000 if split == 'val' else 36000000
    for start in range(0, count, batch):
        remaining(deadline, 30)
        rows = [dataset.generate(offset + i, split, True, domain=domain) for i in range(start, min(start+batch, count))]
        logits, cls = model(torch.stack([r[0] for r in rows]).to(device))
        scores.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
        labels.extend(r[1] for r in rows); metadata.extend(r[2] for r in rows)
        cls_vectors.append(cls.cpu())
    scores, labels = np.asarray(scores), np.asarray(labels)
    cells = []
    for cell in range(6):
        ix = [i for i, m in enumerate(metadata) if m['cell'] == cell]
        if not ix:
            continue
        cells.append(dict(cell=cell, speed=metadata[ix[0]]['speed'], angle=metadata[ix[0]]['angle'],
                         n=len(ix), **summary(scores[ix], labels[ix])))
    vectors = torch.cat(cls_vectors)
    ce = F.binary_cross_entropy_with_logits(torch.tensor(scores), torch.tensor(labels, dtype=torch.float64))
    return dict(domain=domain, n=count, paired_contexts=len({m['context'] for m in metadata}), loss=float(ce),
        decision='terminal two-logit argmax; no calibrated threshold', **summary(scores, labels),
        cls_std=float(vectors.std(0, unbiased=False).mean()), cells=cells,
        predictions=[dict(label=int(y), logit=float(s), metadata=m) for y, s, m in zip(labels, scores, metadata)])


def save(run, model, optimizer, state, config, shuffle, device, best=False):
    payload = cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), state=state,
        config=config, shuffle_rng=shuffle.get_state(), torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None))
    for name in ('latest', 'best') if best else ('latest',):
        temporary = run / (name + '.pt.tmp')
        torch.save(payload, temporary); replace_with_retry(temporary, run / (name + '.pt'))


def profile(args, run):
    if (run / 'budget.json').exists():
        raise RuntimeError('Existing allocation; refusing automatic renewal')
    started = time.time(); deadline = started + 14400.
    atomic_json(run / 'budget.json', dict(started=started, deadline=deadline, wall_seconds=14400, worker_pid=os.getpid()))
    configure(args.device)
    model, optimizer = build(args.device)
    # Bounded learning-capability check on eight fixed genuine trials, within
    # this same budget. Discard these weights; it is not a held-out result.
    tiny = [dataset.training_pair(48000000+i) for i in range(8)]
    trace = []
    for step in range(1, 257):
        remaining(deadline)
        values = update(model, optimizer, tiny, torch.arange(8), args.device)
        if step in (1, 16, 64, 128, 256) or (values['accuracy'] == 1 and values['decision_loss'] < .05):
            trace.append(dict(step=step, **values))
            atomic_json(run / 'tiny_fit_check.json', dict(trace=trace, training_only=True,
                held_out_result=False, fixed_trials=8, weights_discarded=True))
            atomic_json(run / 'live_status.json', dict(phase='tiny_fit_check', pid=os.getpid(), utc=utc(), **trace[-1]))
        if values['accuracy'] == 1 and values['decision_loss'] < .05:
            break
    del model, optimizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    clock = time.time(); rows = [dataset.training_pair(i) for i in range(POOL_SIZE)]; generation = time.time() - clock
    model, optimizer = build(args.device)
    timings = []
    for step in range(4):
        remaining(deadline)
        clock = time.time(); update(model, optimizer, rows, torch.arange(32), args.device); synchronize(args.device)
        if step:
            timings.append(time.time() - clock)
    clock = time.time(); evaluate(model, 'val', 12, args.device, deadline); synchronize(args.device)
    eval_seconds = (time.time() - clock) / 12
    reserve = 600 + 1.25 * eval_seconds * (7 * args.val_count + args.test_count + TRANSFER_COUNT)
    cost = 1.25 * (float(np.median(timings)) + generation / POOL_UPDATES)
    feasible = int(max(0, deadline-time.time()-reserve) / cost) // POOL_UPDATES * POOL_UPDATES
    target = min(args.target_updates, feasible)
    if target < POOL_UPDATES:
        raise RuntimeError('No complete pool fits inside the finite cap')
    atomic_json(run / 'profile.json', dict(started=started, deadline=deadline, target_updates=target,
        requested_target_updates=args.target_updates, update_seconds=float(np.median(timings)),
        pool_generation_seconds=generation, evaluation_seconds_per_pair=eval_seconds,
        reserve_seconds=reserve, profile_weights_discarded=True, source_hashes=source_hashes(),
        device=args.device, val_count=args.val_count, test_count=args.test_count, batch_pairs=32,
        tiny_fit_last=trace[-1], torch_version=torch.__version__))


def production(args, run):
    if (run / 'production_config.json').exists():
        raise RuntimeError('Existing production; explicit checkpoint recovery required')
    pinned = json.loads((run / 'profile.json').read_text())
    if pinned['source_hashes'] != source_hashes():
        raise RuntimeError('Source changed after profile')
    for key in ('device', 'val_count', 'test_count'):
        if pinned[key] != getattr(args, key):
            raise ValueError('Profile/production setting differs: ' + key)
    target, deadline = pinned['target_updates'], pinned['deadline']
    remaining(deadline)
    configure(args.device); model, optimizer = build(args.device)
    initial = cpu_tree(model.state_dict())
    shuffle = torch.Generator().manual_seed(SHUFFLE_SEED)
    config = dict(version='local_correspondence_transport_memory_v1', fresh_whole_model=True,
        checkpoint_inputs=[], all_learned_parameters_trainable=True,
        init_seed=INIT_SEED, shuffle_seed=SHUFFLE_SEED, target_updates=target,
        batch_pairs=32, microbatch_losses=False, full_batch_regularization=True,
        pool_size=POOL_SIZE, pool_epochs=POOL_EPOCHS, pool_updates=POOL_UPDATES,
        optimizer='Adam', lr=3e-4, epsilon=1e-8, weight_decay=0., clipping=None,
        precision='fp32', tf32=False, cpu_threads=2, frames=6, speeds=dataset.SPEEDS,
        angles=dataset.ANGLES, dataset_namespaces=dataset.NAMESPACES,
        architecture='fixed local correlations 4x4x441; learned population refinement; width64 heads4 recurrentViT spatialGRU and CLSGRU; prediction-error readout',
        selection='minimum validation decision CE', decision='terminal argmax, tie0',
        objective='CE +0.5 causal transport prediction +0.1(invariance+variance+0.04covariance)',
        two_independent_appearances=True, simulator_targets=False, fixed_front_end=True,
        source_hashes=source_hashes(), parameter_count=sum(p.numel() for p in model.parameters()),
        device=args.device, pid=os.getpid(), started=pinned['started'], deadline=deadline,
        val_count=args.val_count, test_count=args.test_count)
    atomic_json(run / 'production_config.json', config)
    state = dict(step=0, presentations=0, unique_pairs=0, best_step=None, best_loss=None)
    selected_loaded = False

    def status(phase, **values):
        atomic_json(run / 'live_status.json', dict(phase=phase, pid=os.getpid(), utc=utc(),
            target_updates=target, **{k: v for k, v in state.items() if k != 'order'}, **values))

    def validate():
        status('validation')
        result = evaluate(model, 'val', args.val_count, args.device, deadline)
        result['step'] = state['step']; atomic_json(run / ('validation_%06d.json' % state['step']), result)
        better = state['best_loss'] is None or result['loss'] < state['best_loss']
        if better:
            state.update(best_step=state['step'], best_loss=result['loss'])
        save(run, model, optimizer, state, config, shuffle, args.device, best=better)

    try:
        checks = {64, 5000, 10000, 15000, 20000, target}
        while state['step'] < target:
            remaining(deadline)
            pool_index = state['step'] // POOL_UPDATES; start = pool_index * POOL_SIZE
            rows = [dataset.training_pair(i) for i in range(start, start+POOL_SIZE)]
            state.update(pool_index=pool_index, pool_start=start, unique_pairs=start+POOL_SIZE)
            for epoch in range(POOL_EPOCHS):
                order = balanced_order(shuffle); state.update(epoch=epoch, order=order.tolist(), cursor=0)
                for indices in order.split(BATCH):
                    remaining(deadline); clock = time.time()
                    values = update(model, optimizer, rows, indices, args.device)
                    state['step'] += 1; state['presentations'] += len(indices); state['cursor'] += len(indices)
                    append_json(run / 'progress.jsonl', dict(event='update', utc=utc(),
                        **{k: v for k, v in state.items() if k != 'order'}, **values, update_seconds=time.time()-clock))
                    status('training', **values)
                    if state['step'] == 1 or state['step'] % 250 == 0:
                        save(run, model, optimizer, state, config, shuffle, args.device)
                    if state['step'] == 1:
                        saved = torch.load(run / 'latest.pt', map_location='cpu')
                        adam = saved['optimizer']['state']
                        assert len(adam) == len(list(model.parameters()))
                        assert all(float(s['step']) == 1 for s in adam.values())
                        atomic_json(run / 'production_verified.json', dict(cpu_reloaded=True, step=1,
                            optimizer_states=len(adam), changed_tensors=[k for k,v in saved['model'].items() if not torch.equal(v, initial[k])],
                            checkpoint_inputs=[], full_batch_regularization=True, two_appearance_views=True))
                        del saved, initial
                    if state['step'] in checks:
                        validate()
            del rows
        save(run, model, optimizer, state, config, shuffle, args.device)
        selected = torch.load(run / 'best.pt', map_location='cpu')
        model.load_state_dict(selected['model']); selected_loaded = True
        selected_step = selected['state']['step']; del selected
        status('final_test'); final = evaluate(model, 'test', args.test_count, args.device, deadline)
        final.update(selected_step=selected_step, test_thresholds_fitted=False)
        atomic_json(run / 'final_test.json', final)
        status('transfer_test')
        transfer = evaluate(model, 'test', TRANSFER_COUNT, args.device, deadline, domain='transfer')
        transfer.update(selected_step=selected_step, used_for_selection=False)
        atomic_json(run / 'transfer_test.json', transfer)
        (run / 'FINAL_REPORT.md').write_text(
            '# Correspondence predictive memory\n\n'
            f"Completed {state['step']} updates / {state['presentations']} presentations / {state['unique_pairs']} unique movies.\n\n"
            f"Selected {selected_step}; final BA {final['balanced_accuracy']:.6f}, AUC {final['auc']:.6f}, CE {final['loss']:.6f}.\n\n"
            'Six raw frames with fixed local correspondence populations, learned refinement, causal transport prediction, spatial recurrent memory and paired-appearance regularization.\n'
            'No motion-angle/flow targets; synthetic same-dynamics pairing is an explicit training prior. Neuroscience-inspired, not a biological circuit validation.\n', encoding='utf-8')
        receipt = dict(status='complete', stop_reason='planned_complete', completed=utc(),
            elapsed_seconds=time.time()-pinned['started'], updates=state['step'],
            presentations=state['presentations'], unique_pairs=state['unique_pairs'],
            selected_step=selected_step, final_n=final['n'], paired_contexts=final['paired_contexts'], transfer_n=TRANSFER_COUNT,
            movie_presentations=2*state['presentations'], unique_movies=2*state['unique_pairs'], no_cloud=True)
        atomic_json(run / 'completion_verified.json', receipt)
        atomic_json(run / 'live_status.json', dict(phase='complete', pid=os.getpid(), **receipt))
    except BaseException as exc:
        if state['step'] and not selected_loaded:
            save(run, model, optimizer, state, config, shuffle, args.device)
        atomic_json(run / 'failure.json', dict(type=type(exc).__name__, message=str(exc), utc=utc(),
            state={k: v for k, v in state.items() if k != 'order'}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--target-updates', type=int, default=25024)
    parser.add_argument('--val-count', type=int, default=384)
    parser.add_argument('--test-count', type=int, default=3072)
    args = parser.parse_args()
    if args.target_updates < 64 or args.target_updates % 64 or min(args.val_count, args.test_count) < 12 or args.val_count % 12 or args.test_count % 12:
        parser.error('Need complete64-update pools and positive balanced12-example evaluation multiples')
    run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    lock = GPULock(ROOT / 'LocalRuntime/local_gpu_worker.lock')
    try:
        profile(args, run) if args.profile_only else production(args, run)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
