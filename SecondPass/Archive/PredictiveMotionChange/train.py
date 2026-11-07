"""One bounded local frozen-encoder/FFN motion-change experiment."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
import torch.nn.functional as F
from .dataset import generate, SPEEDS, ANGLES
from .model import FrozenMotionEncoder, ChangeFFN
from SecondPass.VariationalMotionPredictor.worker import atomic_json, cpu_tree


def encode_set(encoder, split, start, count, run, deadline):
    features, labels, metadata = [], [], []
    for offset in range(0, count, 8):
        if time.time() >= deadline:
            raise TimeoutError('Dataset encoding reached wall cap')
        rows = [generate(i, split) for i in range(start + offset, min(start + offset + 8, start + count))]
        before = torch.stack([r[0] for r in rows]).to('mps')
        after = torch.stack([r[1] for r in rows]).to('mps')
        features.append(torch.cat((encoder(before), encoder(after)), dim=1).cpu())
        labels.extend(r[2] for r in rows)
        metadata.extend(r[3] for r in rows)
    return torch.cat(features), torch.tensor(labels, dtype=torch.long), metadata


@torch.no_grad()
def evaluate(head, data):
    x, y, metadata = data
    head.eval()
    logits = torch.cat([head(chunk[:, :512].to('mps'), chunk[:, 512:].to('mps')).cpu()
                        for chunk in x.split(256)])
    probabilities = logits.softmax(-1)[:, 1]
    predictions = logits.argmax(-1)
    cells = []
    for speed in SPEEDS:
        for angle in ANGLES:
            ix = torch.tensor([i for i, m in enumerate(metadata) if m['speed'] == speed and m['angle'] == angle])
            labels, pred, probs = y[ix], predictions[ix], probabilities[ix]
            positive, negative = probs[labels == 1], probs[labels == 0]
            delta = positive[:, None] - negative[None, :]
            auc = float((delta.gt(0).float() + .5 * delta.eq(0)).mean())
            sensitivity = float((pred[labels == 1] == 1).float().mean())
            specificity = float((pred[labels == 0] == 0).float().mean())
            cells.append(dict(speed=speed, angle=angle, n=len(ix), auc=auc,
                              balanced_accuracy=.5 * (sensitivity + specificity),
                              sensitivity=sensitivity, specificity=specificity,
                              loss=float(F.cross_entropy(logits[ix], labels))))
    return dict(n=len(y), loss=float(F.cross_entropy(logits, y)), cells=cells,
                mean_auc=sum(c['auc'] for c in cells) / 6,
                mean_balanced_accuracy=sum(c['balanced_accuracy'] for c in cells) / 6)


def main(run):
    run = Path(run).resolve()
    if any((run / name).exists() for name in ('budget.json', 'best.pt', 'latest.pt', 'report.json')):
        raise RuntimeError('Existing attempt; choose a new run directory')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    run.mkdir(parents=True, exist_ok=True)
    lock = Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    deadline = started + 28800
    atomic_json(run / 'budget.json', dict(cap_started=started, hard_deadline=deadline,
                deadline=deadline - 120, wall_cap_seconds=28800))
    atomic_json(run / 'activation.json', dict(supervisor_pid=os.getpid(), worker_pid=os.getpid()))
    source = Path(__file__).resolve().parents[1] / 'VariationalMotionPredictor/LocalRuntime/attempt01/run/best.pt'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    encoder = FrozenMotionEncoder(source).to('mps').eval()
    torch.manual_seed(185211)
    torch.mps.manual_seed(185211)
    head = ChangeFFN().to('mps')
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    shuffle = torch.Generator().manual_seed(185212)
    config = dict(encoder_checkpoint=str(source), encoder_sha256=source_hash,
                  encoder_step=encoder.source_step, frozen_encoder=True,
                  classifier='concat1024 -> LayerNorm -> 256 GELU -> 64 GELU -> 2 logits',
                  classifier_parameters=sum(p.numel() for p in head.parameters()),
                  batch_size=64, pool_size=1000, pool_epochs=2, target_updates=10240,
                  learning_rate=1e-3, precision='fp32', cpu_threads=2,
                  validation_samples=768, final_test_samples=3072, hard_deadline=deadline)
    atomic_json(run / 'config.json', config)
    state = dict(step=0, presentations=0, unique_trials=0, best_step=None, best_key=None)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

    def status(phase, **extra):
        atomic_json(run / 'live_status.json', dict(phase=phase, pid=os.getpid(),
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    elapsed_seconds=time.time() - started, **state, **extra))

    def save(best=False):
        path = run / 'latest.pt'
        payload = cpu_tree(dict(head=head.state_dict(), optimizer=optimizer.state_dict(),
                               state=state, config=config, torch_rng=torch.get_rng_state(),
                               mps_rng=torch.mps.get_rng_state(), shuffle_rng=shuffle.get_state()))
        tmp = run / 'latest.pt.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)
        atomic_json(run / 'latest_checkpoint.json', dict(step=state['step'], path=str(path),
                    optimizer_states=len(payload['optimizer']['state'])))
        if best:
            temp = run / 'best.pt.tmp'
            os.link(path, temp)
            os.replace(temp, run / 'best.pt')

    validation = None
    latest_validation = None
    reason = 'planned_complete'
    try:
        while state['step'] < 10240:
            if stopped[0] or time.time() >= deadline - 120:
                reason = 'signal' if stopped[0] else 'wall_cap'
                break
            status('encoding_training_pool')
            data = encode_set(encoder, 'train', state['unique_trials'], 1000, run, deadline - 120)
            state['unique_trials'] += 1000
            x, y, _ = data
            for epoch in range(2):
                order = torch.randperm(1000, generator=shuffle)
                for indices in order.split(64):
                    if stopped[0] or time.time() >= deadline - 120:
                        break
                    head.train()
                    batch = x[indices].to('mps')
                    labels = y[indices].to('mps')
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(batch[:, :512], batch[:, 512:]), labels)
                    loss.backward()
                    if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                        raise FloatingPointError('Nonfinite/missing classifier gradients')
                    optimizer.step()
                    state['step'] += 1
                    state['presentations'] += len(indices)
                    row = dict(**state, loss=float(loss.detach()), epoch=epoch)
                    with (run / 'progress.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                    status('training', train_loss=row['loss'], latest_validation=latest_validation)
                    if state['step'] == 1 or state['step'] % 256 == 0:
                        save()
                if stopped[0] or time.time() >= deadline - 120:
                    break
            if state['step'] == 32 or state['step'] % 512 == 0 or state['step'] >= 10240:
                status('validation_encoding' if validation is None else 'validation')
                if validation is None:
                    validation = encode_set(encoder, 'val', 0, 768, run, deadline - 120)
                latest_validation = dict(step=state['step'], **evaluate(head, validation))
                atomic_json(run / f"validation_{state['step']:06d}.json", latest_validation)
                key = (latest_validation['mean_auc'], latest_validation['mean_balanced_accuracy'])
                better = state['best_key'] is None or key > tuple(state['best_key'])
                if better:
                    state.update(best_key=key, best_step=state['step'])
                save(best=better)
        save()
        final = None
        if not stopped[0] and state['best_step'] is not None:
            head.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=False)['head'])
            status('final_test_encoding')
            test = encode_set(encoder, 'test', 0, 3072, run, deadline - 15)
            final = evaluate(head, test)
            atomic_json(run / 'final_test.json', final)
        assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
        atomic_json(run / 'report.json', dict(stop_reason=reason, **state, final=final,
                    encoder_frozen_verified=True, config=config))
        status('finished', stop_reason=reason, final=final)
        atomic_json(run / 'local_supervisor_result.json', dict(status='complete', **state))
    except Exception as exc:
        atomic_json(run / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        if state['step']:
            save()
        status('failed', error=str(exc))
        atomic_json(run / 'local_supervisor_result.json', dict(status='failed', error=str(exc)))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
