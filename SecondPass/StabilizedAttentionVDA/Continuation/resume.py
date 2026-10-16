"""Resume the paused VDA experiment without replacing its original artifacts."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .. import train as base
from ..model import ExpandedSuite
from ..dataset import Stream

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(saved):
    for name, wanted in saved['config']['source_sha256'].items():
        if digest(ROOT / name) != wanted:
            raise ValueError(f'Frozen VDA source changed: {name}')
    prior = json.loads((ROOT.parent / 'StabilizedAttentionSuite/Analysis/run01/protocol.json').read_text())
    for name, wanted in prior['source_hashes'].items():
        if digest(REPO / name) != wanted:
            raise ValueError(f'Frozen dependency changed: {name}')
    if digest(base.PARENT) != saved['config']['parent_sha256']:
        raise ValueError('Original suite checkpoint changed')
    return prior['source_hashes']


def restore(saved, device='cpu'):
    model = ExpandedSuite()
    model.load_state_dict(saved['model'], strict=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    scheduler = base.Scheduler()
    state = saved['scheduler']
    scheduler.rng.setstate(state['rng'])
    scheduler.tasks = copy.deepcopy(state['tasks'])
    scheduler.cells = copy.deepcopy(state['cells'])
    stream = Stream('train')
    stream.load_state_dict(saved['stream'])
    # Restore last: constructors and device initialization may consume RNG draws.
    torch.set_rng_state(saved['torch_rng'].cpu())
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    return model, opt, scheduler, stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not 120 < args.wall_seconds <= 28800:
        raise ValueError('Finite local allocation must be between 120 seconds and 8 hours')
    if args.source.resolve() == args.run.resolve():
        raise ValueError('Preserve the paused run in a separate output directory')
    saved = torch.load(args.source / 'latest.pt', map_location='cpu', weights_only=False)
    sources = verify_sources(saved)
    pause = json.loads((args.source / 'pause_verified.json').read_text())
    original_hashes = {name: digest(args.source / f'{name}.pt') for name in ('best', 'latest')}
    for name, sha in original_hashes.items():
        if sha != pause['checkpoints'][name]['sha256']:
            raise ValueError(f'Paused {name} checkpoint changed')
    args.run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.wall_seconds
    lock = base.GPULock(REPO / 'LocalRuntime/local_gpu_worker.lock')
    try:
        base.configure(args.device)
        model, opt, scheduler, stream = restore(saved, args.device)
        start_step = step = saved['step']
        target = saved['config']['steps']
        best = saved['best']
        config = copy.deepcopy(saved['config'])
        config.update(started=started, deadline=deadline, resumed_from=str(args.source.resolve()),
                      resumed_step=start_step, original_checkpoint_sha256=original_hashes,
                      original_config=copy.deepcopy(saved['config']), dependency_sha256=sources,
                      continuation_source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
        base.atomic_json(args.run / 'config.json', config)
        shutil.copy2(args.source / 'best.pt', args.run / 'best.pt')

        def save(name='latest'):
            payload = base.cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(), stream=stream.state_dict(), step=step, best=best,
                config=config, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
            temp = args.run / f'{name}.pt.tmp'
            torch.save(payload, temp)
            base.replace_with_retry(temp, args.run / f'{name}.pt')

        def status(phase):
            base.atomic_json(args.run / 'status.json', dict(phase=phase, step=step, target=target,
                resumed_step=start_step, new_updates=step-start_step, elapsed=time.time()-started, deadline=deadline))

        # Check restored RNGs before any draws or checkpoint serialization.
        assert torch.equal(torch.get_rng_state(), saved['torch_rng'])
        if torch.device(args.device).type == 'cuda':
            assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(torch.cuda.get_rng_state_all(), saved['cuda_rng']))
        save()
        base.atomic_json(args.run / 'restoration_receipt.json', dict(step=step, rngs_restored=True,
            optimizer_states=len(opt.state), original_hashes=original_hashes, first_next_update=step+1))
        reason = 'planned_complete'
        try:
            while step < target:
                if (args.run / 'stop_requested.json').exists():
                    reason = 'user_stop'
                    break
                if time.time() > deadline - 120:
                    reason = 'time_cap'
                    break
                task, cell = scheduler.next()
                x, y, metadata = stream.batch(32, task, cell)
                metrics = base.update(model, opt, x, y, metadata, task, args.device)
                step += 1
                base.append_json(args.run / 'progress.jsonl', dict(step=step, task=task, cell=cell, **metrics))
                status('training')
                if step % 16 == 0:
                    save()
                if step % 320 == 0:
                    status('validating')
                    result = base.evaluate(model, args.device, 'val', 64, deadline-90)
                    base.atomic_json(args.run / f'validation_{step:06d}.json', result)
                    if result['equal_task_mean_auc'] > best:
                        best = result['equal_task_mean_auc']
                        save('best')
            save()
            if reason == 'planned_complete':
                status('final_validation')
                result = base.evaluate(model, args.device, 'val', 64, deadline-60)
                base.atomic_json(args.run / 'validation_final.json', result)
                if result['equal_task_mean_auc'] > best:
                    best = result['equal_task_mean_auc']
                    save('best')
                selected = copy.deepcopy(model)
                selected.load_state_dict(torch.load(args.run / 'best.pt', map_location=args.device, weights_only=False)['model'])
                status('final_test')
                result = base.evaluate(selected, args.device, 'test', 256, deadline)
                base.atomic_json(args.run / 'final_test.json', result)
        except TimeoutError:
            reason = 'time_cap'
            save()
        except BaseException as error:
            base.atomic_json(args.run / 'failure.json', dict(step=step, error=repr(error)))
            raise
        status('paused' if reason == 'user_stop' else reason)
        base.atomic_json(args.run / 'exit.json', dict(reason=reason, step=step,
            training_complete=reason == 'planned_complete',
            originals_unchanged=all(digest(args.source / f'{name}.pt') == sha for name, sha in original_hashes.items())))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
