"""Substantial, explicitly requested continuation of the local predictor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from . import worker as w


def main(directory):
    w.setup()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required')
    directory = Path(directory).resolve()
    lock = Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    original_initialization = (directory / 'initialization.json').read_text()
    start_step = saved['state']['step']
    started = time.time()
    budget = dict(cap_started=started, hard_deadline=started + 28800,
                  deadline=started + 28500, wall_cap_seconds=28800)
    config = dict(budget, device='mps', effective_batch=32, microbatch=4,
                  cpu_threads=2, source_hashes={}, lr=1e-4, clipping=None,
                  max_steps=start_step + 49984, resumed_from_step=start_step,
                  final_test_index_offset=3000000)
    w.atomic_json(directory / 'budget.json', budget)
    w.atomic_json(directory / 'activation.json', dict(supervisor_pid=os.getpid(),
                  worker_pid=os.getpid(), **budget))
    w.atomic_json(directory / 'config.json', config)
    session = w.Session(directory, config)
    (directory / 'initialization.json').write_text(original_initialization)
    session.model.load_state_dict(saved['model'])
    session.optimizer.load_state_dict(saved['optimizer'])
    session.pool.load_state_dict(saved['pool'])
    session.state = saved['state']
    session.state['config'] = config
    torch.set_rng_state(saved['rng']['cpu'])
    torch.mps.set_rng_state(saved['rng']['mps'])
    np.random.set_state(saved['rng']['numpy'])
    random.setstate(saved['rng']['python'])
    del saved
    w.atomic_json(directory / 'continuation.json', dict(authorized_continuation=True,
                  resumed_from_step=start_step, whole_model_optimizer_rng_pool_restored=True,
                  target_additional_updates=49984, **budget))
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    # Earlier pilot tests remain exploratory; the new final test uses fresh indices.
    original_generate = w.generate
    w.generate = lambda index, split: original_generate(
        index + (config['final_test_index_offset'] if split == 'test' else 0), split)
    reason = 'planned_complete'
    final = None
    failure = None

    def validate():
        session.status('validation')
        result = w.evaluate(session.model, 'val', 64, session.device, directory,
                            f"validation_{session.state['step']:06d}",
                            budget['deadline'] - 60, visuals=True)
        state = session.state
        key = result['balanced_model_mse']
        state['selection_history'].append(dict(step=state['step'], complete=result['complete'],
                                              balanced_model_mse=key))
        improved = result['complete'] and (state['best_key'] is None or key < state['best_key'])
        if improved:
            state.update(best_key=key, best_step=state['step'])
        session.checkpoint()
        if improved:
            session.best()

    try:
        while session.state['step'] < config['max_steps']:
            if stopped[0]:
                reason = 'signal'
                break
            if time.time() >= budget['deadline'] - 120:
                reason = 'wall_budget'
                break
            session.update()
            step = session.state['step']
            if step == start_step + 1 or step % 250 == 0:
                session.checkpoint()
            if step == start_step + 100 or step % 1000 == 0:
                validate()
        if not stopped[0]:
            if session.state['selection_history'][-1]['step'] != session.state['step']:
                validate()
            session.checkpoint()
            selected = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
            session.model.load_state_dict(selected['model'])
            del selected
            session.status('final_test_best')
            final = w.evaluate(session.model, 'test', 256, session.device, directory,
                               'continuation_test_best', budget['deadline'], visuals=True)
        else:
            session.checkpoint()
    except Exception as exc:
        reason = 'worker_error'
        failure = dict(type=type(exc).__name__, message=str(exc))
        w.atomic_json(directory / 'failure.json', failure)
        if session.state['step'] > start_step:
            session.checkpoint()
    report = dict(stop_reason=reason, failure=failure, actual_updates=session.state['step'],
                  additional_updates=session.state['step'] - start_step,
                  presentations=session.state['episodes'],
                  unique_movies_generated=session.state['unique_episodes_generated'],
                  selected_step=session.state['best_step'], final=final,
                  final_coverage_complete=bool(final and final['complete']), config=config)
    w.atomic_json(directory / 'continuation_report.json', report)
    session.status('finished', stop_reason=reason)
    summary = dict(status='complete' if failure is None and final and final['complete'] else 'incomplete',
                   actual_updates=session.state['step'], **budget)
    w.atomic_json(directory / 'local_supervisor_result.json', summary)
    lock.close()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
