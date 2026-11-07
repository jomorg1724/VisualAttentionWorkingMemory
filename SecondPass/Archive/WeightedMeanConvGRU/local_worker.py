"""Fresh bounded local MPS adapter; the cloud worker and model stay unchanged."""
import argparse
import json
from pathlib import Path
import torch
import types

from SecondPass.DelayedFrameGRU import worker as local_core
from SecondPass.StructuredMotionRViT import worker as local_replay
from . import worker as replay
from .model import WeightedMeanConvGRU

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'SecondPass.WeightedMeanConvGRU.local_worker'
VERSION = replay.VERSION
PROTOCOL = VERSION + '_fresh_pool1000_epoch10_local_v1'
TARGET = replay.TARGET
EXPECTED_PARAMETERS = replay.EXPECTED_PARAMETERS
EXPECTED_TENSORS = replay.EXPECTED_TENSORS
TASK, TASKS, CELLS = replay.TASK, replay.TASKS, replay.CELLS
INIT_SEED = replay.INIT_SEED
clone = replay.clone
atomic_json, digest, load_verified = replay.atomic_json, replay.digest, replay.load_verified
validate_budget, verify_sources = local_core.validate_budget, local_core.verify_sources
validation_steps_for = replay.validation_steps_for


def sync(device):
    if str(device) == 'mps':
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def provenance():
    return dict(replay.provenance(), execution_placement='local', mps_seed=INIT_SEED,
                platform_difference='Apple MPS with microbatch4; same FP32 model, effectivebatch32 and stimuli')


_adapter = types.SimpleNamespace(**dict(vars(replay.base), sync=sync))
_verify = clone(replay.verify_progress, provenance=provenance)


def verify_progress(directory):
    result = _verify(directory)
    initial = load_verified(json.loads((Path(directory) / 'initial_checkpoint.json').read_text()))
    latest = load_verified(result['checkpoint'])
    if initial['state']['config']['device'] == 'mps':
        for checkpoint in (initial, latest):
            assert torch.is_tensor(checkpoint['rng'].get('mps')) and checkpoint['rng']['mps'].numel() > 0
        result['persisted_mps_rng'] = True
    atomic_json(Path(directory) / 'persisted_progress_verification.json', result)
    return result


class Session(replay.Session):
    def __init__(self, directory, config):
        if config['device'] == 'mps':
            torch.mps.manual_seed(INIT_SEED)
        super().__init__(directory, config)

    checkpoint = clone(local_core.Session.checkpoint, DelayedFrameGRU=WeightedMeanConvGRU,
                       parameter_count=lambda: EXPECTED_PARAMETERS, provenance=provenance,
                       verify_progress=verify_progress, VERSION=VERSION)
    train_update = clone(replay.Session.train_update, base=_adapter)


_evaluate = clone(replay._evaluate, sync=sync)


def evaluate(model, split, count, microbatch, device, output, deadline):
    # Production and profiling use the same local evaluation microbatch4.
    with torch.no_grad():
        return _evaluate(model, split, count, 4, device, output, deadline)


profile = clone(local_replay.profile, Session=Session, verify_sources=verify_sources,
                verify_progress=verify_progress, evaluate=evaluate, VERSION=VERSION)
measured_plan = clone(local_replay.measured_plan, TARGET=TARGET, replay=replay,
                      validate_budget=validate_budget, validation_steps_for=validation_steps_for)
_run = clone(replay._run, Session=Session, provenance=provenance,
             verify_progress=verify_progress, PROTOCOL=PROTOCOL, VERSION=VERSION,
             evaluate=evaluate, validate_budget=validate_budget)
run = clone(replay.run, _run=_run, provenance=provenance)
completion = replay.completion
supervise = clone(replay.base.supervise, MODULE=MODULE, completion=completion,
                  validate_budget=validate_budget)


def config_for(budget):
    validate_budget(budget, budget)
    manifest_path = ROOT / 'runtime_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    hashes = {str(Path(path) if Path(path).is_absolute() else ROOT / path): sha
              for path, sha in manifest['source_hashes'].items()}
    hashes[str(manifest_path)] = digest(manifest_path)
    config = dict(budget, protocol=PROTOCOL, device='mps', execution_placement='local',
                  checkpoint_encoder=True, effective_batch=32, microbatch=4, eval_microbatch=4,
                  val_n=100, test_n=200, checkpoint_every=100, cpu_threads=2,
                  source_hashes=hashes, runtime_root=str(ROOT), initialization=provenance(),
                  all_trainable=True, lr=1e-4, betas=[.9, .999], eps=1e-8, weight_decay=0,
                  clipping=None, precision='fp32', bptt='full', tf32=False,
                  training_policy='1000 fresh single-stimulus movies, ten shuffled epochs, then new pool')
    verify_sources(config)
    return config


prepare = clone(replay.base.prepare, config_for=config_for, validate_budget=validate_budget)
pin = clone(replay.base.pin, config_for=config_for, measured_plan=measured_plan)
local_supervise = clone(local_replay.local_supervise, config_for=config_for,
                        measured_plan=measured_plan, supervise=supervise,
                        completion=completion, validate_budget=validate_budget)


def main():
    replay.base.cpu_setup()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['local-supervise', 'prepare', 'profile', 'pin', 'run',
                                         'verify', 'supervise-profile', 'supervise-run'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--budget', type=Path)
    parser.add_argument('--cap-start', type=float)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == 'prepare':
        if args.budget is None:
            parser.error('--budget required')
        return prepare(directory, json.loads(args.budget.read_text()))
    if args.mode == 'pin':
        print(json.dumps(pin(directory)))
        return
    if args.mode == 'verify':
        print(json.dumps(verify_progress(directory)))
        return
    if args.mode.startswith('supervise-'):
        raise SystemExit(supervise(directory, args.mode.split('-', 1)[1])['returncode'])
    if args.mode == 'local-supervise':
        result = local_supervise(directory, args.cap_start)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'complete' else 2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for this authorized local attempt')
    import fcntl
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'profile':
            profile(directory)
        else:
            raise SystemExit(run(directory))


if __name__ == '__main__':
    main()
