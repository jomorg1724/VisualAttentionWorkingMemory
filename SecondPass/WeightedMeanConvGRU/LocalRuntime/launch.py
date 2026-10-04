"""Source-only launchd installation for one fresh local MPS training run."""
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
BASE = Path('/Users/jonathanmorgan/VAWMRuntime/weighted_mean_conv_gru_local01')
PYTHON = '/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python'
MODULE = 'SecondPass.WeightedMeanConvGRU.local_worker'
LABEL = 'org.vawm.weighted-mean-conv-gru-local01'


def main():
    sys.path.insert(0, str(REPO))
    from SecondPass.WeightedMeanConvGRU import bundle
    from SecondPass.WeightedMeanConvGRU import local_worker as worker
    cap_start = None
    BASE.mkdir(exist_ok=False)
    runtime = BASE / 'repo'
    runtime.mkdir()
    hashes = {}
    for relative, source in bundle.dependency_sources().items():
        assert source.suffix not in ('.pt', '.pth', '.ckpt')
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(target)] = worker.digest(target)
        target.chmod(0o444)
    (runtime / 'runtime_manifest.json').write_text(json.dumps(dict(
        source_repository=str(REPO), source_hashes=hashes, checkpoint_inputs=[]), indent=2))
    (runtime / 'runtime_manifest.json').chmod(0o444)
    guard = BASE / 'guard.py'
    shutil.copy2(Path(__file__).with_name('guard.py'), guard)
    guard.chmod(0o444)
    run = BASE / 'run'
    run.mkdir()
    jobs = [
        ('guard', LABEL + '-guard', [PYTHON, '-u', str(guard), str(run)]),
        ('supervise', LABEL, [PYTHON, '-u', '-m', MODULE, 'local-supervise', str(run)] + (['--cap-start', str(cap_start)] if cap_start is not None else [])),
    ]
    paths = []
    for mode, label, command in jobs:
        spec = dict(Label=label, ProgramArguments=command, WorkingDirectory=str(runtime),
            RunAtLoad=True, KeepAlive=False, ProcessType='Interactive',
            EnvironmentVariables={key:'2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
            StandardOutPath=str(run / (mode + '.stdout.log')),
            StandardErrorPath=str(run / (mode + '.stderr.log')))
        path = BASE / (mode + '.plist')
        path.write_bytes(plistlib.dumps(spec))
        paths.append(path)
    domain = 'gui/' + str(os.getuid())
    subprocess.run(['launchctl', 'bootstrap', domain, str(paths[0])], check=True)
    try:
        subprocess.run(['launchctl', 'bootstrap', domain, str(paths[1])], check=True)
    except BaseException:
        subprocess.run(['launchctl', 'bootout', domain, str(paths[0])], check=False)
        raise
    receipt = dict(runtime=str(runtime), run=str(run), sources=len(hashes),
        labels=[label for _, label, _ in jobs], launched=time.time(), device='mps',
        effective_batch=32, microbatch=4, initialization='fresh whole model',
        phase='automatic_native_profile_then_pinned_fresh_production')
    (BASE / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (Path(__file__).parent / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
