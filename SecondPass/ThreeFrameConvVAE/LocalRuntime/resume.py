"""Launch one storage-bounded recovery under the original local deadline."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[3]
BASE=Path('/Users/jonathanmorgan/VAWMRuntime/three_frame_conv_vae_local01')
PYTHON='/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python'
MODULE='SecondPass.ThreeFrameConvVAE.resume_worker'
LABEL='org.vawm.three-frame-conv-vae-local01-resume01'


def prepare(base=BASE,restored_step=5600):
    sys.path.insert(0,str(REPO))
    from SecondPass.ThreeFrameConvVAE import worker as w
    from SecondPass.ThreeFrameConvVAE.resume_worker import tensor_digest
    old=base/'run'; directory=base/'resume01'; directory.mkdir(exist_ok=False)
    budget=json.loads((old/'budget.json').read_text())
    cfg=json.loads((old/'config.json').read_text()); w.validate_budget(budget,cfg)
    # The original dependency snapshot remains immutable. Copy source only to a new overlay.
    runtime=directory/'repo'; runtime.mkdir()
    old_manifest=json.loads((base/'repo/runtime_manifest.json').read_text())
    new_hashes={}
    for source,expected in old_manifest['source_hashes'].items():
        source=Path(source); assert w.digest(source)==expected
        target=runtime/source.relative_to(base/'repo'); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target); target.chmod(0o444)
        new_hashes[str(target)]=expected
    for rel in ('SecondPass/ThreeFrameConvVAE/resume_worker.py','SecondPass/ThreeFrameConvVAE/LocalRuntime/resume.py'):
        source=REPO/rel; target=runtime/rel; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target); target.chmod(0o444); new_hashes[str(target)]=w.digest(target)
    source=old/f'checkpoint_{restored_step:06d}.pt'
    receipt=dict(path=str(source),bytes=source.stat().st_size,sha256=w.digest(source),verified=True,step=restored_step)
    saved=w.load_verified(receipt)
    assert saved['state']['step']==saved['scheduler']['updates']==restored_step
    assert saved['state']['config']['max_steps']==cfg['max_steps']==9900
    assert saved['provenance']==w.provenance() and len(saved['optimizer']['state'])==126
    assert all(float(state['step'])==restored_step for state in saved['optimizer']['state'].values())
    progress=[json.loads(line) for line in (old/'progress.jsonl').read_text().splitlines()]
    committed=[row for row in progress if row['step']<=restored_step]
    discarded=[row for row in progress if row['step']>restored_step]
    assert len(committed)==restored_step and sum(row['episodes'] for row in committed)==saved['state']['episodes']
    (directory/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in committed))
    initial=json.loads((old/'initial_checkpoint.json').read_text())
    original_receipt=dict(receipt)
    latest=directory/'latest.pt'
    os.link(source,latest)
    assert latest.stat().st_size==receipt['bytes'] and w.digest(latest)==receipt['sha256']
    receipt['path']=str(latest)
    recovery=dict(restored_checkpoint=receipt,original_restored_checkpoint_metadata=original_receipt,
        historical_initial_checkpoint_metadata=initial,restored_step=restored_step,
        optimizer_names=saved['optimizer_names'],
        restored_parameter_hashes={name:tensor_digest(saved['model'][name]) for name in saved['optimizer_names']},
        historical_best_step=saved['state']['best_step'],historical_best_reconstruction=saved['state']['best_key'],
        historical_best_weights_available=False,checkpoint_policy='only latest.pt and best.pt',
        source_run=str(old),original_hard_deadline=budget['hard_deadline'],original_target=cfg['max_steps'],
        discarded_unsaved_attempt_updates=len(discarded),discarded_unsaved_attempt_presentations=sum(r['episodes'] for r in discarded),
        discarded_attempt_preserved_in=str(old/'progress.jsonl'),
        policy='exact saved model/Adam/RNG/streams/counters; original cap and target unchanged; unsaved work replayed',
        source_hash_migration=dict(original_source_hashes=cfg['source_hashes'],recovery_snapshot_hashes=new_hashes))
    cfg['source_hashes']=dict(cfg['source_hashes'],**new_hashes)
    cfg['recovery_from_step']=restored_step
    # Charge the remaining exact validation looks and paired final evaluations; no new profile.
    recent=committed[-330:]; percell={c:max(r['seconds'] for r in recent if r['cell']==c) for c in w.CELLS}
    remaining=cfg['max_steps']-restored_step; looks=sum(step>restored_step for step in cfg['validation_steps'])
    profile=json.loads((old/'profile/profile.json').read_text())
    generation=sum(r.get('pool_generation_seconds',0.) for r in profile['warmup_rows'])
    cfg['estimated_total_remaining_seconds']=1.25*(remaining*max(percell.values())+generation*(remaining//330+1))+looks*cfg['estimated_validation_seconds']+2*cfg['estimated_one_test_seconds']+900
    cfg['estimated_remaining_seconds']=cfg['estimated_total_remaining_seconds']
    cfg['runtime_root']=str(runtime)
    assert time.time()+cfg['estimated_total_remaining_seconds']<budget['deadline'], 'Remaining exposure no longer fits original cap'
    for name,value in [('budget.json',budget),('config.json',cfg),('allocation.json',json.loads((old/'allocation.json').read_text())),('recovery.json',recovery)]:
        with (directory/name).open('x') as stream: json.dump(value,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
        (directory/name).chmod(0o444)
    # Delete the obsolete filename only after verified latest and fsynced recovery metadata exist.
    source.unlink()
    return directory,runtime,recovery


def launch(directory,runtime):
    # Independent guard recognizes only the recovery module and retains the original deadline.
    guard=directory/'guard.py'
    text=Path(__file__).with_name('guard.py').read_text().replace("MODULE = 'SecondPass.ThreeFrameConvVAE.worker'", "MODULE = '"+MODULE+"'")
    assert "MODULE = '"+MODULE+"'" in text
    guard.write_text(text); guard.chmod(0o444)
    jobs=[('guard',LABEL+'-guard',[PYTHON,'-u',str(guard),str(directory)]),
        ('supervise',LABEL,[PYTHON,'-u','-m',MODULE,'local-supervise',str(directory)])]
    paths=[]
    for mode,label,command in jobs:
        spec=dict(Label=label,ProgramArguments=command,WorkingDirectory=str(runtime),RunAtLoad=True,KeepAlive=False,ProcessType='Interactive',
            EnvironmentVariables={key:'2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            StandardOutPath=str(directory/(mode+'.stdout.log')),StandardErrorPath=str(directory/(mode+'.stderr.log')))
        path=directory/(mode+'.plist'); path.write_bytes(plistlib.dumps(spec)); paths.append(path)
    domain='gui/'+str(os.getuid()); subprocess.run(['launchctl','bootstrap',domain,str(paths[0])],check=True)
    try: subprocess.run(['launchctl','bootstrap',domain,str(paths[1])],check=True)
    except BaseException:
        subprocess.run(['launchctl','bootout',domain,str(paths[0])],check=False); raise
    receipt=dict(run=str(directory),runtime=str(runtime),module=MODULE,labels=[row[1] for row in jobs],launched=time.time(),
        initialization='exact checkpoint5600 recovery, not a fresh model',budget_renewed=False)
    (directory/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--base',type=Path,default=BASE); parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); directory,runtime,recovery=prepare(args.base)
    if args.prepare_only: print(json.dumps(dict(run=str(directory),runtime=str(runtime),recovery=recovery),indent=2))
    else: launch(directory,runtime)

if __name__=='__main__': main()
