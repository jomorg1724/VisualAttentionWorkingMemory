"""One fresh local weighted-frame RViT activation, called by the queue owner."""
import argparse, json, os, plistlib, shutil, subprocess, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
BASE=Path('/Users/jonathanmorgan/VAWMRuntime/weighted_mean_rvit_local01')
PYTHON='/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python'
MODULE='SecondPass.WeightedMeanRViT.worker'
LABEL='org.vawm.weighted-mean-rvit-local01'


def snapshot(base=BASE):
    sys.path.insert(0,str(REPO))
    from SecondPass.ThreeFrameConvVAE import bundle as source_bundle
    from SecondPass.SingleStimulusRViT.worker import clone
    from SecondPass.WeightedMeanRViT import worker
    sources=clone(source_bundle.dependency_sources,__file__=str(REPO/'SecondPass/WeightedMeanRViT/worker.py'))()
    base.mkdir(exist_ok=False); runtime=base/'repo'; runtime.mkdir(); hashes={}
    for relative,source in sources.items():
        assert source.suffix not in ('.pt','.pth','.ckpt')
        target=runtime/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        hashes[str(target)]=worker.digest(target); target.chmod(0o444)
    manifest=runtime/'runtime_manifest.json'
    manifest.write_text(json.dumps(dict(source_repository=str(REPO),source_hashes=hashes,checkpoint_inputs=[],
        pretrained_weights=False,classification_namespaces=dict(train=worker.TRAIN_NAMESPACE,val=worker.VAL_NAMESPACE,test=worker.FINAL_NAMESPACE)),indent=2))
    manifest.chmod(0o444)
    guard=base/'guard.py'; shutil.copy2(Path(__file__).with_name('guard.py'),guard); guard.chmod(0o444)
    (base/'run').mkdir()
    receipt=dict(runtime=str(runtime),run=str(base/'run'),sources=len(hashes),prepared=time.time(),
        initialization='whole model fresh; no VAE or other checkpoint reference',budget_started=False)
    (base/'snapshot.json').write_text(json.dumps(receipt,indent=2)+'\n'); return receipt


def activate(base=BASE):
    receipt=json.loads((base/'snapshot.json').read_text()); runtime=Path(receipt['runtime']); run=Path(receipt['run'])
    assert not (run/'budget.json').exists(), 'One activation, no cap renewal'
    jobs=[('guard',LABEL+'-guard',[PYTHON,'-u',str(base/'guard.py'),str(run)]),
        ('supervise',LABEL,[PYTHON,'-u','-m',MODULE,'local-supervise',str(run)])]
    paths=[]
    for mode,label,command in jobs:
        spec=dict(Label=label,ProgramArguments=command,WorkingDirectory=str(runtime),RunAtLoad=True,KeepAlive=False,ProcessType='Interactive',
            EnvironmentVariables={key:'2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            StandardOutPath=str(run/(mode+'.stdout.log')),StandardErrorPath=str(run/(mode+'.stderr.log')))
        path=base/(mode+'.plist'); path.write_bytes(plistlib.dumps(spec)); paths.append(path)
    domain='gui/'+str(os.getuid()); subprocess.run(['launchctl','bootstrap',domain,str(paths[0])],check=True)
    try: subprocess.run(['launchctl','bootstrap',domain,str(paths[1])],check=True)
    except BaseException:
        subprocess.run(['launchctl','bootout',domain,str(paths[0])],check=False); raise
    receipt.update(launched=time.time(),labels=[label for _,label,_ in jobs],module=MODULE,device='mps',effective_batch=32,microbatch=1,
        phase='disposable native profile then pinned fresh production; only best/latest files')
    (base/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2)); return receipt


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--snapshot-only',action='store_true'); parser.add_argument('--activate-existing',action='store_true')
    parser.add_argument('--base',type=Path,default=BASE); args=parser.parse_args()
    if args.activate_existing: return activate(args.base)
    receipt=snapshot(args.base)
    if args.snapshot_only: print(json.dumps(receipt,indent=2))
    else: activate(args.base)
if __name__=='__main__': main()
