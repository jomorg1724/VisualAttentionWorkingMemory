"""Source-only hash-verified local launchd deployment; never copies checkpoints."""
import hashlib,json,os,plistlib,shutil,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]
BASE=Path('/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01')
PY='/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python'
MODULE='SecondPass.SpatialReadout.SpatialConsolidation.KrauzlisOnly.worker'

def main():
    assert not BASE.exists(), 'Never overwrite runtime'
    runtime=BASE/'repo'; runtime.mkdir(parents=True)
    hashes={}
    for top in ('PreAttentiveVision','WorkingMemory','SecondPass'):
        for root,dirs,files in os.walk(REPO/top):
            dirs[:]=[d for d in dirs if d not in ('artifacts','runs','data','__pycache__','.pytest_cache','.git','locked_source','node_modules','.venv')]
            for name in files:
                src=Path(root)/name
                if src.suffix not in ('.py','.json'): continue
                dst=runtime/src.relative_to(REPO); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
                digest=hashlib.sha256(src.read_bytes()).hexdigest()
                assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
                hashes[str(dst)]=digest
    (runtime/'runtime_manifest.json').write_text(json.dumps(dict(source_repository=str(REPO),source_hashes=hashes,checkpoint_inputs=[]),indent=2))
    run=BASE/'run'; run.mkdir()
    for mode in ('probe','supervise'):
        label='org.vawm.krauzlis-wholemodel-fresh01'+('-probe' if mode=='probe' else '')
        directory=BASE/'probe' if mode=='probe' else run; directory.mkdir(exist_ok=True)
        spec=dict(Label=label,ProgramArguments=[PY,'-u','-m',MODULE,mode,str(directory)],WorkingDirectory=str(runtime),
            RunAtLoad=True,KeepAlive=False,ProcessType='Interactive',
            EnvironmentVariables={k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            StandardOutPath=str(directory/'launchd.stdout.log'),StandardErrorPath=str(directory/'launchd.stderr.log'))
        path=BASE/(mode+'.plist'); path.write_bytes(plistlib.dumps(spec))
    print(json.dumps(dict(runtime=str(runtime),run=str(run),sources=len(hashes),plists=[str(BASE/'probe.plist'),str(BASE/'supervise.plist')]),indent=2))
if __name__=='__main__': main()
