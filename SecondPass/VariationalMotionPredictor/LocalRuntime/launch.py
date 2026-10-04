"""Snapshot source only and launch one bounded fresh local predictor."""
import ast,hashlib,json,os,plistlib,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent/'attempt01'
PYTHON='/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python'
MODULE='SecondPass.VariationalMotionPredictor.worker'
LABEL='org.vawm.variational-motion-predictor-local01'

def sources():
    local=REPO/'SecondPass/VariationalMotionPredictor'
    rows={str(p.relative_to(REPO)):p for p in local.iterdir() if p.suffix=='.py'};todo=list(rows);seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name);package=list(Path(name).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(rows[name].read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[n.name.split('.') for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                base+=node.module.split('.') if node.module else []
                modules=[base]+[base+n.name.split('.') for n in node.names if n.name!='*']
            for parts in modules:
                if not parts or parts[0] not in ('SecondPass','WorkingMemory','PreAttentiveVision'):continue
                paths=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']+[Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
                for path in paths:
                    key=str(path)
                    if key not in rows and (REPO/path).is_file():rows[key]=REPO/path;todo.append(key)
    return rows

def main():
    BASE.mkdir(exist_ok=False);runtime=BASE/'repo';runtime.mkdir();hashes={}
    for relative,source in sources().items():
        target=runtime/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        hashes[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest();target.chmod(0o444)
    manifest=runtime/'runtime_manifest.json';manifest.write_text(json.dumps(dict(source_hashes=hashes,checkpoint_inputs=[],
        snapshots='source only; whole model fresh',local_pilot_cap_seconds=1200),indent=2));manifest.chmod(0o444)
    guard=BASE/'guard.py';shutil.copy2(Path(__file__).with_name('guard.py'),guard)
    run=BASE/'run';run.mkdir();jobs=[('guard',LABEL+'-guard',[PYTHON,'-u',str(guard),str(run)]),('supervise',LABEL,[PYTHON,'-u','-m',MODULE,'local-supervise',str(run)])]
    paths=[]
    for mode,label,command in jobs:
        body=dict(Label=label,ProgramArguments=command,WorkingDirectory=str(runtime),RunAtLoad=True,KeepAlive=False,ProcessType='Interactive',
            EnvironmentVariables={k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            StandardOutPath=str(run/(mode+'.stdout.log')),StandardErrorPath=str(run/(mode+'.stderr.log')))
        path=BASE/(mode+'.plist');path.write_bytes(plistlib.dumps(body));paths.append(path)
    domain='gui/'+str(os.getuid());subprocess.run(['launchctl','bootstrap',domain,str(paths[0])],check=True)
    try:subprocess.run(['launchctl','bootstrap',domain,str(paths[1])],check=True)
    except BaseException:
        subprocess.run(['launchctl','bootout',domain,str(paths[0])],check=False);raise
    receipt=dict(runtime=str(runtime),run=str(run),sources=len(hashes),labels=[j[1] for j in jobs],launched=time.time(),
        device='mps',effective_batch=32,microbatch=4,cap_seconds=1200,initialization='whole fresh model/Adam/RNG/streams; no VAE weights',
        phase='three disposable native profile updates then measured pinned production; only best/latest checkpoints')
    (BASE/'launch.json').write_text(json.dumps(receipt,indent=2));Path(__file__).with_name('launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
