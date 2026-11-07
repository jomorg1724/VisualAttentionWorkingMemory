"""Explicit local launch only; the queue itself does not activate training."""
import json
import os
from pathlib import Path
import plistlib
import subprocess


def main():
    root=Path(__file__).resolve().parents[2]
    base=Path(__file__).resolve().parent/'LocalRuntime'
    run=base/'run';run.mkdir(exist_ok=False)
    guard=base/'guard.py'
    source=(root/'SecondPass/VariationalMotionPredictor/LocalRuntime/guard.py').read_text()
    guard.write_text(source.replace("MODULE = 'SecondPass.VariationalMotionPredictor.worker'",
        "MODULE = 'SecondPass.AngularContrastiveMotion.'").replace('== 1200','== 28800'))
    python='/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python'
    label='org.vawm.angular-contrastive-motion-local01'
    for mode,name,command in [('guard',label+'-guard',[python,'-u',str(guard),str(run)]),
        ('train',label,[python,'-u','-m','SecondPass.AngularContrastiveMotion.train',str(run)])]:
        body=dict(Label=name,ProgramArguments=command,WorkingDirectory=str(root),RunAtLoad=True,KeepAlive=False,
            ProcessType='Interactive',EnvironmentVariables={k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            StandardOutPath=str(run/(mode+'.stdout.log')),StandardErrorPath=str(run/(mode+'.stderr.log')))
        path=base/(mode+'.plist');path.write_bytes(plistlib.dumps(body))
        subprocess.run(['launchctl','bootstrap','gui/'+str(os.getuid()),str(path)],check=True)
    (base/'queue.json').write_text(json.dumps(dict(status='activated',run=str(run)),indent=2))


if __name__=='__main__':main()
