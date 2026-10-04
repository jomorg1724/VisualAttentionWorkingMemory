"""CPU-only launchd ownership/deadline probe; never imports a model or trains."""
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time


def main():
    directory=Path(sys.argv[2]).resolve()
    if sys.argv[1]=='worker':
        from SecondPass.SpatialReadout.protocol import supervise
        result=supervise([sys.executable,'-c','import time; time.sleep(20)'],time.time()+1.,directory,'cpu_probe')
        (directory/'ownership.json').write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),result=result)))
        return
    from SecondPass.SpatialReadout.protocol import launchd_spec
    directory.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    label='org.vawm.spatialreadout.cpu-probe-'+str(os.getpid())
    target=f'gui/{os.getuid()}/{label}'
    spec=launchd_spec(label,directory,sys.executable,root)
    spec['ProgramArguments']=[sys.executable,'-u','-m','SecondPass.SpatialReadout.checks.launchd_probe','worker',str(directory)]
    plist=directory/'probe.plist'
    with plist.open('wb') as f: plistlib.dump(spec,f)
    subprocess.run(['/bin/launchctl','bootstrap',f'gui/{os.getuid()}',str(plist)],check=True)
    try:
        until=time.time()+30
        while time.time()<until and not (directory/'ownership.json').exists(): time.sleep(.2)
        result=json.loads((directory/'ownership.json').read_text())
        result['launchctl_print']=subprocess.check_output(['/bin/launchctl','print',target],text=True)
        assert result['ppid']==1 and result['result']['hard_cap_triggered']
        (directory/'verified.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        subprocess.run(['/bin/launchctl','bootout',target],check=True)
        absent=subprocess.run(['/bin/launchctl','print',target],capture_output=True,text=True)
        (directory/'cleanup.json').write_text(json.dumps(dict(returncode=absent.returncode,absent=absent.returncode!=0)))
        assert absent.returncode!=0


if __name__=='__main__': main()
