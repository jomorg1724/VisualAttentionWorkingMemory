"""Run exact-resume preflight, then explicitly launch the supervised continuation."""
import json,subprocess,time
from pathlib import Path

RUN=Path('/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/continuation02')
PY='/opt/doom-venv/bin/python'

def main():
    assert json.loads((RUN/'upload_verified.json').read_text())['all_hashes_match']
    guard=json.loads((RUN/'guard.json').read_text());assert guard['status']=='armed'
    assert 'cloud_control.py' in Path(f"/proc/{guard['pid']}/cmdline").read_bytes().decode()
    if not (RUN/'contract_tests_verified.json').exists():
        subprocess.run([PY,'-m','pytest','SecondPass/DeadlyCorridor/test_contract.py','-q'],cwd='/workspace/vawm',check=True)
        (RUN/'contract_tests_verified.json').write_text(json.dumps(dict(passed=4,time=time.time())))
    subprocess.run([PY,'-u','-m','SecondPass.DeadlyCorridor.CloudRuntime.continuation02.prepare_resume'],cwd='/workspace/vawm',check=True)
    assert not (RUN/'worker.json').exists()
    with (RUN/'supervisor.log').open('w') as f:
        p=subprocess.Popen([PY,'-u',str(RUN/'cloud_control.py'),'supervise'],cwd='/workspace/vawm',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(3);assert p.poll() is None,p.returncode
    print('Supervisor',p.pid,flush=True);print((RUN/'worker.json').read_text(),flush=True)

if __name__=='__main__':main()
