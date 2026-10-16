import json,os,subprocess,time
from pathlib import Path
run=Path('/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/run01')
assert json.loads((run/'supervisor_exit.json').read_text())['exit_code']==1
assert not (run/'training/latest.pt').exists()
assert not (run/'training/updates.jsonl').exists()
os.kill(json.loads((run/'guard.json').read_text())['pid'],0)
archive=run/'failed_logging_attempt';archive.mkdir(exist_ok=False)
for name in ('training','worker.json','worker.log','supervisor_exit.json','supervise.log'):
 (run/name).rename(archive/name)
with (run/'supervise.log').open('w') as log:
 p=subprocess.Popen(['/workspace/doom-venv/bin/python','-u',str(run/'cloud_control.py'),'supervise'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/workspace/vawm')
time.sleep(2);assert p.poll() is None
print((run/'worker.json').read_text())
