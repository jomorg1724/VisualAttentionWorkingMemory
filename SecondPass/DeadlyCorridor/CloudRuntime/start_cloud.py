import json,os,subprocess,time
from pathlib import Path
run=Path('/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/run01')
assert (run/'preflight_verified.json').exists()
assert not (run/'guard.json').exists() and not (run/'worker.json').exists()
python='/workspace/doom-venv/bin/python'
for mode in ('guard','supervise'):
 with (run/(mode+'.log')).open('w') as log:
  p=subprocess.Popen([python,'-u',str(run/'cloud_control.py'),mode],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/workspace/vawm')
 time.sleep(2)
 assert p.poll() is None,(mode,p.returncode)
 print(mode,p.pid)
print((run/'guard.json').read_text());print((run/'worker.json').read_text())
