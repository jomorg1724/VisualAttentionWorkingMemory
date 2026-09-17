import subprocess,time,json,sys,os
from pathlib import Path
p=Path(__file__).resolve().parent;start=time.time()
(p/'supervisor.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),start_epoch=start,deadline_epoch=start+1800)))
with (p/'supervisor.log').open('w') as log:
    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(p/'run.py')],stdout=log,stderr=subprocess.STDOUT)
    (p/'launch.json').write_text(json.dumps(dict(worker_pid=child.pid,supervisor_pid=os.getpid(),start_epoch=start,deadline_epoch=start+1800)))
    try:code=child.wait(timeout=1740);reason='child_exit'
    except subprocess.TimeoutExpired:child.kill();code=child.wait();reason='hard_cap_reserve60s'
(p/'exit.json').write_text(json.dumps(dict(exit_code=code,reason=reason,wall_seconds=time.time()-start,worker_pid=child.pid,worker_terminated=True),indent=2))
