"""Single diagnostic child; absolute local600second cap."""
import subprocess,time,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent;start=time.time()
with (p/'supervisor.log').open('w') as log:
    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(p/'run.py')],stdout=log,stderr=subprocess.STDOUT)
    (p/'supervisor.json').write_text(json.dumps(dict(supervisor_pid=__import__('os').getpid(),worker_pid=child.pid,start_epoch=start,deadline_epoch=start+600)))
    try:code=child.wait(timeout=580);reason='child_exit'
    except subprocess.TimeoutExpired:
        child.kill();code=child.wait();reason='hard_cap_reserve20seconds'
(p/'exit.json').write_text(json.dumps(dict(exit_code=code,reason=reason,wall_seconds=time.time()-start,worker_pid=child.pid,worker_terminated=True),indent=2))
