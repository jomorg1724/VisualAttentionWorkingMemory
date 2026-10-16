import subprocess,json
from pathlib import Path
run=Path('/workspace/vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01')
assert not (run/'activation_started.json').exists()
with (run/'activation.log').open('w') as f:
    p=subprocess.Popen(['/opt/doom-venv/bin/python','-u','-m','SecondPass.DeadlyCorridorKills.CloudRuntime.run01.activate'],cwd='/workspace/vawm',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(run/'activation_started.json').write_text(json.dumps(dict(pid=p.pid)))
print('Activation',p.pid)
