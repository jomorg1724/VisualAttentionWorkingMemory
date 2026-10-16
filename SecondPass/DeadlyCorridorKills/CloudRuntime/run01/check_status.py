import json,subprocess
from pathlib import Path
run=Path('/workspace/vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01')
for name in ['guard.json','preparation_verified.json','worker.json','supervisor_exit.json','training/status.json','training/exit.json']:
    p=run/name
    if p.exists():print(name,p.read_text())
for name in ['setup.log','activation.log','worker.log']:
    p=run/name
    if p.exists():print(name,p.read_text()[-2500:])
print(subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout)
