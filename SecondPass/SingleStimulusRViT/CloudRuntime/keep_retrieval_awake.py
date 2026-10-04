import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
end=json.loads((root/'budget.json').read_text())['hard_deadline']+120
p=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())])
try:
    while time.time()<end:
        if (root/'mirror_result.json').exists() or (root/'deployment_error.json').exists():break
        time.sleep(30)
finally:p.terminate();p.wait()
