"""Local-only bounded idle-sleep assertion for queued cloud retrieval."""
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parent
pred=ROOT.parents[1]/'SequenceKDA3'/'CloudRuntime'
end=json.loads((pred/'budget.json').read_text())['hard_deadline']+3600+28800+120
awake=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())])
try:
    while time.time()<end:
        path=ROOT/'queue_status.json'
        if path.exists():
            state=json.loads(path.read_text())['state']
            if state in ('blocked','deployment_failed','predecessor_cleanup_unresolved'):break
        if (ROOT/'mirror_result.json').exists():break
        if (ROOT/'budget.json').exists():
            end=min(end,json.loads((ROOT/'budget.json').read_text())['hard_deadline']+120)
        time.sleep(30)
finally:
    awake.terminate()
    awake.wait()
