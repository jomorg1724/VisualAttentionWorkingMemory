"""Explicit user stop, keeping SSH available until verified retrieval."""
import json,os,signal,time
from pathlib import Path
run=Path('/workspace/vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01')
guard=json.loads((run/'guard.json').read_text())
cmd=Path(f"/proc/{guard['pid']}/cmdline")
if cmd.exists():
    assert b'cloud_control.py' in cmd.read_bytes() and b'guard' in cmd.read_bytes()
    os.kill(guard['pid'],signal.SIGTERM)
(run/'manual_recovery.json').write_text(json.dumps(dict(time=time.time(),reason='User requested download and deletion',guard_disabled_for_transfer=True)))
(run/'training/stop_requested.json').write_text(json.dumps(dict(reason='user_requested_download_and_delete',time=time.time())))
print('Clean stop requested; compute held temporarily for retrieval')
print((run/'training/status.json').read_text())
