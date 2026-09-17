"""Finish saved-score reporting inside the already running experiment deadline."""
import os,time,json,sys,subprocess,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;last=None
def read(p):return json.loads(Path(p).read_text())
while True:
    agg=read(P/'results.json');run=Path(agg['run_root']);ledger=read(run/'budget.json');deadline=ledger['deadline_unix']
    status=(agg['status'],tuple((a,len(v.get('validation',[])),v.get('status')) for a,v in agg.get('runs',{}).items()))
    if status!=last:print(json.dumps(dict(status=status,elapsed=time.time()-ledger['started_unix'])),flush=True);last=status
    if agg['status'] in ('completed','failed') and (run/'exit.json').exists():break
    if time.time()>deadline:raise TimeoutError('Original GPU experiment deadline passed')
    time.sleep(10)
if agg['status']!='completed':raise RuntimeError(agg.get('error','Run failed'))
child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(P/'analyze.py')],stdout=sys.stdout,stderr=sys.stderr)
try:code=child.wait(timeout=max(.1,deadline-time.time()-20))
except subprocess.TimeoutExpired:child.kill();child.wait();raise
if code:raise RuntimeError('Saved-score reporting failed')
receipt=dict(status='completed',completed_epoch=time.time(),seconds_including_analysis=time.time()-ledger['started_unix'],deadline_epoch=deadline,allowance_seconds=14400,events=ledger['events'],all_gpu_workers_exited=True,analysis_exit_code=code,run_root=str(run),source_hashes=ledger['source_hashes'],artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name not in ('completion_receipt.json',)})
(P/'completion_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');(run/'completion_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(dict(status='report_completed',seconds=receipt['seconds_including_analysis'])),flush=True)
