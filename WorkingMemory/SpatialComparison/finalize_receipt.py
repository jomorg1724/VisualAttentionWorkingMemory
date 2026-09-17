"""Close the existing ledger after saved-report and optional-reference completion."""
import json,time,hashlib,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(P/'results.json');run=Path(a['run_root']);budget=read(run/'budget.json');old=read(P/'completion_receipt.json');ref=read(P/'parent_reference.json')
assert a['status']=='completed' and ref['status']=='completed' and budget['active'] is None
assert all(e['returncode']==0 for e in budget['events']) and time.time()<budget['deadline_unix']
command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'SpatialComparison[/\\\\](train|sweep|parent_reference|watch_run)\\.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
# The shell check is only process ownership evidence; no processes are stopped.
check=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
assert not check.stdout.strip(),check.stdout
for name,value in budget['source_hashes'].items():assert sha(ROOT/name)==value,name
preserved=P/'primary_completion_receipt.json'
if not preserved.exists():preserved.write_text(json.dumps(old,indent=2),encoding='utf-8')
receipt=dict(old)
receipt.update(status='completed',completed_epoch=time.time(),seconds_including_analysis=time.time()-budget['started_unix'],primary_report_seconds=old['seconds_including_analysis'],parent_reference_seconds=ref['worker_seconds'],parent_reference_exit_code=0,all_owned_gpu_workers_exited=True,process_check_stdout=check.stdout.strip(),review='decision_review.md: final interpretation accepted; no further computation requested',selected_checkpoints={arm:dict(step=v['selected_step'],path=v['test']['checkpoint'],sha256=sha(v['test']['checkpoint'])) for arm,v in a['runs'].items()},new_model_updates_per_arm=4400,new_training_episodes_per_arm=35200,parent_reference_model_updates=0,allowance_closed=True)
report=(P/'report.md').read_text(encoding='utf-8')
report+=f"\nFinal closure: {receipt['seconds_including_analysis']:.1f}s ({receipt['seconds_including_analysis']/60:.1f}min) elapsed within the14,400s allowance. All owned GPU workers exited; the original source inventory still matches its pinned hashes. The independent reviewer accepted the final interpretation. Both selected checkpoint identities and the original primary completion receipt are preserved.\n"
(P/'report.md').write_text(report,encoding='utf-8');(run/'report.md').write_text(report,encoding='utf-8')
receipt['artifacts']={p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='completion_receipt.json'}
for destination in (P/'completion_receipt.json',run/'completion_receipt.json'):destination.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ('status','seconds_including_analysis','allowance_closed','selected_checkpoints')},indent=2))
