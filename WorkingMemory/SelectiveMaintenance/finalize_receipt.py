"""Final closure of this local experiment; no model loading or new computation."""
import json,time,hashlib,subprocess,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(P/'results.json');run=Path(a['run_root']);b=read(run/'budget.json');e=read(run/'exit.json');old=read(P/'completion_receipt.json')
assert a['status']==e['status']=='completed' and b['active'] is None and time.time()<b['deadline_unix']
command=r"Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'SelectiveMaintenance[/\\](train|sweep|resume_supervisor|watch_run)\.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
check=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True);assert not check.stdout.strip(),check.stdout
for path,digest in b['source_hashes'].items():assert sha(ROOT/path)==digest,path
preserved=P/'primary_completion_receipt.json'
if not preserved.exists():shutil.copy2(P/'completion_receipt.json',preserved)
closure=dict(old,status='completed',completed_epoch=time.time(),seconds_including_analysis=time.time()-b['started_unix'],allowance_closed=True,all_owned_gpu_workers_exited=True,process_check_stdout=check.stdout.strip(),source_inventory_matches=True,supervisor_recovery=read(run/'recovery_receipt.json'),recovery_supervisor_sha256=sha(P/'resume_supervisor.py'),events=b['events'],selected_checkpoints={arm:dict(global_step=v['selected_step'],added_episodes=32000,path=v['test']['checkpoint'],sha256=sha(v['test']['checkpoint'])) for arm,v in a['runs'].items()},no_training_or_profile_repeated=True,training_updates_per_arm=4000,training_episodes_per_arm=32000,teaching_unchanged=True)
closure['worker_exit_note']='All direct child exit codes were observed as0 except adopted third feedback validation: completed result plus absent process, original exit handle unavailable. Original supervisor failed only during status publication; error and snapshots are preserved.'
report=(P/'report.md').read_text(encoding='utf-8')
report+=f"\nFinal closure: {closure['seconds_including_analysis']:.1f}s ({closure['seconds_including_analysis']/60:.1f}min) elapsed within the original14,400s allowance. All owned local workers exited, pinned model/trainer/stimulus hashes remain unchanged, and the allowance is closed. Selected checkpoint identities, the status-recovery history and final interpretation review are preserved with the completion receipt.\n"
for path in (P/'report.md',run/'report.md'):path.write_text(report,encoding='utf-8')
closure['artifacts']={p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='completion_receipt.json'}
for path in (P/'completion_receipt.json',run/'completion_receipt.json'):path.write_text(json.dumps(closure,indent=2),encoding='utf-8')
print(json.dumps({k:closure[k] for k in ('status','seconds_including_analysis','selected_checkpoints','allowance_closed')},indent=2))
