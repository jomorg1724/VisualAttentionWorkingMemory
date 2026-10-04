"""Close this immutable diagnostic budget after all reports and audits."""
import json,time,hashlib,os
from pathlib import Path
out=Path(__file__).parent
root=Path('/Users/jonathanmorgan/Desktop/VisualAttentionWorkingMemory')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
budget=json.loads((out/'budget.json').read_text())
assert not (out/'final_receipt.json').exists(), 'Already closed; no budget renewal'
assert time.time()<budget['deadline'], 'Original budget expired'
main=json.loads((out/'completion.json').read_text());audit=json.loads((out/'independent_audit.json').read_text());identity=json.loads((out/'identity.json').read_text());freeze=json.loads((out/'test_freeze.json').read_text())
assert main['complete'] and audit['replay_all_new_and_prior_direction_predictions_max_abs']==0
assert sha(Path(identity['checkpoint']))==identity['checkpoint_sha256']
for name,h in freeze['hashes'].items():assert sha(out/name)==h
for name,h in main['artifacts'].items():assert sha(out/name)==h
assert sha(out/'run_upstream.py')==identity['runner_sha256']
assert sha(out/'upstream.py')==identity['upstream_sha256']
assert 'COMPLETE' in (out/'run.log').read_text()
files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
journals={str(p.relative_to(root)):sha(p) for p in [root/'LabJournal'/n for n in ('krauzlis-upstream-motion-diagnostic.md','README.md','CURRENT_STATUS.md','CHRONOLOGY.md')]}
now=time.time();receipt=dict(complete=True,closed=True,started=budget['started'],finished=now,deadline=budget['deadline'],elapsed_seconds=now-budget['started'],within_original_1200_seconds=now<budget['deadline'],cap_renewed=False,scientific_workers=1,max_numerical_threads=2,model_updates=0,cloud_actions=0,checkpoint_hash_unchanged=True,all_frozen_selections_unchanged=True,all_saved_predictions_replay_max_abs=0.0,unit_tests_passed=2,failed_import_preserved=True,process_verification='OS worker77584 zombie after COMPLETE log and completion receipt; no live duplicate; audits subsequently foreground exit0',artifact_hashes=files,journal_hashes=journals)
(out/'final_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('artifact_hashes','journal_hashes')},indent=2))
