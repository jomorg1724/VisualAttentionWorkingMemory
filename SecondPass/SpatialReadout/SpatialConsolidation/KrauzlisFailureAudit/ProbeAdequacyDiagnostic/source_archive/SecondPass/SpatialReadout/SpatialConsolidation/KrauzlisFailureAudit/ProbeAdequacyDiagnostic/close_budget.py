"""One-time closure; no extraction/fitting/reselection and no cap renewal."""
from adequacy import *
import shutil

budget=json.loads((OUT/'budget.json').read_text())
assert not (OUT/'final_receipt.json').exists() and time.time()<budget['deadline']
manifest=json.loads((OUT/'source_manifest.json').read_text())
for p in (OUT/'test_contract.py',Path(__file__)):
    rel=p.relative_to(d.ROOT);dest=OUT/'source_archive'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest[str(rel)]=d.sha(dest)
d.dump(OUT/'source_manifest.json',manifest)
for p,h in manifest.items():assert d.sha(d.ROOT/p)==h and d.sha(OUT/'source_archive'/p)==h
freeze=json.loads((OUT/'test_freeze.json').read_text())
for p,h in freeze['hashes'].items():assert d.sha(OUT/p)==h
ident=json.loads((OUT/'identity.json').read_text());assert d.sha(d.CHECKPOINT)==ident['checkpoint_sha256']
replay=json.loads((OUT/'independent_replay.json').read_text());assert replay['fitted_predictors_replayed']==72 and replay['metric_records_exact']==79 and replay['paired_gains_exact']==64
elapsed=time.time()-budget['started']
text=f'\n**Final closeout:** report, journal, source archive, all fits and independent no-fit replay completed within the original1800-second cap; elapsed at closeout{elapsed:.1f}s. No automatic extension or further fitting. Exact final elapsed and hashes are in `final_receipt.json`.\n'
for p in (OUT/'REPORT.md',d.ROOT/'LabJournal/krauzlis-probe-adequacy-diagnostic.md'):
    with p.open('a') as f:f.write(text)
files={str(p.relative_to(OUT)):dict(sha256=d.sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
journals={str(p.relative_to(d.ROOT)):d.sha(p) for p in (d.ROOT/'LabJournal/krauzlis-probe-adequacy-diagnostic.md',d.ROOT/'LabJournal/README.md',d.ROOT/'LabJournal/CURRENT_STATUS.md')}
result=dict(complete=True,elapsed_seconds=time.time()-budget['started'],deadline=budget['deadline'],within_cap=time.time()<budget['deadline'],budget_seconds=1800,renewed=False,closed=True,no_further_fits=True,artifact_count=len(files),artifact_bytes=sum(v['bytes'] for v in files.values()),artifacts=files,journal_hashes=journals,checkpoint_sha256=ident['checkpoint_sha256'],random_state_sha256=ident['random_state_sha256'],trained_state_sha256=ident['trained_state_sha256'],verified_fitted_models=72,metric_records=79,paired_gains=64,threads=2,workers=1,issues='one pre-fit raw dimension mismatch and pre-fit df feasibility correction; preserved logs/protocol/source; no fitted or test results existed then')
assert result['within_cap'];d.dump(OUT/'final_receipt.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','journal_hashes')}))
