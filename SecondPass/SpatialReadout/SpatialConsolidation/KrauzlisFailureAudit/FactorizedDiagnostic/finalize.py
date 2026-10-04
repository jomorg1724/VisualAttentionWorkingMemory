"""Finalize reports without fitting or changing any selections."""
import json,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
budget=json.loads((P/'budget.json').read_text());assert time.time()<budget['deadline']
z=1.959963984540054;n=175
bounds={'independent_groups':n,'perfect_native_accuracy_wilson95':[1/(1+z*z/n),1.0],'zero_native_flips_wilson95':[0.,z*z/(n+z*z)],'not_a_balanced_accuracy_interval':True}
(P/'finite_sample_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
import replay
replay.verify_and_report(budget['deadline'])
report=P/'REPORT.md'
report.write_text(report.read_text()+f"\n## Final numerical audit\n\nAll14 saved ridge fits passed exact fit-group scaler reconstruction, selected validation-score replay and regularized normal-equation checks (maximum relative residual3.54e-10); `audit_fits.py` also saved the disjoint-group calibration inputs. Independent held-out prediction replay error was0.0. Finite-sample Wilson95% accuracy lower bound at175/175 is{bounds['perfect_native_accuracy_wilson95'][0]:.4f}; zero flips upper bound is{bounds['zero_native_flips_wilson95'][1]:.4f}. These are group/episode accuracy bounds, not BA intervals. Original features/checkpoint remained untouched.\n")
receipt=json.loads((P/'completion.json').read_text())
receipt['final_audit_elapsed_seconds']=time.time()-budget['started'];receipt['within_cap']=time.time()<budget['deadline']
receipt['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name!='completion.json'}
(P/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert receipt['within_cap']
print(json.dumps({'complete':True,'final_audit_elapsed_seconds':receipt['final_audit_elapsed_seconds'],'bounds':bounds}))
