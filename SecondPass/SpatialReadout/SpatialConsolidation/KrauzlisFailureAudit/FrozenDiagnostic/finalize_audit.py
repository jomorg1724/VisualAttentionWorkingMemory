"""Audit persisted predictions and finish report within the ORIGINAL cap.
No model inference, probe refitting, test-based selection or cap renewal.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import argparse, json, hashlib, time, signal
from pathlib import Path
import numpy as np
from diagnostic import ba,auc,class_metrics,sha,dump,ROOT,CHECKPOINT
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).parent);a=p.parse_args();out=a.out
budget=json.loads((out/'budget.json').read_text());remaining=budget['deadline']-time.time()
assert remaining>0, 'Original nonrenewable cap expired; refusing analysis'
signal.setitimer(signal.ITIMER_REAL,remaining)
m=json.loads((out/'metrics.json').read_text());sel=json.loads((out/'selection.json').read_text());receipt=json.loads((out/'receipt.json').read_text());spec=json.loads((out/'allocation.json').read_text())
meta=json.loads((out/'test_metadata.json').read_text());pr=np.load(out/'test_predictions.npz');y=pr['labels'];groups=len(meta)//2
assert groups==spec['counts']['test']
for name,h in receipt['artifacts'].items():assert sha(out/name)==h
assert sha(CHECKPOINT)==json.loads((out/'identity.json').read_text())['checkpoint_sha256']
freeze=json.loads((out/'test_freeze.json').read_text());assert sha(out/'selection.json')==freeze['selection_sha256'];assert sha(out/'fitted_probes.npz')==freeze['fits_sha256']
allmeta=[json.loads((out/f'{s}_metadata.json').read_text()) for s in ('train','val','test')]
sets=[set(z['noncue_sha256'] for z in x) for x in allmeta]
assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
for rows in allmeta:
    for x,z in zip(rows[::2],rows[1::2]):
        assert x['group']==z['group'] and x['noncue_sha256']==z['noncue_sha256'] and x['movie_sha256']!=z['movie_sha256']
        assert x['target_location']==1-z['target_location']
        assert x['changed_patch']==z['changed_patch']
        assert x['label']==int(x['changed_patch']==x['target_location'])
        assert z['label']==int(z['changed_patch']==z['target_location'])
for key,v in sel.items():
    if v['target']=='angle':continue
    scores=pr[v['archive_prefix']];truth=pr['labels'] if v['target'] in ('label','shuffled_label') else pr[v['target']]
    actual=class_metrics(truth,scores)
    for metric in ('balanced_accuracy','accuracy'):assert actual[metric]==m[key][metric]
rng=np.random.default_rng(47831);boot=rng.integers(0,groups,(2000,groups));base=np.arange(0,len(meta),2)
logits=pr['native_logits'];margin=logits[:,1]-logits[:,0];delta=margin[1::2]-margin[::2];changed=y[::2]!=y[1::2];aligned=(2*y[1::2]-1)*delta
idx=np.flatnonzero(changed);bchange=rng.choice(idx,(2000,len(idx)),replace=True)
aucs=[auc(y[base][ix],margin[base][ix]) for ix in boot]
cuekey=sel['memory__final/cue']['archive_prefix'];cuep=pr[cuekey].argmax(1)
extra={'native_original_auc_ci95':np.quantile(aucs,[.025,.975]).tolist(),'native_original_counts':{e:sum(z['event_type']==e for z in meta[::2]) for e in ('target','foil','catch')},'native_paired':m['native_paired'],'paired_aligned_margin_mean':float(aligned[changed].mean()),'paired_aligned_margin_ci95':np.quantile(aligned[bchange].mean(1),[.025,.975]).tolist(),'paired_abs_margin_median':float(np.median(np.abs(delta))),'paired_abs_margin_max':float(np.max(np.abs(delta))),'paired_action_flip_wilson95_upper':float(1.959963984540054**2/(groups+1.959963984540054**2)),'memory_final_cue_by_baseline':{},'native_by_baseline':{},'noncue_raster_hashes_disjoint':True,'all_prediction_metrics_replayed':True}
for b in (12,20,28):
    ix=np.array([i for i,z in enumerate(meta) if z['baseline_transitions']==b]);ori=ix[ix%2==0]
    extra['memory_final_cue_by_baseline'][str(b)]={'n_groups':len(ix)//2,'ba':ba(pr['cue'][ix],cuep[ix])}
    extra['native_by_baseline'][str(b)]=class_metrics(y[ori],logits[ori])
dump(out/'verification.json',extra)
# Replace bulky raw-control JSON with compact, auditable result tables.
text=(out/'REPORT.md').read_text();start=text.index('## Temporal-access controls');end=text.index('## Protocol')
ctrl=['## Temporal-access controls','| Matched capacity | Final label BA | Phase label BA | Phase−final BA,95% grouped CI | Shuffled final / phase BA |','|---|---:|---:|---|---|']
for q in ('linear','quadratic'):
    g=m['phase_minus_final_'+q]
    ctrl.append(f"| {q} | {m[f'access_final_{q}/label']['balanced_accuracy']:.3f} | {m[f'access_phases_{q}/label']['balanced_accuracy']:.3f} | {g['ba_gain']:+.3f} [{g['ci95'][0]:+.3f},{g['ci95'][1]:+.3f}] | {m[f'access_final_{q}/shuffled_label']['balanced_accuracy']:.3f} / {m[f'access_phases_{q}/shuffled_label']['balanced_accuracy']:.3f} |")
text=text[:start]+'\n'.join(ctrl)+'\n\n'+text[end:]
summary='''## Decision and measured diagnosis
**No go for a claim that a simple replacement readout rescues this checkpoint. Inconclusive about the unique failing mechanism.** The early cue is not wholly erased: full ConvGRU memory decodes it perfectly after the five-frame gap and above chance at report. Coarse physical direction is also accessible, with mean angular errors larger than the26/28° changes; correlated errors could still cancel in a comparator, so this is not proof that event information is absent. Physical changed-patch decoding is weak, and neither final-only nor capacity-matched stored-phase label probes establish correct cue-conditioned report accessibility. This favors investigating fine-motion comparison/binding rather than asserting complete cue loss; it does not identify a causal lesion or justify altering task teaching.

'''
text=text.replace('## Behavioral result',summary+'## Behavioral result',1)
ci=extra['native_original_auc_ci95'];cue=m['memory__final/cue'];chg=m['access_phases_quadratic/changed'];lab=m['memory__final/label']
evidence=f"\nNative head still reports positive on every original target/foil/catch trial ({extra['native_original_counts']}); natural-frequency AUC95% group-bootstrap CI [{ci[0]:.3f},{ci[1]:.3f}]. The higher AUC on this fresh small sample than the historical selected test does not establish task acquisition: paired-cue AUC is {m['native_paired']['auc']:.3f}, no paired action flips, and task-aligned margin shift is {extra['paired_aligned_margin_mean']:.3g} [{extra['paired_aligned_margin_ci95'][0]:.3g},{extra['paired_aligned_margin_ci95'][1]:.3g}]. These tiny float32 differences do not establish meaningful cue use. Zero flips has a Wilson95% upper bound {extra['paired_action_flip_wilson95_upper']:.3f}, not proof of exactly zero future effect.\n"
text=text.replace('## Cue accessibility through time',evidence+'\n## Cue accessibility through time',1)
evidence=f"\nFull-memory final cue BA {cue['balanced_accuracy']:.3f},95% grouped CI [{cue['ba_ci95'][0]:.3f},{cue['ba_ci95'][1]:.3f}]. By baseline duration: "+'; '.join(f"B{b}={z['ba']:.3f} ({z['n_groups']}groups)" for b,z in extra['memory_final_cue_by_baseline'].items())+'. These are different independent trials; B manipulates motion duration, not a pure blank-delay experiment.\n'
text=text.replace('## Physical change and correct-label accessibility',evidence+'\n## Physical change and correct-label accessibility',1)
text=text.replace('## Actual per-dot circular direction',f"\nFinal-memory label BA95% CI [{lab['ba_ci95'][0]:.3f},{lab['ba_ci95'][1]:.3f}]. Phase-quadratic changed-patch BA95% CI [{chg['ba_ci95'][0]:.3f},{chg['ba_ci95'][1]:.3f}], three-class chance reference1/3; this modest exploratory result is not correct target binding.\n\n## Actual per-dot circular direction",1)
text=text.replace('Targets are circular means','The native baseline directions are90° apart and probe inputs include both patches: successful per-patch decoding does not establish independent local encoding of both patches. Early feature ROIs are deliberately fixed, not selected by the true cue.\n\nTargets are circular means',1)
text=text.replace('`diagnostic.py`, `test_diagnostic.py`;','`diagnostic.py`, `finalize_audit.py`, `test_diagnostic.py`; `verification.json` independently replays stored metrics and verifies disjoint noncue raster hashes;')
text += '\nFinal audit reads only saved artifacts, uses the same absolute budget deadline, and performs no refits or test-based selections. Native CPU inference process completed; a process-manager premature null exit notification was contradicted by OS PID/log inspection and was not used as evidence of completion.\n'
(out/'REPORT.md').write_text(text)
(ROOT/'LabJournal/krauzlis-frozen-diagnostic.md').write_text('# Native-angle Krauzlis frozen diagnostic — selected2297\n\n'+summary+f"Native fresh n175: BA0.500, AUC{m['native_original']['auc']:.3f}; all positive. Memory cue BA: gap1.000, baseline0.831, post0.709, final0.663. Memory direction error41–46°; native changes26/28°. Final-memory label BA0.513/AUC0.490; matched phase-linear BA0.488, phase-quadratic0.501; no reliable paired temporal-access gain.\n\n425/100/175 independent train/validation/test groups, two cue variants each,71 frozen ridge probes; grouped held-out uncertainty and shuffled-label controls. Native/wrapper logits and renderer/RNG parity passed; checkpoint/tensor hashes unchanged, zero deployed optimizer updates, no cloud access.\n\nFull artifacts: `SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/FrozenDiagnostic/REPORT.md`. One nonrenewable1800s cap covered profile, extraction, fitting and report; core run435.53s, final audit elapsed in `final_audit_receipt.json`.\n")
dump(out/'final_audit_receipt.json',dict(complete=True,elapsed_from_original_start_seconds=time.time()-budget['started'],within_original_cap=time.time()<budget['deadline'],original_deadline=budget['deadline'],report_sha256=sha(out/'REPORT.md'),verification_sha256=sha(out/'verification.json'),new_model_inferences=0,new_probe_fits=0,checkpoint_sha256=sha(CHECKPOINT)))
print(json.dumps(extra,indent=2));print((out/'final_audit_receipt.json').read_text())
signal.setitimer(signal.ITIMER_REAL,0)
