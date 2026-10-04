"""Report-only closeout under original deadline; never fits or selects a model."""
from adequacy import *
from run_upstream import change_from_scores,change_error
import shutil

def main():
    budget=json.loads((OUT/'budget.json').read_text());assert time.time()<budget['deadline']
    m=json.loads((OUT/'metrics.json').read_text());sel=json.loads((OUT/'selection.json').read_text());replay=json.loads((OUT/'independent_replay.json').read_text());pred=np.load(OUT/'test_predictions.npz');boot=np.load(OUT/'bootstrap_groups.npz')['indices'];ev=pred['truth_event'].astype(bool);side=pred['truth_side'];truth=pred['truth_delta'];idx=np.flatnonzero(ev);eb=[r[ev[r]] for r in boot]
    changegains={}
    for key,z in sel.items():
        if z['target']!='change':continue
        err=change_error(truth,change_from_scores(pred[key]));zero=abs(truth);vals=[(zero[r,side[r]]-err[r,side[r]]).mean() for r in eb];changegains[key]=dict(changed_mae_reduction_vs_zero=float((zero[idx,side[ev]]-err[idx,side[ev]]).mean()),ci95=np.quantile(vals,[.025,.975]).tolist())
    d.dump(OUT/'signed_change_gains.json',changegains)
    supplement=['# All fitted readouts and capacity','','Nominal quadratic dimensions differ; matched candidate df16/32/48 does not force validation to choose the same df. Intercept adds1.','|Fit|Input dim|Implicit features/output|Train groups|Selected df|Alpha|Threshold|Validation score|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,z in sel.items():supplement.append(f"|{k}|{z['dimensions']}|{z['implicit_features']}|{z['train_groups']}|{z['effective_df']:.3f}|{z['alpha']:.6g}|{z['threshold']:.2f}|{max(z['validation_scores']):.6f}|")
    supplement+=['','## Every classifier and shuffled baseline','Confusion order: true rows0/1, predicted columns0/1. Event0=catch; side0=left, side1=right, only actual events.','|Fit|BA|AUC [95%CI]|Confusion|','|---|---:|---:|---|']
    for k,q in m.items():
        if isinstance(q,dict) and 'auc' in q:supplement.append(f"|{k}|{q['balanced_accuracy']:.4f}|{q['auc']:.4f} [{q['auc_ci95'][0]:.4f},{q['auc_ci95'][1]:.4f}]|{q['confusion']}|")
    supplement+=['','## Every direct signed-change fit','|Fit|All-patch MAE|Changed-patch MAE [95%CI]|Catch MAE|Predicted SD L/R|Changed-MAE reduction vs zero [95%CI]|','|---|---:|---:|---:|---|---|']
    for k,q in m.items():
        if isinstance(q,dict) and 'changed_patch_mae' in q:
            g=changegains.get(k);gt='' if g is None else f"{g['changed_mae_reduction_vs_zero']:.3f} [{g['ci95'][0]:.3f},{g['ci95'][1]:.3f}]";supplement.append(f"|{k}|{q['mean_mae']:.3f}|{q['changed_patch_mae']:.3f} [{q['changed_patch_ci95'][0]:.3f},{q['changed_patch_ci95'][1]:.3f}]|{q['catch_mae']:.3f}|{q['predicted_change_std']}|{gt}|")
    supplement+=['','## Matched BA differences','|Comparison|BA gain [95%CI]|','|---|---:|']
    for k,q in m['paired_gains'].items():supplement.append(f"|{k}|{q['ba_gain']:+.4f} [{q['ci95'][0]:+.4f},{q['ci95'][1]:+.4f}]|")
    supplement+=['','## Failure and pre-fit correction','The first attempt stopped before fitting or test generation because raw-pixel dimensional bookkeeping omitted the second spatial patch (6348 actual coordinates;3174 per endpoint). `failed_dimensions.log` and original source/protocol copies preserve evidence. A failing regression test reproduced the mismatch, then passed after projection size was derived from saved training feature shape. Saved train/validation arrays were reused exactly. No trial rerender or scientific fit was duplicated.','Before any fit, the effective-df grid was corrected from16/64/128 to16/32/48: a64-dimensional projected linear design cannot support128 effective degrees of freedom. This was a feasibility correction, not outcome-based selection. The final protocol and all selected fits were hashed before the new test stream was generated. The original1800-second clock was never reset. All72 final fits solved successfully; max normal-equation relative residual3.65e-15.','Two contract tests passed. Separate no-fit replay reproduced72 predictions,79 metric records,64 paired differences, selected validation scores and effective df; regenerated random initialization from the seed without loading any checkpoint, matching its saved tensor hash. Nine actual raster examples across train/val/test reproduced all three representations bit-for-bit.']
    (OUT/'DETAILS.md').write_text('\n'.join(supplement)+'\n')
    conclusion='''## Decision: this probe family is not validated as an information-loss assay

**Measured probe limitation, not neural erasure.** On the same fresh movies, endpoint image-only optical flow localized the changed side at BA **0.740 [0.684,0.792]**, while matched ridge readouts on the **raw endpoint pixels themselves** achieved only **0.500–0.512**. Endpoint flow exceeded unprojected pixel ridge by **+0.240 [0.184,0.292]**. A negative result from this low-effective-capacity ridge family is therefore not diagnostic of absent motion-change information—even before a CNN is involved. This supports an adequacy concern about interpreting the earlier neural negatives, but does not prove that their exact older probe basis would fail: the new df-controlled polynomial kernel is not an exact replication.

**Removing projection did not rescue these fits.** Trained-CNN event-side BA was0.499/0.508 with projected linear/quadratic readouts and0.500/0.500 unprojected. Unprojected-minus-projected gains were **+0.001 [−0.053,+0.060]** linear and **−0.008 [−0.043,+0.021]** quadratic. Thus projection is not established as the sole problem; underfitting, insufficient sample support, unsuitable motion/comparison inductive bias, ROI sampling and threshold calibration remain unresolved. This finite matrix does not identify which one is responsible.

**Trained versus random did not separate reliably.** Projected trained-minus-random side-BA gains were+0.022 [−0.060,+0.103] linear and+0.036 [−0.031,+0.106] quadratic; both unprojected classifiers were constant decisions. This is failure to distinguish one trained encoder from one freshly initialized same-architecture control under these probes—not evidence of equivalent encoding or trained destruction. Trained unprojected linear side AUC was0.558 [0.494,0.633], a weak unresolved trend, not a rescue.

**Direct signed-change readouts largely shrank toward zero.** Across real-label ridge arms, changed-patch MAE was26.84–27.10°, versus27.02° for always-zero. Endpoint optical flow gave21.04° and full-history flow7.05°. Catch error and output variability are separately tabulated in [DETAILS.md](DETAILS.md); small all-patch error is not successful event recovery. Any-event BA remained0.476–0.526 for real-label ridge arms. Full/endpoint flow event BA was0.871/0.632, with210/258 and154/258 event hits, and3/42 and14/42 catch false positives respectively. Shuffled controls show no consistent real-label rescue; no post-test winner was selected.

**Temporal comparison remains unlocalized.** Both endpoints were explicitly supplied to every matched readout, so these failures cannot be attributed only to lack of endpoint access. But known signal in endpoint pixels plus failed raw-pixel and neural ridge fits prevents inferring an encoding lesion or a specific neural comparison failure. Full-history flow side BA0.946 [0.917,0.971] shows additional usable temporal evidence for that privileged observer, not a deployed-head remedy. No optical-flow deployment, architecture fix or change to teaching is recommended or authorized here.

Intervals are descriptive and unadjusted across the fixed exploratory matrix. BA intervals[0.500,0.500] and gain intervals[0,0] arise from constant decisions and empirical bootstrap degeneracy, **not certainty or proof of equivalence**; AUC intervals and raw confusion counts retain the uncertainty. The original native target-report task/stimuli were unchanged, but these analyses score physical event/side/change targets, not successful cue-conditioned reporting. No cueing, inhibition, microstimulation or biological circuit claim is made.

'''
    report=(OUT/'REPORT.md').read_text();report=report.replace('## Matched readouts',conclusion+'## Matched readouts',1)
    report+='\n## Reproduction and closeout\n\n`replay.py` performs no fitting and reproduces saved predictions/metrics under the same original deadline; it refuses execution after expiry. `test_contract.py` checks kernel algebra and actual feature/projection shape. Re-extraction/refitting is not automatically authorized. Exact dimensions, all classifier confusions/AUC intervals, all shuffled fits, signed-change errors/gains, and failure evidence: [DETAILS.md](DETAILS.md). `independent_replay.json` records the independent process audit. `final_receipt.json` records final elapsed time and artifact hashes after report/journal completion.\n'
    (OUT/'REPORT.md').write_text(report)
    journal='# Probe adequacy — selected2297 versus fresh random encoder\n\n'+conclusion+'Full report: `SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/ProbeAdequacyDiagnostic/REPORT.md`.\n\nFresh600train/150validation/300test groups;72 fits, matched linear/quadratic readouts and projected/unprojected inputs; effective ridge df candidates16/32/48. All predictions and metrics replayed exactly. At most2CPUthreads, one extractionworker, no cloud or model optimizer. One pre-fit dimension error was fixed with preserved evidence and no budget renewal.\n'
    (d.ROOT/'LabJournal/krauzlis-probe-adequacy-diagnostic.md').write_text(journal)
    sources=[Path(__file__),OUT/'adequacy.py',OUT/'run_adequacy.py',OUT/'replay.py',OUT/'test_contract.py',Path(d.__file__),OUT.parent/'UpstreamMotionDiagnostic/upstream.py',OUT.parent/'UpstreamMotionDiagnostic/run_upstream.py',Path(px.__file__)]+[d.ROOT/p for p in json.loads((OUT/'identity.json').read_text())['source_hashes']]
    manifest={}
    for p in sources:
        relative=p.relative_to(d.ROOT);dst=OUT/'source_archive'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);manifest[str(relative)]=d.sha(dst)
    d.dump(OUT/'source_manifest.json',manifest)
    print('REPORT CLOSED',time.time()-budget['started'],'seconds',json.dumps(changegains['trained/unprojected/degree1/change']))

if __name__=='__main__':main()
