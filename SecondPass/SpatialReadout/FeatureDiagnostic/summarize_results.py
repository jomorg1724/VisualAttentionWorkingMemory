"""Describe frozen saved predictions; no fitting, selection or new inference."""
import csv,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    budget=json.loads((ROOT/'budget.json').read_text());assert time.time()<budget['deadline_unix']
    results=json.loads((ROOT/'results.json').read_text());rows=[];behavior=[];specific={}
    for taskrow in results:
        task=taskrow['task'];p=np.load(ROOT/'predictions'/f'{task}_test.npz');meta=json.loads((ROOT/'features'/f'{task}_test_metadata.json').read_text())['metadata']
        target=p['location'];labels=p['label'];delta=np.abs((p['probe_truth']-p['sample_truth']+np.pi/2)%np.pi-np.pi/2)*180/np.pi
        delta=np.round(delta).astype(int)
        taskstrata={}
        for model,m in taskrow['comparators'].items():
            pred=p['comparator_'+model]>=.5
            behavior.append(dict(task=task,model=model,n=m['n'],balanced_accuracy=m['balanced_accuracy'],auc=m['auc'],accuracy_wilson_low=m['accuracy_wilson95'][0],accuracy_wilson_high=m['accuracy_wilson95'][1],paired_gain=m['paired_accuracy_gain_vs_deployed'],paired_gain_low=m['paired_gain_bootstrap95'][0],paired_gain_high=m['paired_gain_bootstrap95'][1]))
            taskstrata[model]=[]
            selected_delta=delta[np.arange(len(labels)),target]
            for mag in np.unique(selected_delta):
                mask=selected_delta==mag;taskstrata[model].append(dict(target_absolute_change_degrees=int(mag),n=int(mask.sum()),accuracy=float((pred[mask]==labels[mask]).mean())))
        specific[task]=taskstrata
        for row in taskrow['probes']:
            if 'mean_degrees' not in row:continue
            truth=p[row['content']+'_truth']
            for loc in range(4):
                pred=p[f'{row["layer"]}_{row["stage"]}_{row["content"]}_loc{loc}'];err=np.abs((pred-truth[:,loc]+np.pi/2)%np.pi-np.pi/2)*180/np.pi
                for cue in ['all','cued','uncued']:
                    cm=np.ones(len(target),bool) if cue=='all' else (target==loc if cue=='cued' else target!=loc)
                    for magnitude in ['all']+np.unique(delta[:,loc]).tolist():
                        mask=cm if magnitude=='all' else cm&(delta[:,loc]==magnitude)
                        if not mask.any():continue
                        rows.append(dict(task=task,layer=row['layer'],stage=row['stage'],content=row['content'],location=loc,cue_status=cue,actual_absolute_change_degrees=magnitude,n=int(mask.sum()),mean_axial_error_degrees=float(err[mask].mean()),median_axial_error_degrees=float(np.median(err[mask])),fraction_below_7_5_degrees=float((err[mask]<7.5).mean()),fraction_error_below_half_actual_change=None if magnitude=='all' or magnitude==0 else float((err[mask]<magnitude/2).mean())))
    for name,data in [('angular_error_strata.csv',rows),('comparator_summary.csv',behavior)]:
        with (ROOT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (ROOT/'target_change_strata.json').write_text(json.dumps(specific,indent=2))
    overview='''## Measured conclusion

**The trained network contains the cue and usable local orientation information, but its deployed decision does not combine them successfully.** This is strongest for early activations: auxiliary-trained component decoders plus a feature-predicted relational readout solve91.6% of signed-orientation and100% of binding held-out trials. This is **not** a successful label-only rescue, nor proof that the final256-unit representation can solve either task.

- **Cue blindness is not supported.** At report, location decoding is100% across all four layers for both tasks. Signed-cue decoding is100% in early/final-visual/ConvGRU features and99.8% in the256-unit readout. Native label×location×sign balancing and label-stratified cue results prevent task-label association from explaining these cue scores.
- **Sensory precision changes along the processing path.** Mean early sample-angle error is1.33° for orientation and1.39° for binding. At report, early probe-angle errors are5.89° and2.66°; final-visual errors18.20° and11.46°; ConvGRU errors18.39° and13.81°; readout errors30.06° and22.14°. Orientation's15° minimum change is therefore small relative to late-stage probe errors. Sample-angle information remains more accessible than new probe-angle information. These are different finite-capacity linear decoders and ROI summaries, so this pattern does not prove irreversible information loss or a causal bottleneck.
- **Label-only learning remains unresolved.** The small relation-aware comparator scores49.2% BA on orientation and56.3% on binding versus deployed48.6%/50.8%. Both paired gain intervals include zero. Shuffled-label controls score53.1%/54.1%, reinforcing caution rather than establishing a reliable label-only improvement.
- **Parts can be combined with explicit auxiliary supervision and a supplied circular relation.** Using only predicted cue/sign/angles at evaluation, the structured diagnostic scores91.6%/100%, with paired gains42.97/49.22 percentage points (95% intervals37.89–47.85/44.73–53.52). Auxiliary decoders used only512 TRAIN episodes; the task-label calibrator used the other512. The orientation decoder's small errors still matter at15° and at unchanged targets. This comparator has geometry/timing, auxiliary-supervision and relational-inductive-bias privileges; it is neither the deployed head nor equivalent to the label-only arm.
- **No clean-retention conclusion.** All D0 report stacks still contain a sample image. The successful structured arm also accesses stored sample-time activations, an external diagnostic memory privilege. It demonstrates feature accessibility and composability across known times, not a successful single-final-state readout or recurrent memory mechanism.

**Next decision:** prioritize the acquisition/use of the cue-conditioned circular comparison, while preserving the evidence that late probe precision is weaker. These results do not support complete cue erasure or a retention-only explanation. They do not authorize more model training or establish an architectural remedy. A future comparison of equivalent-capacity readouts restricted to one final state versus time-separated activations would distinguish the diagnostic memory privilege from final-state accessibility; not run here.

'''
    report=ROOT/'REPORT.md';text=report.read_text();text=text.replace('## Feature accessibility',overview+'## Feature accessibility',1)
    text=text.replace('A stronger structured diagnostic than deployed performance would favor a failure of usable combination/readout over complete absence of component information, within these privileged fixed geometry/timing inputs.', 'The measured structured advantage supports usable early component information and successful explicitly supervised combination, not a final-state-only or label-only rescue.')
    v=json.loads((ROOT/'independent_verification.json').read_text());c=json.loads((ROOT/'completion.json').read_text())
    text+='\n## Independent readback and descriptive tables\n'
    text+=f'- Main worker completed extraction, fitting, frozen held-out scoring and initial report in{c["elapsed_seconds"]:.2f}s. Independent CPU readback finished at{v["elapsed_from_first_mps_seconds"]:.2f}s from first MPS work, below the1800s cap. All136 saved probe models and6 learned comparator models replay their saved test predictions; all feature/model hashes and metric calculations match. No new fits or test-driven selection occurred during readback.\n'
    text+='- `angular_error_strata.csv` adds all4 locations × cued/uncued/all × actual absolute-change strata from saved predictions. `target_change_strata.json` reports task-decision accuracy by actual selected-target change, including unchanged cases. `comparator_summary.csv` gives compact metrics and uncertainty. These are descriptive post-fit summaries of the same held-out predictions, not additional experiments.\n'
    text+='- Native/direct renderer equality and direct/extraction-wrapper equality were checked on each split\'s first8 episodes per task, plus the separate16-episode profile; native stream generation remains unmodified for every episode. Raster hashes/IDs are unique across all3584 production base episodes; profile episodes are separate. Exactly two completed tasks, four measured layers,136 selected angle/cue probes.\n'
    text+='- Reproduction: from the repository root use `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 /tmp/vawm-task-suite-venv/bin/python -B -m SecondPass.SpatialReadout.FeatureDiagnostic.run --run`. The original directory deliberately refuses a second run because its immutable budget exists. Reproduction requires separately authorized fresh output/budget, not deleting or renewing this receipt. Readback used `-m SecondPass.SpatialReadout.FeatureDiagnostic.verify`.\n'
    report.write_text(text)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name not in ['artifact_manifest.json','final_receipt.json'] and '__pycache__' not in p.parts:
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=files),indent=2))
    assert time.time()<budget['deadline_unix']
    (ROOT/'final_receipt.json').write_text(json.dumps(dict(status='complete_and_independently_verified',worker_elapsed_seconds=c['elapsed_seconds'],finalized_unix=time.time(),elapsed_including_readback_and_report_seconds=time.time()-budget['start_unix'],deadline_unix=budget['deadline_unix'],within_cap=True,n_base_episodes=v['n_base_episodes'],angular_summary_rows=len(rows),artifact_count=len(files),main_model_updates=0,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(angular_rows=len(rows),comparator_rows=len(behavior),final_elapsed=time.time()-budget['start_unix'])))
if __name__=='__main__':main()
