"""Report-only summaries from frozen saved predictions, no new fits/selections."""
from upstream import *

def main():
    budget=json.loads((OUT/'budget.json').read_text());assert time.time()<budget['deadline']
    m=json.loads((OUT/'metrics.json').read_text());p=np.load(OUT/'test_predictions.npz');pix=np.load(OUT/'pixel_test.npz');boot=np.load(OUT/'bootstrap_groups.npz')['indices'];ev=p['truth_event'][::2].astype(bool);side=p['truth_side'][::2];meta=json.loads((OUT/'test_metadata.json').read_text())[::2]
    full=abs(p['pixel/full/delta']);end=abs(p['pixel/endpoint/delta']);sf=full[:,1]-full[:,0];se=end[:,1]-end[:,0]
    bb=[r[ev[r]] for r in boot];gain=[d.ba(side[r],(sf[r]>0).astype(int))-d.ba(side[r],(se[r]>0).astype(int)) for r in bb]
    coverage={}
    for name in ('full','endpoint'):
        missing=[];accepted=[]
        for i,b in enumerate(pix['baseline']):
            windows=(slice(0,b),slice(b,b+8)) if name=='full' else (slice(b-2,b),slice(b+6,b+8))
            cc=np.stack([pix['counts'][i,s].sum(0) for s in windows]);missing.append(cc==0);accepted.append(cc)
        coverage[name]=dict(zero_flow_patch_phases=int(np.sum(missing)),total_patch_phases=int(np.size(missing)),mean_accepted_flow_vectors_by_phase_patch=np.mean(accepted,axis=0).tolist())
    supplemental=dict(full_minus_endpoint_side_ba=dict(gain=d.ba(side[ev],(sf[ev]>0).astype(int))-d.ba(side[ev],(se[ev]>0).astype(int)),ci95=np.quantile(gain,[.025,.975]).tolist()),coverage=coverage)
    thresholds=json.loads((OUT/'pixel_thresholds.json').read_text())
    label_subgroups={}
    for name,delta in [('full',full),('endpoint',end)]:
        scores=delta[np.arange(len(meta)),pix['cue']];pred=scores>thresholds[name+'/label']['threshold'];label_subgroups[name]={}
        for event in ('target','foil','catch'):
            ix=np.array([r['event_type']==event for r in meta]);label_subgroups[name][event]=dict(n=int(ix.sum()),positive_reports=int(pred[ix].sum()))
    supplemental['target_report_subgroups']=label_subgroups
    # Wilson accuracy bounds, not BA bounds; protect against bootstrap ceiling claims.
    n=len(meta);z=1.959963984540054
    supplemental['cue_accuracy_wilson95']=[float(1/(1+z*z/n)),1.0]
    d.dump(OUT/'supplemental_metrics.json',supplemental)
    lines=['# Supplemental scoring and execution audit','', '## Event hits and catch false alarms', '| Observer | Hits /258 | Catch false positives /42 |','|---|---:|---:|']
    for key in ['pixel/full','pixel/endpoint']+[s+'/'+a for s in ('cnn25','kda25','kda7','memory') for a in ('phases','final')]:
        q=m[key+'/event']['confusion'];lines.append(f'| {key} | {q[1][1]} | {q[0][1]} |')
    lines+=['','The pixel any-event thresholds are16° full and18° endpoint. Target-report thresholds are13° full and20° endpoint; the target is decoded from the cue pixels, never supplied from truth. All phase neural event probes and both memory access probes declare event on all258 events AND all42 catches. Their degenerate BA bootstrap [.5,.5] is not proof of no information or exact population performance; inspect AUC uncertainty.','', '## Actual target reports', '| Pixel access | Target hits /171 | Foil false reports /87 | Catch false reports /42 |','|---|---:|---:|---:|']
    for name in ('full','endpoint'):
        a=label_subgroups[name];lines.append(f"| {name} | {a['target']['positive_reports']} | {a['foil']['positive_reports']} | {a['catch']['positive_reports']} |")
    lines+=['','## Group-shuffled controls', 'One deterministic shuffled-label fit per task/access, not a permutation-test null distribution. Validation selection uses the same candidate grid and untouched validation truth.','| Access | True /shuffled side BA | True /shuffled event BA | True /shuffled changed-patch error |','|---|---:|---:|---:|']
    for site in ('cnn25','kda25','kda7','memory'):
        for accessname in ('phases','final'):
            key=site+'/'+accessname;vals=[]
            for task in ('side','event'):
                vals.append(f"{m[key+'/'+task]['balanced_accuracy']:.3f} /{m[key+'/'+task+'/shuffled']['balanced_accuracy']:.3f}")
            vals.append(f"{m[key+'/change']['changed_patch_mae']:.2f}° /{m[key+'/change/shuffled']['changed_patch_mae']:.2f}°");lines.append('| '+key+' | '+' | '.join(vals)+' |')
    lines+=['','## Matched access uncertainty', '| Phase minus final | Side BA gain [95% CI] |','|---|---:|']
    for site in ('cnn25','kda25','kda7','memory'):
        q=m['paired_gains'][f'{site}/phases/side MINUS {site}/final/side'];lines.append(f"| {site} | {q['ba_gain']:+.3f} [{q['ci95'][0]:+.3f},{q['ci95'][1]:+.3f}] |")
    q=supplemental['full_minus_endpoint_side_ba'];lines+=['',f"Pixel full minus endpoint side BA: {q['gain']:+.3f} [{q['ci95'][0]:+.3f},{q['ci95'][1]:+.3f}]. Unlike the neural access pairs, full pixels explicitly contain more raw samples; this is a temporal-evidence benchmark, not a capacity-matched architecture test.", '', '## Pixel support and timing',json.dumps(coverage,indent=2), 'All episodes retained. A phase with no accepted flow receives angle0 by the unchanged method; no ground-truth-dependent filtering. Endpoint vectors sample only the last two transitions while targets summarize all movement angles in the phase. This target/window mismatch and dot resets contribute to endpoint error. No learned neural pooling-loss measurement is made.',f"Image cue decoding300/300 has a Wilson95% accuracy bound [{supplemental['cue_accuracy_wilson95'][0]:.4f},1], not population certainty.", '', '## Scope of temporal-access comparison', 'Each neural assay reads TWO fixed32-coordinate random projections (with the same slot-specific coordinate rolls), then all quadratic products. It does not read all raw pre/post coordinates or an explicit uncompressed feature difference. True labels/directions only supervise fits and score predictions. Prior direction decoders use full1152/576/3136 inputs and are not capacity-equivalent to the2144 polynomial-feature direct comparator. These restrictions can hide information; a failure is not neural erasure. The physical-side probe is conditional on a real event only for fitting/scoring; truth is not input to its predictor and it emits scores on catches too.', '', '## Failure, recovery and verification', 'The initial process failed at import with ModuleNotFoundError: threadpoolctl, before main(), fitting, selection or test generation. failed_import.log and failure_recovery.json preserve it. The optional dependency was removed; all numerical environment caps were set before imports and PyTorch used2 intra-op/1 inter-op threads. No install, budget reset, duplicate scientific worker or renewed allowance. The recovered run used the SAME persisted deadline.', 'The process tool subsequently returned premature exited/exit_code:null notices. OS ps verified the sole worker77584 still running while run.log advanced; no duplicate was launched. Completion was accepted only after completion.json, final log and OS zombie/no-running-worker state. Saved-array audit ran afterward, serially.', 'Independent audit:48 scalers exactly reconstructed from their true fit groups; maximum ridge normal-equation relative residual3.05e-11. All new and reused direction predictions replay with maximum absolute error0.0. Legacy full-history pixel method agrees within3.55e-15 degrees on the three saved B12/B20/B28 examples. Frozen fit/selection hashes unchanged; renderer/hook/source/checkpoint checks pass. Unit tests cover input-only final access/NaN removal and circular wrap/pixel window contract.', '', 'All uncertainty is episode-sampling uncertainty conditional on this one checkpoint and fixed probe fits; validation was reused and no multiplicity correction applied. No fitted target-label neural rescue or training remedy was chosen.']
    (OUT/'DETAILS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(supplemental,indent=2))

if __name__=='__main__':main()
