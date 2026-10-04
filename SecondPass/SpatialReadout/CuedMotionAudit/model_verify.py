"""CPU-only verification/reduction of saved frozen diagnostic receipts."""
from SecondPass.SpatialReadout.CuedMotionAudit.model_diagnostic import *
def main():
    torch.set_num_threads(1)
    primary=[json.loads(a) for a in (OUT/'model_trials.jsonl').read_text().splitlines()]
    s=json.loads((OUT/'model_summary.json').read_text());re=summarize(primary)
    for k in re:assert re[k]==s[k],k
    assert sha(CKPT)==SHA
    assert all(sha(p)==h for p,h in s['source_hashes_unchanged'].items())
    g={}
    for r in primary:
        a=g.setdefault(r['condition'],dict(labels=[],logits=[],trajectory=[],metadata=[]))
        for k in a:a[k].extend(r[k])
    for c,n in [('native_D0',128),('native_D24',128),('cue_retarget_D0',128),('native_B20',200),('cue_swap_B20',200),('native_anchor',32)]:
        assert len(g[c]['labels'])==n
        assert len(set(m['trial_id'] for m in g[c]['metadata']))==n
    for c in ('native_D0','native_D24','cue_retarget_D0'):
        for label,m in zip(g[c]['labels'],g[c]['metadata']):assert np.argmax(m['duration_counts_by_patch'][m['target_location']])==label
    for c in ('native_B20','cue_swap_B20'):
        for label,m in zip(g[c]['labels'],g[c]['metadata']):assert int(m['changed_patch']==m['target_location'])==label
    extra=[json.loads(a) for a in (OUT/'model_evidence_trials.jsonl').read_text().splitlines()];h={}
    for r in extra:
        a=h.setdefault(r['condition'],dict(labels=[],logits=[],metadata=[]))
        for k in a:a[k].extend(r[k])
    expected=dict(OOD_time_reversal_D0=128,native_B20_replicate=200,native_law_event_removed_B20=200,native_B12=100,native_B28=100)
    assert {k:len(v['labels']) for k,v in h.items()}==expected
    assert h['native_B20_replicate']['logits']==g['native_B20']['logits']
    for c in ('native_B12','native_B28'):
        ev=[m['event_type'] for m in h[c]['metadata']];assert {e:ev.count(e) for e in set(ev)}==dict(target=57,foil=29,catch=14)
    rng=np.random.default_rng(935040);uncertainty={}
    for c in ('native_D0','native_D24','native_B20'):
        a=g[c];y=np.array(a['labels']);z=np.array(a['logits']);boots=[]
        for _ in range(1500):
            ids=rng.integers(0,len(y),len(y));boots.append(metrics(y[ids],z[ids])['auc_macro'])
        uncertainty[c+'_auc95']=np.quantile(boots,[.025,.975]).tolist()
    a=g['native_D0'];pred=np.array(a['logits']).argmax(1);y=np.array(a['labels']);last=np.array([m['directions_by_patch'][m['target_location']][-1] for m in a['metadata']]);uncued=np.array([[m['winner_by_patch'][j] for j in range(4) if j!=m['target_location']] for m in a['metadata']]);uncertainty['duration_majority_minus_last_agreement']=bootstrap((pred==y).astype(float)-(pred==last).astype(float));uncertainty['duration_cued_minus_mean_uncued_agreement']=bootstrap((pred==y).astype(float)-(pred[:,None]==uncued).mean(1))
    result=dict(status='passed',primary_condition_counts={k:len(v['labels']) for k,v in g.items()},supplement_condition_counts=expected,all_trial_ids_unique_within_condition=True,all_task_labels_recomputed=True,primary_summary_recomputed_exact=True,krauzlis_replication_logits_exact=True,checkpoint_and_source_hashes_verified=True,auc_bootstrap_replicates=1500,uncertainty=uncertainty,verified_unix=time.time(),elapsed_from_original_budget=time.time()-json.loads((OUT/'model_budget.json').read_text())['start_unix'])
    dump('model_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
