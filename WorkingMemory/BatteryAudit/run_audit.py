"""Ideal-observer audit of the five-task spatial battery. CPU only, no training.

Usage: python -m WorkingMemory.BatteryAudit.run_audit [--seed 63973001] [--n-scale 1.0]
Writes results.json and report.md next to this file.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from WorkingMemory.SpatialTaskBattery import stimuli as S
from WorkingMemory.BatteryAudit.observers import (GaborBank,decode_ring,decode_sign_glyph,wrap_half_pi,motion_observer,
    krauzlis_observer,balanced_accuracy,auc_binary,best_threshold_ba)

HERE=Path(__file__).resolve().parent

def trials(stream,n,task,cond,chunk=32):
    """Yield (frames ndarray TCHW float32, label, meta) one trial at a time."""
    done=0
    while done<n:
        k=min(chunk,n-done);x,y,m=stream.batch(k,task,cond)
        for i in range(k):yield x[i].numpy(),int(y[i]),m[i]
        done+=k

# ---------------------------------------------------------------------------
def audit_orientation(stream,n):
    bank=GaborBank();rows=[];multiset=defaultdict(Counter);normalised=defaultdict(Counter)
    for frames,label,meta in trials(stream,n,'orientation_cued',dict(delay=0)):
        tgt,sign,dark=decode_sign_glyph(frames[0]);probe=meta['probe_frame']
        est=[];truth=[]
        for k,(x,y) in enumerate(S.CENTERS):
            a0,_=bank.angle(frames[1][0],x,y);a1,_=bank.angle(frames[probe][0],x,y)
            est.append(np.rad2deg(wrap_half_pi(a1-a0)));truth.append(meta['rotations_degrees'][k])
            # Also check absolute angle recovery on the sample frame.
            rows.append(dict(kind='angle',err=float(np.rad2deg(np.abs(wrap_half_pi(a0-meta['sample_angles_radians'][k]))))))
        est=np.array(est);truth=np.array(truth)
        pred=int(sign*est[tgt]>7.5)
        rows.append(dict(kind='trial',label=label,pred=pred,score=float(meta['cue_sign']*est[meta['target_location']]),cue_ok=int(tgt==meta['target_location'] and sign==meta['cue_sign']),
            magnitude=meta['rotation_magnitude'],delta_err=float(np.abs(est-truth).max()),target_row=int(meta['target_location']<2)))
        multiset[label][tuple(sorted(meta['rotations_degrees']))]+=1
        normalised[label][tuple(sorted(int(round(v/(meta['cue_sign']*meta['rotation_magnitude']))) for v in meta['rotations_degrees']))]+=1
    t=[r for r in rows if r['kind']=='trial'];a=[r['err'] for r in rows if r['kind']=='angle']
    y=[r['label'] for r in t];p=[r['pred'] for r in t]
    per_mag={m:balanced_accuracy([r['label'] for r in t if r['magnitude']==m],[r['pred'] for r in t if r['magnitude']==m],2) for m in (15,30,45)}
    per_row={('top' if rr else 'bottom'):balanced_accuracy([r['label'] for r in t if r['target_row']==rr],[r['pred'] for r in t if r['target_row']==rr],2) for rr in (1,0)}
    # Leak test without the cue: total-variation distance between label-conditional multiset distributions.
    keys=set(multiset[0])|set(multiset[1]);n0=sum(multiset[0].values());n1=sum(multiset[1].values())
    tv=.5*sum(abs(multiset[0][k]/n0-multiset[1][k]/n1) for k in keys)
    keys=set(normalised[0])|set(normalised[1]);tvn=.5*sum(abs(normalised[0][k]/n0-normalised[1][k]/n1) for k in keys)
    return dict(n=len(t),balanced_accuracy=balanced_accuracy(y,p,2),auc=auc_binary(y,[r['score'] for r in t]),cue_decode_accuracy=float(np.mean([r['cue_ok'] for r in t])),
        ba_by_magnitude=per_mag,ba_by_target_row=per_row,angle_error_deg=dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a))),
        rotation_error_deg=dict(median=float(np.median([r['delta_err'] for r in t])),p95=float(np.percentile([r['delta_err'] for r in t],95))),
        no_cue_multiset_tv_distance=float(tv),no_cue_normalised_multiset_tv_distance=float(tvn),normalised_multiset_counts={str(l):{str(k):v for k,v in c.items()} for l,c in normalised.items()},multiset_counts={str(l):{str(k):v for k,v in c.items()} for l,c in multiset.items()})

# ---------------------------------------------------------------------------
def audit_motion(stream,n):
    rows=[]
    for frames,label,meta in trials(stream,n,'motion_duration_cued',dict(delay=0)):
        tgt,margin=decode_ring(frames[0]);mv=meta['moving_frames'];ref=meta['reference_frame']
        per_patch=[]
        for k,(x,y) in enumerate(S.CENTERS):
            hard,soft,pf,sc=motion_observer(frames,x,y,mv,ref)
            truth_seq=meta['directions_by_patch'][k]
            per_patch.append(dict(hard=hard,soft=soft,frame_agree=float(np.mean(np.array(pf)==np.array(truth_seq))),winner=meta['winner_by_patch'][k],score=sc.sum(axis=0).tolist()))
        T=meta['target_location'];uncued=[k for k in range(4) if k!=T]
        counts=sorted(meta['duration_counts_by_patch'][T])
        rows.append(dict(label=label,step=meta['step_pixels'],cue_ok=int(tgt==T),cue_margin=margin,hard=per_patch[T]['hard'],soft=per_patch[T]['soft'],frame_agree=per_patch[T]['frame_agree'],
            uncued_self=[int(per_patch[k]['soft']==per_patch[k]['winner']) for k in uncued],uncued_vs_label=[int(per_patch[k]['soft']==label) for k in uncued],
            margin_count=int(counts[-1]-counts[-2])))
    y=[r['label'] for r in rows]
    def by(key,vals,field):return {str(v):balanced_accuracy([r['label'] for r in rows if r[key]==v],[r[field] for r in rows if r[key]==v],4) for v in vals}
    return dict(n=len(rows),cue_decode_accuracy=float(np.mean([r['cue_ok'] for r in rows])),cue_margin_min=float(min(r['cue_margin'] for r in rows)),
        ba_hard_vote=balanced_accuracy(y,[r['hard'] for r in rows],4),ba_soft_sum=balanced_accuracy(y,[r['soft'] for r in rows],4),
        per_frame_direction_agreement=float(np.mean([r['frame_agree'] for r in rows])),
        ba_soft_by_step=by('step',(.8,1.2,1.6),'soft'),ba_hard_by_step=by('step',(.8,1.2,1.6),'hard'),
        ba_soft_by_count_margin=by('margin_count',sorted({r['margin_count'] for r in rows}),'soft'),
        count_margin_histogram={str(k):v for k,v in sorted(Counter(r['margin_count'] for r in rows).items())},
        uncued_patch_self_accuracy=float(np.mean([v for r in rows for v in r['uncued_self']])),
        uncued_patch_predicts_cued_label=float(np.mean([v for r in rows for v in r['uncued_vs_label']])))

# ---------------------------------------------------------------------------
def audit_krauzlis(stream,n_per):
    out={};centers=np.array([[20.,50.],[80.,50.]]);yy,xx=np.mgrid[:100,:100]
    rings=[np.abs(np.sqrt((xx-cx)**2+(yy-cy)**2)-(8.125+1.8))<.7 for cx,cy in centers]
    for base in (12,20,28):
        rows=[]
        for frames,label,meta in trials(stream,n_per,'krauzlis_cued_motion',dict(baseline_transitions=base)):
            sc=[frames[0][0][m].mean() for m in rings];tgt=int(np.argmax(sc))
            obs=krauzlis_observer(frames,centers,base,meta['reference_frame'])
            T=meta['target_location'];F=1-T
            rows.append(dict(label=label,event=meta['event_type'],cue_ok=int(tgt==T),cued_delta=obs[T]['delta_deg'],foil_delta=obs[F]['delta_deg'],
                true_signed=meta['signed_change_degrees'],changed=meta['changed_patch'],pre_speed=obs[T]['pre_speed'],
                pre_angle_err=float(((obs[T]['pre_angle']-meta['baseline_means_degrees'][T]+180)%360)-180)))
        y=[r['label'] for r in rows];score=[abs(r['cued_delta']) for r in rows]
        tgt_rows=[r for r in rows if r['event']=='target'];catch=[r for r in rows if r['event']=='catch'];foil=[r for r in rows if r['event']=='foil']
        signed_err=[r['cued_delta']-r['true_signed'] for r in tgt_rows]
        noise=[r['cued_delta'] for r in catch]+[r['cued_delta'] for r in foil]
        ba_best,th=best_threshold_ba(y,score)
        ta=[abs(r['cued_delta']) for r in tgt_rows];na=np.abs(noise)
        out[f'B{base}']=dict(n=len(rows),event_counts=dict(Counter(r['event'] for r in rows)),cue_decode_accuracy=float(np.mean([r['cue_ok'] for r in rows])),
            auc_abs_delta=auc_binary(y,score),ba_at_13deg=balanced_accuracy(y,[int(s>=13) for s in score],2),ba_best_threshold=ba_best,best_threshold_deg=th,
            target_trials=dict(mean_abs_est=float(np.mean(ta)),signed_error_mean=float(np.mean(signed_err)),signed_error_sd=float(np.std(signed_err))),
            unchanged_cued_patch=dict(mean_abs_est=float(np.mean(na)),sd=float(np.std(na))),
            foil_patch_when_changed=dict(mean_abs_est=float(np.mean([abs(r['foil_delta']) for r in foil]))),
            dprime=float((np.mean(ta)-np.mean(na))/np.sqrt(.5*(np.var(ta)+np.var(na)))),
            pre_angle_error_deg=dict(median=float(np.median(np.abs([r['pre_angle_err'] for r in rows]))),p95=float(np.percentile(np.abs([r['pre_angle_err'] for r in rows]),95))),
            mean_pre_speed_px_per_frame=float(np.mean([r['pre_speed'] for r in rows])))
    return out

# ---------------------------------------------------------------------------
def audit_binding(stream,n0,n24):
    bank=GaborBank();out={};pair_label=defaultdict(Counter)
    for delay,n in ((0,n0),(24,n24)):
        rows=[]
        for frames,label,meta in trials(stream,n,'spatial_binding',dict(delay=delay)):
            cue_f=meta['cue_frames'][0];probe=meta['probe_frame'];tgt,margin=decode_ring(frames[cue_f])
            deltas=[]
            for k,(x,y) in enumerate(S.CENTERS):
                a0,_=bank.angle(frames[1][0],x,y);a1,_=bank.angle(frames[probe][0],x,y);deltas.append(abs(np.rad2deg(wrap_half_pi(a1-a0))))
            changed=sorted(np.argsort(deltas)[-2:].tolist());pred=int(tgt in changed)
            rows.append(dict(label=label,pred=pred,cue_ok=int(tgt==meta['target_location']),pair_ok=int(changed==sorted(meta['swapped_locations']))))
            pair_label[tuple(sorted(meta['swapped_locations']))][label]+=1
        y=[r['label'] for r in rows]
        out[f'D{delay}']=dict(n=len(rows),balanced_accuracy=balanced_accuracy(y,[r['pred'] for r in rows],2),cue_decode_accuracy=float(np.mean([r['cue_ok'] for r in rows])),
            changed_pair_recovered=float(np.mean([r['pair_ok'] for r in rows])))
    # Without the cue the only pixel evidence is which pair swapped. P(label=1 | pair) should be 0.5 for every pair.
    out['no_cue_pair_leak']={str(k):dict(n=sum(c.values()),p_label1=c[1]/sum(c.values())) for k,c in sorted(pair_label.items())}
    return out

# ---------------------------------------------------------------------------
def audit_recognition(stream,n):
    out={}
    for load,hold in ((0,3),(4,3),(24,5)):
        rows=[]
        for frames,label,meta in trials(stream,n,'image_recognition',dict(load=load,probe_hold=hold)):
            study=[hashlib.sha256(frames[i].tobytes()).hexdigest() for i in meta['study_frames']]
            probes=[hashlib.sha256(frames[i].tobytes()).hexdigest() for i in meta['probe_frames']]
            pred=int(probes[0] in study)
            rows.append(dict(label=label,pred=pred,hold_identical=int(len(set(probes))==1),study_unique=int(len(set(study))==len(study)),
                blanks_ok=int(all(np.all(frames[i]==.5) for i in meta['blank_frames'])),meta_match=int(study==meta['study_raster_sha256'] and probes[0]==meta['probe_raster_sha256'])))
        y=[r['label'] for r in rows]
        out[f'N{load}_H{hold}']=dict(n=len(rows),accuracy=float(np.mean([r['pred']==r['label'] for r in rows])),label_counts={str(k):v for k,v in Counter(y).items()},
            balanced_accuracy=(balanced_accuracy(y,[r['pred'] for r in rows],2) if len(set(y))==2 else None),
            probe_hold_identical=float(np.mean([r['hold_identical'] for r in rows])),study_unique=float(np.mean([r['study_unique'] for r in rows])),
            blanks_are_gray=float(np.mean([r['blanks_ok'] for r in rows])),metadata_hashes_match=float(np.mean([r['meta_match'] for r in rows])))
    manifest=json.loads((Path(S.DATA_ROOT)/'manifest.json').read_text())
    splits=defaultdict(set);bases=defaultdict(set)
    for r in manifest['images']:splits[r['split']].add(r['sha256']);bases[r['split']].add(r['base_id'])
    names=sorted(splits)
    out['manifest']=dict(counts={k:len(v) for k,v in splits.items()},sha_disjoint=all(not(splits[a]&splits[b]) for a in names for b in names if a<b),
        base_id_disjoint=all(not(bases[a]&bases[b]) for a in names for b in names if a<b))
    return out

# ---------------------------------------------------------------------------
def frame_roles(stream):
    """Collect every frame-index field from the metadata for the extreme conditions of each task."""
    out={}
    probes={'orientation_cued':[dict(delay=0),dict(delay=24)],'spatial_binding':[dict(delay=0),dict(delay=24)],'motion_duration_cued':[dict(delay=0),dict(delay=24)],
        'image_recognition':[dict(load=0,probe_hold=3),dict(load=24,probe_hold=5)],'krauzlis_cued_motion':[dict(baseline_transitions=12),dict(baseline_transitions=28)]}
    for task,conds in probes.items():
        for c in conds:
            x,y,m=stream.batch(1,task,c);meta=m[0];T=x.shape[1]
            fields={k:v for k,v in meta.items() if (k.endswith('_frame') or k.endswith('_frames')) and 'lifetime' not in k}
            idx=[i for v in fields.values() for i in (v if isinstance(v,list) else [v])]
            # Non-gray pixel count per frame: cheap view of which frames carry stimulus or cue.
            nongray=[int((np.abs(x[0,t].numpy()-.5)>1e-6).sum()) for t in range(T)]
            out[f'{task} {c}']=dict(frames=T,frame_count=S.frame_count(task,c),fields=fields,all_indices_in_range=bool(idx and max(idx)<T and min(idx)>=0),last_referenced=max(idx) if idx else None,nongray_pixels_per_frame=nongray)
    return out

# ---------------------------------------------------------------------------
def fmt(x):
    return f'{x:.3f}' if isinstance(x,float) else str(x)

def write_report(res,path):
    L=[]
    L.append('# Ideal-observer audit of the five-task battery\n')
    L.append(f"Generated {res['generated']} from `WorkingMemory/SpatialTaskBattery/stimuli.py` (SHA256 `{res['stimuli_sha256'][:16]}...`), validation stream seed {res['seed']}. CPU only, no training. Full numbers in `results.json`.\n")
    o=res['orientation'];L.append('## Orientation (cued sign, D0)\n')
    L.append('| n | BA | AUC | BA 15deg | BA 30deg | BA 45deg | BA top row | BA bottom row | glyph decode | angle err median / p95 (deg) | no-cue multiset TV raw / sign-normalised |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|')
    L.append(f"| {o['n']} | {fmt(o['balanced_accuracy'])} | {fmt(o['auc'])} | {fmt(o['ba_by_magnitude'][15])} | {fmt(o['ba_by_magnitude'][30])} | {fmt(o['ba_by_magnitude'][45])} | {fmt(o['ba_by_target_row']['top'])} | {fmt(o['ba_by_target_row']['bottom'])} | {fmt(o['cue_decode_accuracy'])} | {fmt(o['angle_error_deg']['median'])} / {fmt(o['angle_error_deg']['p95'])} | {fmt(o['no_cue_multiset_tv_distance'])} |\n")
    m=res['motion'];L.append('## Motion duration (cued patch, D0)\n')
    L.append('| n | BA hard vote | BA soft sum | per-frame direction agreement | ring decode | uncued patch vs own winner | uncued patch vs cued label |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|')
    L.append(f"| {m['n']} | {fmt(m['ba_hard_vote'])} | {fmt(m['ba_soft_sum'])} | {fmt(m['per_frame_direction_agreement'])} | {fmt(m['cue_decode_accuracy'])} | {fmt(m['uncued_patch_self_accuracy'])} | {fmt(m['uncued_patch_predicts_cued_label'])} |\n")
    L.append('| step (px) | BA soft | BA hard |');L.append('|---:|---:|---:|')
    for s in ('0.8','1.2','1.6'):L.append(f"| {s} | {fmt(m['ba_soft_by_step'][s])} | {fmt(m['ba_hard_by_step'][s])} |")
    L.append('');L.append('| count margin (winner minus runner-up frames) | trials | BA soft |');L.append('|---:|---:|---:|')
    for k,v in m['count_margin_histogram'].items():L.append(f"| {k} | {v} | {fmt(m['ba_soft_by_count_margin'][k])} |")
    L.append('');L.append('## Krauzlis cued motion change\n')
    L.append('| baseline | n | events t/f/c | AUC abs(dtheta) | BA at 13deg | BA best thr (deg) | est abs(dtheta) target | est abs(dtheta) unchanged (sd) | dprime | signed err mean (sd) | pre-angle err med / p95 | ring decode |')
    L.append('|---:|---:|---|---:|---:|---|---:|---|---:|---|---|---:|')
    for b,k in res['krauzlis'].items():
        ec=k['event_counts'];L.append(f"| {b} | {k['n']} | {ec.get('target',0)}/{ec.get('foil',0)}/{ec.get('catch',0)} | {fmt(k['auc_abs_delta'])} | {fmt(k['ba_at_13deg'])} | {fmt(k['ba_best_threshold'])} ({fmt(k['best_threshold_deg'])}) | {fmt(k['target_trials']['mean_abs_est'])} | {fmt(k['unchanged_cued_patch']['mean_abs_est'])} ({fmt(k['unchanged_cued_patch']['sd'])}) | {fmt(k['dprime'])} | {fmt(k['target_trials']['signed_error_mean'])} ({fmt(k['target_trials']['signed_error_sd'])}) | {fmt(k['pre_angle_error_deg']['median'])} / {fmt(k['pre_angle_error_deg']['p95'])} | {fmt(k['cue_decode_accuracy'])} |")
    L.append('');L.append('## Spatial binding (retrocue)\n')
    L.append('| delay | n | BA | ring decode | swapped pair recovered |');L.append('|---:|---:|---:|---:|---:|')
    for d in ('D0','D24'):
        b=res['binding'][d];L.append(f"| {d} | {b['n']} | {fmt(b['balanced_accuracy'])} | {fmt(b['cue_decode_accuracy'])} | {fmt(b['changed_pair_recovered'])} |")
    L.append('');L.append('No-cue leak: P(label=1 | swapped pair) over all binding trials, expected 0.5 everywhere.\n')
    L.append('| pair | n | P(label=1) |');L.append('|---|---:|---:|')
    for p,v in res['binding']['no_cue_pair_leak'].items():L.append(f"| {p} | {v['n']} | {fmt(v['p_label1'])} |")
    L.append('');L.append('## Image recognition\n')
    L.append('| condition | n | accuracy | BA | labels | probe hold identical | study unique | blanks gray | metadata hashes match |');L.append('|---|---:|---:|---|---|---:|---:|---:|---:|')
    for c,r in res['recognition'].items():
        if c=='manifest':continue
        L.append(f"| {c} | {r['n']} | {fmt(r['accuracy'])} | {fmt(r['balanced_accuracy']) if r['balanced_accuracy'] is not None else 'n/a (all negative)'} | {r['label_counts']} | {fmt(r['probe_hold_identical'])} | {fmt(r['study_unique'])} | {fmt(r['blanks_are_gray'])} | {fmt(r['metadata_hashes_match'])} |")
    mf=res['recognition']['manifest'];L.append(f"\nManifest: {mf['counts']}; sha256 disjoint across splits: {mf['sha_disjoint']}; base_id disjoint: {mf['base_id_disjoint']}.\n")
    L.append('## Frame roles per task (extreme conditions)\n')
    for k,v in res['frame_roles'].items():
        L.append(f"**{k}**: tensor frames {v['frames']}, frame_count {v['frame_count']}, indices in range {v['all_indices_in_range']}, last referenced {v['last_referenced']}.  ")
        L.append('Fields: '+', '.join(f'{a}={b}' for a,b in v['fields'].items())+'.  ')
        L.append('Non-gray pixels per frame: '+str(v['nongray_pixels_per_frame'])+'\n')
    path.write_text('\n'.join(L),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=63973001);ap.add_argument('--n-scale',type=float,default=1.);a=ap.parse_args()
    n=lambda k:max(8,int(k*a.n_scale))
    res=dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),seed=a.seed,split='val',stimuli_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
        observers_sha256=hashlib.sha256((HERE/'observers.py').read_bytes()).hexdigest())
    stream=S.SpatialBatteryStream(a.seed,'val')
    for name,fn in (('orientation',lambda:audit_orientation(stream,n(768))),('motion',lambda:audit_motion(stream,n(768))),('krauzlis',lambda:audit_krauzlis(stream,n(256))),
                    ('binding',lambda:audit_binding(stream,n(256),n(128))),('recognition',lambda:audit_recognition(stream,n(128))),('frame_roles',lambda:frame_roles(stream))):
        t=time.time();res[name]=fn();print(f'{name}: {time.time()-t:.1f}s',flush=True)
    (HERE/'results.json').write_text(json.dumps(res,indent=1,default=float),encoding='utf-8')
    write_report(res,HERE/'report.md');print('wrote',HERE/'report.md')

if __name__=='__main__':main()
