"""Stream and pipeline checks for the five-task battery (handoff section 4.5). CPU only.

Usage: python -m WorkingMemory.BatteryAudit.stream_checks
Writes stream_checks.json next to this file.
"""
from __future__ import annotations
import hashlib,json,subprocess,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from WorkingMemory.SpatialTaskBattery import stimuli as S

HERE=Path(__file__).resolve().parent
SEEDS=dict(train=61973001,scheduler=62973001,val=63973001,test=64973001)

def case_balance(n=10000):
    """Label and target-location counts from the pending-queue sampler alone (no rendering)."""
    out={}
    s=S.SpatialBatteryStream(SEEDS['train'],'train')
    for task in S.FAMILIES:
        labels=Counter();targets=Counter();joint=Counter();events=Counter();signs=Counter()
        for _ in range(n):
            label,target=s._case(task);labels[label]+=1;targets[target]+=1;joint[(label,target)]+=1
            if task=='krauzlis_cued_motion':events[s._event_kind]+=1
            if task=='orientation_cued':signs[s._cue_sign]+=1
        out[task]=dict(n=n,labels={str(k):v for k,v in sorted(labels.items())},targets={str(k):v for k,v in sorted(targets.items())},
            joint_min=min(joint.values()),joint_max=max(joint.values()),events={k:v for k,v in sorted(events.items())} or None,cue_signs={str(k):v for k,v in sorted(signs.items())} or None)
    return out

def batch_hashes(stream,plan):
    return [hashlib.sha256(stream.batch(n,task,cond)[0].numpy().tobytes()).hexdigest() for n,task,cond in plan]

PLAN=[(4,'orientation_cued',dict(delay=4)),(4,'motion_duration_cued',dict(delay=0)),(2,'krauzlis_cued_motion',dict(baseline_transitions=20)),
      (4,'spatial_binding',dict(delay=12)),(4,'image_recognition',dict(load=4,probe_hold=4)),(4,'orientation_cued',dict(delay=0))]

def state_roundtrip():
    """Save state after a prefix, draw a suffix, restore, redraw: the suffix must be byte-identical."""
    a=S.SpatialBatteryStream(SEEDS['val'],'val');batch_hashes(a,PLAN)
    saved=a.state_dict();json_saved=json.loads(json.dumps(saved))  # the resume path stores state through JSON
    suffix1=batch_hashes(a,PLAN)
    b=S.SpatialBatteryStream(SEEDS['val'],'val');b.load_state_dict(json_saved);suffix2=batch_hashes(b,PLAN)
    c=S.SpatialBatteryStream(SEEDS['val'],'val');c.load_state_dict(saved);suffix3=batch_hashes(c,PLAN)
    fresh=batch_hashes(S.SpatialBatteryStream(SEEDS['val'],'val'),PLAN)
    return dict(suffix_identical_after_json_roundtrip=suffix1==suffix2,suffix_identical_after_direct_roundtrip=suffix1==suffix3,
        fresh_stream_differs_from_suffix=fresh!=suffix1,counts_restored=b.counts==a.counts and b.pending==a.pending)

def cross_process(split):
    code=('import json,sys;sys.path.insert(0,%r);from WorkingMemory.BatteryAudit.stream_checks import *;'
          's=S.SpatialBatteryStream(SEEDS[%r],%r);print(json.dumps(batch_hashes(s,PLAN)))')%(str(HERE.parents[1]),split,split)
    runs=[json.loads(subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True,cwd=str(HERE.parents[1])).stdout) for _ in range(2)]
    local=batch_hashes(S.SpatialBatteryStream(SEEDS[split],split),PLAN)
    return dict(two_subprocesses_identical=runs[0]==runs[1],subprocess_matches_in_process=runs[0]==local,first_hash=local[0][:16])

def frame_counts():
    """Rendered length versus frame_count for every train and eval condition."""
    s=S.SpatialBatteryStream(SEEDS['val'],'val');bad=[];checked=0
    for task,conds in S.TRAIN_CONDITIONS.items():
        for c in conds:
            x,y,m=s.batch(1,task,c);checked+=1
            if x.shape[1]!=S.frame_count(task,c) or m[0]['frame_count']!=x.shape[1]:bad.append((task,c,x.shape[1],S.frame_count(task,c)))
    for name,spec in S.EVAL_CONDITIONS.items():
        x,y,m=s.batch(1,spec['task'],spec['condition']);checked+=1
        if x.shape[1]!=S.frame_count(spec['task'],spec['condition']):bad.append((name,x.shape[1]))
    return dict(conditions_checked=checked,mismatches=bad)

def split_isolation():
    """Train/val/test streams with their seeds must not share recognition photographs, and per-task RNGs must differ."""
    out={}
    for split in ('train','val','test'):
        s=S.SpatialBatteryStream(SEEDS[split],split);x,y,m=s.batch(8,'image_recognition',dict(load=4,probe_hold=3))
        out[split]=sorted({i for r in m for i in r['study_source_ids']+[r['probe_source_id']]})
    names=list(out)
    return dict(source_ids_disjoint=all(not(set(out[a])&set(out[b])) for a in names for b in names if a<b),sample_ids={k:v[:5] for k,v in out.items()})

def main():
    res=dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stimuli_sha256=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),version=S.VERSION)
    for name,fn in (('case_balance',case_balance),('state_roundtrip',state_roundtrip),('cross_process_val',lambda:cross_process('val')),('cross_process_test',lambda:cross_process('test')),
                    ('frame_counts',frame_counts),('split_isolation',split_isolation)):
        t=time.time();res[name]=fn();print(f'{name}: {time.time()-t:.1f}s -> {json.dumps(res[name],default=str)[:300]}',flush=True)
    (HERE/'stream_checks.json').write_text(json.dumps(res,indent=1,default=str),encoding='utf-8')

if __name__=='__main__':main()
