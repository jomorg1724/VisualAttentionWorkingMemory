"""Focused CPU checks for sequence labels, causal rendering and cue contracts."""
import collections
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from WorkingMemory.stimuli import (
    SequenceStream,FAMILIES,BRIDGE_CONDITIONS,TRAIN_CONDITIONS,CONDITIONS,
    condition_frames,setting,scalar_oracle,duration_oracle,visual_cues,
)


def main():
    torch.set_num_threads(2);start=time.monotonic();stream=SequenceStream(941121,'val')
    results=dict(device='cpu',families={},protocol='wm_visual_sequences_v1')
    for family in FAMILIES:
        shapes={}
        for condition in ('anchor','bridge_integrate_visible','integrate_L8','recall_N2','decision_distractor'):
            images,labels,metadata=stream.batch(4,family,condition)
            assert images.shape==(4,condition_frames(condition,family),3,100,100)
            assert torch.isfinite(images).all() and images.min()>=0 and images.max()<=1
            for label,meta in zip(labels.tolist(),metadata):
                assert label==meta['label'] and meta['encoder_updates']==images.shape[1]
                if meta['protocol']=='integration':
                    if family=='motion_direction':
                        observed,counts=duration_oracle(meta['directions']);assert observed==label
                        assert sum(counts)==meta['condition_config']['length']
                    elif family=='contour':
                        evidence=sum(s==meta['sign'] for s in meta['occupancy_signs'])
                        assert evidence==meta['evidence'] and int(evidence>meta['threshold'])==label
                    else:
                        observed,evidence=scalar_oracle(meta['levels'],meta['sign'],meta['threshold'])
                        assert observed==label and evidence==meta['evidence']
                if meta['protocol']=='recall':
                    assert len(set(meta['identities']))==2
                    target=meta['target_index'];value=meta['values'][target]
                    if family=='motion_direction':assert value==label
                    else:assert int(value!=meta['probe_value'])==label
                    assert meta['target_identity']==meta['identities'][target]
                    assert meta['target_age_at_report']==images.shape[1]-1-meta['target_last_sample_frame']
            shapes[condition]=list(images.shape)
        # State restore reproduces exactly the next images, labels and records.
        saved=stream.state_dict();a=stream.batch(2,family,'recall_N2')
        stream.load_state_dict(saved);b=stream.batch(2,family,'recall_N2')
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
        results['families'][family]=dict(shapes=shapes,oracles=True,checkpoint_replay=True)
    # Every main condition fits the declared same-length batching contract.
    for name,condition in {**BRIDGE_CONDITIONS,**TRAIN_CONDITIONS}.items():
        x,_,_=stream.batch(1,'orientation',name)
        assert x.shape[1]==condition_frames(condition,'orientation')
    results['all_named_main_conditions_sampled']=True
    # True motion cohorts: images are one continuous render across transitions.
    images,labels,meta=stream.matched_motion_pair()
    assert labels.tolist()==[0,1]
    assert meta[0]['net_displacement']==meta[1]['net_displacement']
    assert meta[0]['final_direction']==meta[1]['final_direction']
    # Same RNG and four-transition suffix erase old two-cohort dot positions.
    assert torch.equal(images[0,-2],images[1,-2])
    assert torch.equal(images[0,-1],images[1,-1])
    results['motion_matched_net_and_visible_suffix_different_winners']=True
    results['motion_matched_example']=[dict(label=m['label'],counts=m['duration_counts'],net=m['net_displacement'],suffix=m['directions'][-4:]) for m in meta]
    # Exact four-way counterbalance at fixed latent/nuisance draw. This verifies
    # both relevant-value marginals and the actual sample/probe raster marginals.
    for family in FAMILIES:
        if family=='motion_direction':continue
        primitive=stream.streams[family];saved=primitive.state_dict();cases={}
        for label,bit in ((0,0),(0,1),(1,0),(1,1)):
            primitive.load_state_dict(saved)
            frames,meta=stream._recall(primitive,family,label,setting(protocol='recall',delay=0,post_delay=0),bit)
            cases[label,bit]=(frames,meta)
        for bit in (0,1):
            assert np.array_equal(cases[0,bit][0][1],cases[1,bit][0][1])
            assert np.array_equal(cases[0,bit][0][-1],cases[1,1-bit][0][-1])
        results['families'][family]['exact_AA_BB_AB_BA_raster_marginals']=True
    # Six unique identity glyphs and sign glyphs; same glyph at sample/query.
    blank=np.full((3,100,100),.5,np.float32)
    identities=[visual_cues(blank,'recall','sample',identity=i)[:,-10:,-10:] for i in range(6)]
    assert len({x.tobytes() for x in identities})==6
    for i in range(6):assert np.array_equal(identities[i],visual_cues(blank,'recall','query',identity=i)[:,-10:,-10:])
    assert not np.array_equal(visual_cues(blank,'integration',sign=-1),visual_cues(blank,'integration',sign=1))
    areas=[float((.5-visual_cues(blank,'integration',threshold=t)[:,97:99,91:99]).sum()) for t in (.6,.9,2.4,3.6,4.8,7.2,9.6,14.4,40)]
    assert all(a<b for a,b in zip(areas,areas[1:]))
    results['cue_and_identity_mappings']=True
    results['threshold_bar_absolute_monotone']=True
    # Natural gain uses the full fixed candidate set, independent of probe.
    primitive=stream.streams['natural_spectrum'];context=stream._context(primitive,'natural_spectrum')
    bank=context['bank'];assert bank.min()>=0 and bank.max()<=1
    assert np.max(np.abs(bank.mean((1,2))-.5))<1e-6
    assert np.max(bank.std((1,2)))-np.min(bank.std((1,2)))<1e-6
    results['natural_fixed_candidate_bank_mean_rms_gain']=True
    results.update(status='passed',wall_seconds=time.monotonic()-start,
                   maximum_main_frames=max(condition_frames(c,f) for c in TRAIN_CONDITIONS.values() for f in FAMILIES),
                   adaptations=['Threshold bar fixed logarithmic mapping of absolute level-unit amount; two trained levels per length.',
                   'Nine scalar levels; orientation7.5deg, logcontrast ln4.5/8, logfrequency log2(2.5)/8, chromatic.03, natural negativebeta.15.',
                   'Other-item nuisance coordinate system shared within recall except natural item-specific crops; lures transfer beta to target crop.',
                   'New motion continuous alternating age cohorts; matched-net/suffix diagnostic is a paired group, not independent episodes.',
                   'Natural anchor also uses fixed full candidate-bank photometric gain.'])
    target=Path(__file__).with_name('stimulus_checks.json');target.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in results.items() if k!='families'},indent=2))


if __name__=='__main__':main()
