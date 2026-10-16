import copy,hashlib
import numpy as np
import pytest
import torch
from .dataset import TASK,TASKS,CATALOG,CONDITIONS,POPOUT,POPOUT_CONDITIONS,MOTION,MOTION_CONDITIONS,Stream,all_cells
from .model import PARENT,ParentModel,initialize
from .stimuli import CENTERS
from .train import Scheduler,update,monitoring_cells

torch.set_num_threads(2)

def test_all_conditions_exact_match_and_cue_switch():
    stream=Stream('test')
    for cell in CONDITIONS:
        x,y,meta=stream.paired_queries(cell['id'],37);c=cell['kwargs'];onset=1+c['sample_frames']+c['delay']
        assert x.shape==(4,onset+c['exposure'],3,100,100) and y.tolist()==[0,1,2,3]
        assert x.dtype==torch.float32 and x.min()>=0 and x.max()<=1
        assert torch.equal(x[:,onset:],x[0:1,onset:].expand_as(x[:,onset:]))
        assert len(set(meta[0]['sprite_sha256']))==4
        for query in range(4):
            sample=x[query,1,:,36:64,36:64]
            for slot,(cx,cy) in enumerate(CENTERS):
                assert torch.equal(sample,x[query,onset,:,cy-14:cy+14,cx-14:cx+14])==(slot==query)
        assert torch.all(x[:,0]==128/255)
        assert torch.equal(x[:,1:1+c['sample_frames']],x[:,1:2].expand(-1,c['sample_frames'],-1,-1,-1))
        assert torch.equal(x[:,onset:],x[:,onset:onset+1].expand(-1,c['exposure'],-1,-1,-1))
        assert meta[0]['phases']==['fixation']+['sample']*c['sample_frames']+['delay']*c['delay']+['search']*c['exposure']
        if c['delay']:assert torch.all(x[:,1+c['sample_frames']:onset]==128/255)

def test_histograms_and_conjunctions():
    stream=Stream()
    for family in ('features','fractals'):
        x,_,meta=stream.paired_queries(f'{family}_similar_S2_D0_H2',11)
        if family=='features':
            for feature in ('color','shape','texture'):
                values=[d[feature] for d in meta[0]['object_descriptors']]
                assert sorted(np.unique(values,return_counts=True)[1])==[2,2]
        else:
            tiles=[torch.sort(x[0,3,:,cy-14:cy+14,cx-14:cx+14].flatten()).values for cx,cy in CENTERS]
            assert all(torch.equal(tiles[0],t) for t in tiles[1:])

def test_balanced_fresh_split_identities_and_restore():
    cell='objects_similar_S2_D4_H2';s=Stream();s.batch(3,TASK,cell);state=s.state_dict()
    a=s.batch(5,TASK,cell);r=Stream();r.load_state_dict(state);b=r.batch(5,TASK,cell)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
    n=np.random.get_state();t=torch.get_rng_state().clone();s=Stream();x,y,m=s.batch(12,TASK,cell)
    assert torch.bincount(y).tolist()==[3]*4
    assert np.array_equal(n[1],np.random.get_state()[1]) and torch.equal(t,torch.get_rng_state())
    hashes=[h for row in m for h in row['sprite_sha256']]
    assert len(set(hashes))==48
    _,_,other=Stream('test').batch(12,TASK,cell)
    assert not set(hashes)&{h for row in other for h in row['sprite_sha256']}
    with pytest.raises(ValueError):Stream('val').load_state_dict(state)

def test_legacy_streams_restore_and_unmaterialized_cells():
    s=Stream();s.batch(1,'orientation','mixed');s.batch(1,'vda4','TH26');state=s.state_dict()
    restored=Stream();restored.load_state_dict(state)
    for task,cell in [('orientation','mixed'),('vda4','TH26'),('contrast','mixed'),('vda9','TH8')]:
        a=s.batch(2,task,cell);b=restored.batch(2,task,cell)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]

def test_scheduler_and_catalog():
    assert len(TASKS)==19 and len(all_cells())==437 and len(CONDITIONS)==288 and len(POPOUT_CONDITIONS)==24
    scheduler=Scheduler();assert {scheduler.next()[0] for _ in range(19)}==set(TASKS)
    assert TASKS[TASK]['classes']==4
    panel=monitoring_cells()
    assert len(panel)==len(set(panel))==71 and {t for t,c in panel}==set(TASKS)
    assert set(panel).issubset(set(all_cells()))

def test_migration_and_disposable_cpu_update():
    before=hashlib.sha256(PARENT.read_bytes()).hexdigest();saved=torch.load(PARENT,map_location='cpu',weights_only=False)
    assert saved['step']==16000
    model,opt=initialize(PARENT);old=ParentModel();old.load_state_dict(saved['model'])
    old_opt=torch.optim.Adam(old.parameters());old_opt.load_state_dict(saved['optimizer'])
    new_named=dict(model.named_parameters())
    for name,p in old.named_parameters():
        assert torch.equal(p,new_named[name])
        for key,value in old_opt.state[p].items():
            assert torch.equal(value,opt.state[new_named[name]][key]) if torch.is_tensor(value) else value==opt.state[new_named[name]][key]
    assert len(opt.state)==86 and len(list(model.parameters()))==92
    x,y,meta=Stream().batch(4,TASK,'fractals_similar_S2_D0_H2')
    with torch.no_grad():
        assert torch.equal(model(x,'vda4'),old(x,'vda4'))
        logits=model(x,TASK);assert logits.shape==(4,4) and torch.isfinite(logits).all()
    weights=copy.deepcopy(model.state_dict());metrics=update(model,opt,x,y,meta,TASK,micro=2)
    assert np.isfinite(metrics['loss']) and len(opt.state)==88
    assert not torch.equal(weights[f'heads.{TASK}.weight'],model.heads[TASK].weight)
    assert torch.equal(weights['heads.vda4.weight'],model.heads.vda4.weight)
    assert torch.equal(weights[f'heads.{POPOUT}.weight'],model.heads[POPOUT].weight)
    x,y,meta=Stream().batch(4,POPOUT,'orientation_X15_H4')
    metrics=update(model,opt,x,y,meta,POPOUT,micro=2)
    assert np.isfinite(metrics['loss']) and len(opt.state)==90
    assert not torch.equal(weights[f'heads.{POPOUT}.weight'],model.heads[POPOUT].weight)
    assert torch.equal(weights[f'heads.{MOTION}.weight'],model.heads[MOTION].weight)
    x,y,meta=Stream().batch(4,MOTION,'motion_A45_S3_D0_H3')
    metrics=update(model,opt,x,y,meta,MOTION,micro=2)
    assert np.isfinite(metrics['loss']) and len(opt.state)==92
    assert not torch.equal(weights[f'heads.{MOTION}.weight'],model.heads[MOTION].weight)
    for p,state in opt.state.items():
        for k in ('exp_avg','exp_avg_sq'):assert torch.isfinite(state[k]).all() and state[k].shape==p.shape
    assert before==hashlib.sha256(PARENT.read_bytes()).hexdigest()


def test_duration_counterfactuals():
    s=Stream('test')
    for family in ('features','objects','fractals'):
        a=s.trial(f'{family}_similar_S1_D0_H1',17)
        b=s.trial(f'{family}_similar_S8_D12_H8',17)
        assert a[1]==b[1] and torch.equal(a[0][1],b[0][1]) and torch.equal(a[0][-1],b[0][-1])
        assert a[0].shape[0]==3 and b[0].shape[0]==29
    for feature,difference in [('color',36),('orientation',15)]:
        a=s.popout_trial(f'{feature}_X{difference}_H1',17)
        b=s.popout_trial(f'{feature}_X{difference}_H8',17)
        assert a[1]==b[1] and torch.equal(a[0][-1],b[0][-1])


def test_popout_singleton_labels_timing_and_restore():
    for condition in POPOUT_CONDITIONS:
        s=Stream();cell=condition['id'];x,y,meta=s.batch(4,POPOUT,cell)
        assert sorted(y.tolist())==list(range(4))
        assert x.shape==(4,1+condition['kwargs']['exposure'],3,100,100)
        assert torch.all(x[:,0]==128/255)
        assert torch.equal(x[:,1:],x[:,1:2].expand_as(x[:,1:]))
        for trial in range(4):
            tiles=[x[trial,1,:,cy-14:cy+14,cx-14:cx+14] for cx,cy in CENTERS]
            distractors=[tile for i,tile in enumerate(tiles) if i!=y[trial]]
            assert all(torch.equal(t,distractors[0]) for t in distractors)
            assert not torch.equal(tiles[y[trial]],distractors[0])
            assert meta[trial]['phases']==['fixation']+['search']*condition['kwargs']['exposure']
            assert meta[trial]['sample_frames']==0
        other=Stream('test').batch(4,POPOUT,cell)
        assert not {m['array_sha256'] for m in meta}&{m['array_sha256'] for m in other[2]}
        state=s.state_dict();a=s.batch(2,POPOUT,cell);restored=Stream();restored.load_state_dict(state);b=restored.batch(2,POPOUT,cell)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]


def test_motion_conditions_and_counterfactual_queries():
    assert len(MOTION_CONDITIONS)==81
    for condition in MOTION_CONDITIONS:
        x,y,m=Stream('test').paired_motion_queries(condition['id'],17);c=condition['kwargs']
        onset=1+c['sample_frames']+c['delay']
        assert x.shape==(4,onset+c['exposure'],3,100,100) and y.tolist()==[0,1,2,3]
        assert torch.isfinite(x).all() and x.min()>=0 and x.max()<=1
        assert torch.equal(x[:,onset:],x[0:1,onset:].expand_as(x[:,onset:]))
        assert torch.equal(x[:,1],x[0:1,1].expand_as(x[:,1]))  # first frame carries no query direction
        assert torch.all(x[:,0]==128/255) and torch.all(x[:,1+c['sample_frames']:onset]==128/255)
        assert len({row['sample_direction'] for row in m})==4
        for q in range(4):
            assert m[q]['sample_direction']==m[q]['directions'][q]
            assert m[q]['phases']==['fixation']+['sample']*c['sample_frames']+['delay']*c['delay']+['search']*c['exposure']
            assert not torch.equal(x[q,1],x[q,2])
            for cx,cy in CENTERS:
                assert not torch.equal(x[q,1,:,36:64,36:64],x[q,onset,:,cy-14:cy+14,cx-14:cx+14])
                assert not torch.equal(x[q,onset,:,cy-14:cy+14,cx-14:cx+14],x[q,onset+1,:,cy-14:cy+14,cx-14:cx+14])
            for other in range(q):assert not torch.equal(x[q,2],x[other,2])


def test_motion_matched_durations_freshness_balance_and_restore():
    s=Stream();short=s.motion_trial('motion_A20_S3_D0_H3',29);long=s.motion_trial('motion_A20_S12_D12_H12',29)
    assert short[1]==long[1] and torch.equal(short[0][1:4],long[0][1:4])
    assert torch.equal(short[0][-3:],long[0][25:28])
    cell='motion_A90_S6_D4_H6';x,y,m=s.batch(8,MOTION,cell)
    assert torch.bincount(y).tolist()==[2]*4
    other=Stream('test').batch(8,MOTION,cell)
    assert not {r['array_sha256'] for r in m}&{r['array_sha256'] for r in other[2]}
    state=s.state_dict();a=s.batch(2,MOTION,cell);restored=Stream();restored.load_state_dict(state);b=restored.batch(2,MOTION,cell)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]


def test_motion_pixels_encode_signed_translation():
    from .motion import dot_movie
    right=dot_movie([1,2,3],0,6);left=dot_movie([1,2,3],180,6);down=dot_movie([1,2,3],90,6)
    assert np.array_equal(right[0],left[0]) and np.array_equal(right[0],down[0])
    # Three-pixel displacement over two intervals, checked inside the aperture.
    assert np.array_equal(right[2,7:21,7:21],right[0,7:21,4:18])
    assert np.array_equal(left[2,7:21,7:21],left[0,7:21,10:24])
    assert np.array_equal(down[2,7:21,7:21],down[0,4:18,7:21])
