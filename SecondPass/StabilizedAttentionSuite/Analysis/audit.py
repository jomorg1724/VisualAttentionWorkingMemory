"""Independent read-only completeness, pairing, projection and checkpoint audit."""
import json,hashlib
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
from .design import OUT,ROOT,CHECKPOINT,STRENGTHS,interventions
from .report import load_rows,project
from SecondPass.TaskSuite.suite import TASKS
from SecondPass.PredictiveConsistencyVAE.queue import process_identity,FileLock,atomic_json

def main():
    complete=json.loads((OUT/'completion.json').read_text());assert complete['status']=='inference_complete'
    assert json.loads((OUT/'exit.json').read_text())['exit_code']==0
    owner=json.loads((OUT/'owner.json').read_text());worker=json.loads((OUT/'active_child.json').read_text())['identity']
    assert process_identity(worker['pid'])!=worker
    p=load_rows(OUT/'psychometric_trials.jsonl');i=load_rows(OUT/'intervention_trials.jsonl')
    assert set(p.task)==set(TASKS)
    assert len(p)==33568,('psychometric rows',len(p))
    assert len(i)==83168,('intervention rows',len(i))
    zero=p[(p.task=='krauzlis_cued_motion')&(p.strength==0)].pivot(index='cluster',columns='event',values='p1')
    assert np.array_equal(zero.target,zero.foil) and np.array_equal(zero.target,zero['catch']),'Zero-angle scheduled labels must not affect model responses'
    for task in ('krauzlis_cued_motion','orientation_cued'):
        g=i[i.task==task];assert set(g.arm)=={a['name'] for a in interventions(task)}
        sizes=g.groupby('arm').size();assert sizes.nunique()==1
        assert sizes.iloc[0]==(1056 if task=='krauzlis_cued_motion' else 1024)
        counts=g.groupby(['cluster','strength','event']).arm.nunique();assert (counts==len(interventions(task))).all()
        h=g.groupby(['cluster','strength','event']).pixel_batch_sha256.nunique();assert (h==1).all(),'Different pixels across paired interventions'
        # All global and zero-length-delay controls must preserve choices.
        pivot=g.pivot(index=['cluster','strength','event'],columns='arm',values='choice')
        for arm in ('global_-12_all','global_+12_all'):assert np.array_equal(pivot[arm],pivot.sham)
        if task=='orientation_cued':
            d=g[g.condition=='0'].pivot(index=['cluster','strength','event'],columns='arm',values='choice')
            for site in ('target','distractor'):
                for dose in (-8,8):assert np.array_equal(d[f'{site}_{dose:+d}_delay'],d.sham)
    entries=json.loads((OUT/'atlas/index.json').read_text());cells={(e['task'],e['cell']) for e in entries}
    assert cells=={(t,c['id']) for t,s in TASKS.items() for c in s['conditions']}
    alltime=0
    for e in entries:
        key=e['id'];d=np.load(OUT/'atlas'/(key+'.npz'));t=e['timesteps'];alltime+=t
        assert d['images'].shape==(t,3,100,100) and d['attention'].shape==(t,10,10)
        if e['task']=='image_recognition':
            m=e['metadata'];imgs=d['images'];study=[hashlib.sha256(np.ascontiguousarray(imgs[k]).tobytes()).hexdigest() for k in m['study_frames']]
            assert study==m['study_raster_sha256'] and len(set(study))==m['study_load']
            for k in m['probe_frames']:assert hashlib.sha256(np.ascontiguousarray(imgs[k]).tobytes()).hexdigest()==m['probe_raster_sha256']
            assert (m['probe_raster_sha256'] in study)==bool(e['label'])
            assert np.all(imgs[m['blank_frames']]==.5)
        for field in ('attention','mean_attention'):
            a=d[field];assert np.isfinite(a).all() and (a>=0).all() and np.allclose(a.sum((1,2)),1,atol=1e-5)
        assert e['n']>=1 and len(e['phases'])==t
        for mode,ik,ak in [('example','images','attention'),('mean','mean_images','mean_attention')]:
            im=np.array(Image.open(OUT/'gallery/assets'/(key+'--'+mode+'.png')).convert('RGB'));assert im.shape==(300,t*100,3)
            raw=np.clip(d[ik].transpose(0,2,3,1),0,1);raw=(raw*255).round().astype('uint8');heat=project(d[ak]);over=(.58*raw+.42*heat).round().astype('uint8')
            expected=np.concatenate([np.concatenate(v,axis=1) for v in (raw,over,heat)],axis=0)
            assert np.array_equal(im,expected),'Sprite pixels differ from saved model arrays'
        assert len(list((OUT/'timesteps').glob(key+'--*.png')))==(t+9)//10
    assert set(p.stem for p in (OUT/'animations').glob('*.gif'))==set(TASKS)
    gif_frames={t:2 for t in TASKS};gif_frames.update(orientation_ring=4,orientation_cued=16,motion_duration_cued=23,krauzlis_cued_motion=37,spatial_binding=17,image_recognition=20)
    for task,nframes in gif_frames.items():
        with Image.open(OUT/'animations'/(task+'.gif')) as gif:
            assert gif.n_frames==nframes,(task,gif.n_frames,nframes)
            for k in range(nframes):gif.seek(k);assert gif.info['duration']==400
    proto=json.loads((OUT/'protocol.json').read_text())
    for name,wanted in proto['checkpoint_sha256'].items():assert hashlib.sha256(CHECKPOINT.with_name(name+'.pt').read_bytes()).hexdigest()==wanted
    for path,wanted in proto['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==wanted,path
    for name,wanted in proto['analysis_source_hashes'].items():assert hashlib.sha256((OUT.parent/name).read_bytes()).hexdigest()==wanted,name
    assert complete['presentations']==len(p)+len(i)+35*256
    assert complete['groups']==len(entries)
    lock=FileLock(CHECKPOINT.parents[4]/'LocalRuntime/local_gpu_worker.lock');lock.close()
    result=dict(status='complete_independently_verified',psychometric_trials=len(p),intervention_presentations=len(i),atlas_presentations=35*256,total_presentations=complete['presentations'],tasks=13,native_cells=35,attention_groups=len(entries),group_timesteps=alltime,paired_pixels_identical=True,global_shift_choices_unchanged=True,zero_delay_noop_choices_unchanged=True,zero_angle_event_labels_identical_responses=True,all_exported_sprite_pixels_verified=True,native_recognition_frame_hashes_verified=True,all_maps_normalized=True,all_original_checkpoints_hash_unchanged=True,worker_exited=True,gpu_lock_free=True)
    result.update(all_frozen_source_hashes_unchanged=True,all_inference_source_hashes_unchanged=True,all_model_tensors_bitwise_unchanged=complete['all_model_tensors_bitwise_unchanged'])
    atomic_json(OUT/'completion_verified.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
