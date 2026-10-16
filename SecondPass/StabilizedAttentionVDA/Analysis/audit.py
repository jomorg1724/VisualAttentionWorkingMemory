"""Independent completion, pairing, control and exported-pixel audit."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from .design import OUT,CHECKPOINT,PHASES
from .report import rows,pixels,map_pixels,overlay
from SecondPass.PixelContrastiveVAE.train import GPULock

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    expected=dict(primary=8064,native=4608,psychometrics=18144,interventions=100800)
    counts={};control_count=0
    for phase,wanted in expected.items():
        df=rows(phase);counts[phase]=len(df);assert len(df)==wanted,(phase,len(df),wanted)
        assert df.p1.between(0,1).all() and df.choice.isin([0,1]).all()
        if phase in ('primary','interventions'):
            keys=['size','validity','event','angle','rep']+(['geometry'] if phase=='interventions' else [])
            assert df.groupby(keys).pixels_sha256.nunique().max()==1
            if phase=='primary':
                assert df.groupby(['arm','event']).size().eq(384).all()
                assert df['size'].eq(4).all() and df.cue.eq(3).all() and df.validity.eq(1).all()
                changes=df[df.event=='uncued'];assert changes.change_index.eq(0).all() and changes.label.eq(1).all()
                import pandas as pd
                effects=pd.read_csv(OUT/'primary_effects.csv')
                for _,effect in effects.iterrows():
                    hit=df[(df.arm==effect.arm)&(df.event==effect.event)].choice.mean()
                    fa=df[(df.arm==effect.arm)&(df.event=='catch')].choice.mean()
                    assert np.isclose(effect.hit_rate,hit) and np.isclose(effect.false_alarm_rate,fa)
                    assert np.isclose(effect.ba,.5*(hit+1-fa))
                for _,g in df.groupby(keys):
                    baseline=g[g.arm=='sham'].iloc[0].attention_mass
                    for _,row in g.iterrows():
                        mass=row.attention_mass
                        for site in baseline:assert np.allclose(mass[site][:5],baseline[site][:5],atol=1e-7)
                        if row.arm!='sham':
                            site,dose,_=row.arm.split('_');dose=float(dose)
                            # At the first manipulated frame, recurrent history is
                            # identical, so the independent softmax formula applies.
                            original=baseline[site][5]
                            denom=1-original+np.exp(dose)*original
                            predicted=np.exp(dose)*original/denom
                            assert np.isclose(mass[site][5],predicted,atol=2e-6)
            else:
                base=df[df.arm=='sham'].set_index(keys)
                for name in ('global_-8_all','global_+8_all'):
                    control=df[df.arm==name].set_index(keys).reindex(base.index)
                    assert np.array_equal(base.choice,control.choice)
                    assert np.allclose(base.p1,control.p1,atol=2e-5)
                    control_count+=len(control)
        if phase=='native':assert df.groupby(['size','theta']).size().eq(512).all()
    completion=json.loads((OUT/'completion.json').read_text());assert completion['presentations']==sum(counts.values())
    assert completion['model_tensors_bitwise_unchanged'] and completion['checkpoint_hashes_unchanged']
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,wanted in protocol['checkpoint_hashes'].items():assert sha(CHECKPOINT.with_name(name+'.pt'))==wanted
    for name,wanted in protocol['analysis_sources'].items():assert sha(Path(__file__).with_name(name))==wanted
    index=json.loads((OUT/'atlas/index.json').read_text());assert len(index)==90
    checked=0
    for entry in index:
        key=entry['id'];z=np.load(OUT/'atlas'/f'{key}.npz')
        assert entry['n']==48 and entry['phases']==PHASES
        assert z['images'].shape==(3,7,3,100,100) and z['attention'].shape==(3,7,10,10)
        assert np.allclose(z['attention'].sum((-2,-1)),1,atol=1e-5)
        assert np.allclose(z['mean_attention'].sum((-2,-1)),1,atol=1e-5)
        for j in range(3):
            for name,frames in [('image',pixels(z['images'][j])),('map',map_pixels(z['attention'][j])),('overlay',overlay(z['images'][j],z['attention'][j]))]:
                actual=np.array(Image.open(OUT/'gallery/assets'/f'{key}_e{j}_{name}.png'))
                assert np.array_equal(actual,np.concatenate(list(frames),axis=1));checked+=1
        for name,frames in [('image',pixels(z['mean_images'])),('map',map_pixels(z['mean_attention'])),('overlay',overlay(z['mean_images'],z['mean_attention']))]:
            actual=np.array(Image.open(OUT/'gallery/assets'/f'{key}_mean_{name}.png'))
            assert np.array_equal(actual,np.concatenate(list(frames),axis=1));checked+=1
    for size in (4,9,16):
        gif=Image.open(OUT/'figures'/f'attention_vda{size}.gif');assert gif.n_frames==7
    exit_info=json.loads((OUT/'exit.json').read_text());assert exit_info['exit_code']==0
    lock=GPULock(Path(__file__).resolve().parents[3]/'LocalRuntime/local_gpu_worker.lock');lock.close()
    result=dict(status='complete_and_verified',time=time.time(),counts=counts,total=sum(counts.values()),
        identical_paired_pixels=True,primary_reported_rates_verified=True,primary_premanipulation_attention_unchanged=True,first_change_softmax_formula_verified=True,global_control_presentations=control_count,
        global_control_choices_unchanged=True,atlas_groups=90,atlas_timesteps=630,exported_sprites_pixel_checked=checked,
        gifs=3,gif_frames_each=7,checkpoint_hashes_unchanged=True,model_tensors_unchanged=True,
        analysis_source_hashes_match=True,worker_exit_code=0,gpu_lock_free=True,
        report_source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('report.py'),Path(__file__).with_name('viewer.js')]})
    (OUT/'completion_verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
