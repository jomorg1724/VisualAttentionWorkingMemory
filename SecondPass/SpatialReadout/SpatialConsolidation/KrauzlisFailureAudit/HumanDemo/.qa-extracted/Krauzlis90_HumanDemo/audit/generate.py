"""Renderer-only native Krauzlis90 export; no model imports or checkpoint reads."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, json, hashlib, subprocess, base64, collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path[:0]=[str(HERE/'.deps'),str(ROOT)]
import numpy as np
import torch
from PIL import Image
import imageio_ffmpeg
from SecondPass.SpatialReadout.SpatialConsolidation.Krauzlis90Fresh.stimuli import SpatialBatteryStream
torch.set_num_threads(1)
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
TASK='krauzlis_cued_motion'
OUT=HERE/'Krauzlis90_HumanDemo'
for folder in ('videos','sprites','native','audit'):(OUT/folder).mkdir(parents=True,exist_ok=True)
def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2))
# Bounded selection only by metadata: no perceptual/ease screening, no pixel edits.
requirements={12:[('target',0),('target',1),('foil',0),('catch',1)],20:[('target',0),('target',1),('foil',1)],28:[('target',0),('target',1),('foil',0)]}
selected=[];curation=[]
for b,needed in requirements.items():
    seed=2026100200+b
    stream=SpatialBatteryStream(seed,'test')
    wanted=collections.Counter(needed)
    for ordinal in range(300):
        x,y,meta=stream.batch(1,TASK,dict(baseline_transitions=b)); m=meta[0]
        key=(m['event_type'],m['target_location'])
        if wanted[key]:
            selected.append((x[0].numpy().copy(),m,seed,ordinal));wanted[key]-=1
        if not +wanted:break
    assert not +wanted,'Bounded curation exhausted'
    curation.append(dict(baseline=b,seed=seed,draws=ordinal+1,bound=300))
np.random.default_rng(2026100299).shuffle(selected)
records=[];assets=[]
for i,(native,m,seed,ordinal) in enumerate(selected,1):
    name=f'trial{i:02d}';n=len(native)
    assert n==m['baseline_transitions']+17 and n in (29,37,45)
    # Fresh exact replay from seed, never reconstruct or adjust the pixels.
    replay=SpatialBatteryStream(seed,'test')
    for _ in range(ordinal+1):rx,ry,rm=replay.batch(1,TASK,m['condition'])
    assert np.array_equal(rx[0].numpy(),native) and rm[0]==m
    uint=np.rint(native.transpose(0,2,3,1)*255).clip(0,255).astype(np.uint8)
    np.savez_compressed(OUT/'native'/f'{name}.npz',frames=native)
    sprite=np.concatenate(list(uint),axis=1)
    spritepath=OUT/'sprites'/f'{name}.png';Image.fromarray(sprite).save(spritepath)
    restored=np.asarray(Image.open(spritepath)).reshape(100,n,100,3).transpose(1,0,2,3)
    assert np.array_equal(restored,uint)
    video=OUT/'videos'/f'{name}.mp4'
    enlarged=np.repeat(np.repeat(uint,6,axis=1),6,axis=2)
    cmd=[FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','600x600','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','slow','-crf','10','-pix_fmt','yuv420p','-r','10','-frames:v',str(n),'-movflags','+faststart',str(video)]
    subprocess.run(cmd,input=enlarged.tobytes(),check=True)
    decoded=subprocess.run([FFMPEG,'-v','error','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
    decoded=np.frombuffer(decoded,np.uint8).reshape(-1,600,600,3)
    assert len(decoded)==n
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');info=next(reader);reader.close()
    assert info['fps']==10 and tuple(info['size'])==(600,600) and abs(info['duration']-n/10)<.011
    crc=subprocess.run([FFMPEG,'-v','error','-i',str(video),'-f','framecrc','pipe:1'],capture_output=True,check=True).stdout.decode()
    assert '#tb 0: 1/10' in crc
    crclines=[line.split(',') for line in crc.splitlines() if not line.startswith('#') and line.strip()]
    assert len(crclines)==n
    assert [int(line[2]) for line in crclines]==list(range(n)) and all(int(line[3])==1 for line in crclines)
    (OUT/'audit'/f'{name}.framecrc.txt').write_text(crc)
    err=np.abs(decoded.astype(np.int16)-enlarged.astype(np.int16))
    quant=np.abs(uint.transpose(0,3,1,2).astype(np.float32)/255-native)
    row=dict(id=name,seed=seed,ordinal=ordinal,metadata=m,native_float32_layout='TCHW',native_float32_sha256=sha(native.tobytes()),native_uint8_layout='THWC',native_uint8_sha256=sha(uint.tobytes()),float_frame_sha256=[sha(f.tobytes()) for f in native],uint8_frame_sha256=[sha(f.tobytes()) for f in uint],sprite_sha256=sha(spritepath.read_bytes()),video_sha256=sha(video.read_bytes()),native_replay_exact=True,png_all_frames_exact=True,decoded_frame_count=len(decoded),fps=info['fps'],duration_seconds=info['duration'],timestamp_grid_exact=True,mp4_decoded_frame_sha256=[sha(f.tobytes()) for f in decoded],mp4_error_uint8=dict(max=int(err.max()),mean=float(err.mean()),first_frame_max=int(err[0].max()),last_frame_max=int(err[-1].max()),per_frame_mean=err.mean(axis=(1,2,3)).tolist()),float_to_uint8_error=dict(max=float(quant.max()),mean=float(quant.mean())))
    records.append(row)
    save(OUT/'audit'/'manifest.json',records) # persist every completed trial
    answer=dict(yes=bool(m['label']),cued=['left','right'][m['target_location']],changed='neither side' if m['changed_patch'] is None else ['left','right'][m['changed_patch']])
    assets.append(dict(id=name,n=n,sprite='data:image/png;base64,'+base64.b64encode(spritepath.read_bytes()).decode(),video='videos/'+name+'.mp4',key=base64.b64encode(json.dumps(answer).encode()).decode()))
# Re-read persisted records for aggregation, rather than counting in memory.
records=json.loads((OUT/'audit'/'manifest.json').read_text())
assert len(records)==10 and len({r['native_float32_sha256'] for r in records})==10
assert {r['metadata']['baseline_transitions'] for r in records}=={12,20,28}
assert collections.Counter(r['metadata']['event_type'] for r in records)==dict(target=6,foil=3,catch=1)
assert collections.Counter(r['metadata']['target_location'] for r in records)=={0:5,1:5}
sourcepaths=['SecondPass/TaskSuite/README.md','SecondPass/TaskSuite/catalog.json','SecondPass/SpatialReadout/SpatialConsolidation/Krauzlis90Fresh/stimuli.py','WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py']
verification=dict(passed=True,trial_count=len(records),unique_trials=len({r['native_float32_sha256'] for r in records}),conditions=sorted({r['metadata']['baseline_transitions'] for r in records}),total_frames=sum(r['decoded_frame_count'] for r in records),fps=10,video_resolution=[600,600],native_resolution=[100,100],native_nominal_fps=100,all_native_replays_exact=all(r['native_replay_exact'] for r in records),all_png_frames_exact=all(r['png_all_frames_exact'] for r in records),all_mp4_timestamps_exact=all(r['timestamp_grid_exact'] for r in records),mp4_max_uint8_error=max(r['mp4_error_uint8']['max'] for r in records),mp4_mean_uint8_error=float(sum(r['mp4_error_uint8']['mean']*r['decoded_frame_count'] for r in records)/sum(r['decoded_frame_count'] for r in records)),source_sha256={p:sha((ROOT/p).read_bytes()) for p in sourcepaths},curation=curation,shuffle_seed=2026100299,ffmpeg_version=subprocess.run([FFMPEG,'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],renderer_only=True,training_updates=0,checkpoint_reads=0)
save(OUT/'verify.json',verification)
(OUT/'data.js').write_text('window.TRIALS='+json.dumps(assets,separators=(',',':'))+';\n')
print(json.dumps({k:v for k,v in verification.items() if k not in ('curation','source_sha256')},indent=2))
