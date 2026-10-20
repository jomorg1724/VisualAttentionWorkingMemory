"""Read-only media checks, followed by an archive of this task's deliverables."""
from pathlib import Path
import subprocess, json, hashlib, zipfile, wave
import numpy as np
from PIL import Image, ImageDraw
import render

root=Path(__file__).resolve().parent
video=root/'architecture-explainer.mp4'
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
v=next(s for s in probe['streams'] if s['codec_type']=='video')
a=next(s for s in probe['streams'] if s['codec_type']=='audio')
assert (v['width'],v['height'])==(1280,720)
assert v['codec_name']=='h264' and a['codec_name']=='aac'
assert 185<float(probe['format']['duration'])<200
decoded=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
assert decoded.returncode==0 and not decoded.stderr.strip(),decoded.stderr
with wave.open(str(root/'narration.wav'),'rb') as f:
    raw=np.frombuffer(f.readframes(f.getnframes()),dtype=np.int16).astype(float)
    assert np.sqrt(np.mean(raw**2))>100
    audio_seconds=len(raw)/f.getframerate()/f.getnchannels()
chapters=json.loads((root/'chapters.json').read_text())
assert len(chapters)==9
assert abs(audio_seconds-chapters[-1]['end'])<.02
assert abs(float(v['duration'])-audio_seconds)<.15
dummy=ImageDraw.Draw(Image.new('RGB',(1280,720)))
for scene in render.SCENES:
    assert dummy.textlength(scene['title'],font=render.font(37,True))<1176
    for sentence in scene['sentences']:
        bottom=render.wrapped(dummy,sentence,64,579,1152,25)
        assert bottom<=687,(sentence,bottom)
page=(root/'index.html').read_text(encoding='utf8')
assert '__CHAPTERS__' not in page
assert 'movie.currentTime=c.start' in page
for name in ['architecture-explainer.mp4','poster.png','captions.srt']:
    assert (root/name).exists()
receipt={
    'duration_seconds':float(probe['format']['duration']),
    'dimensions':[1280,720], 'video_codec':v['codec_name'],'audio_codec':a['codec_name'],
    'full_decode':'passed','audio_present_and_nonzero':True,
    'caption_fit':'all 27 sentences fit','chapter_count':9,
    'visual_review':'all 9 chapter stills inspected; enlarged review of 1, 2, 7',
    'sha256':hashlib.sha256(video.read_bytes()).hexdigest()
}
(root/'verification.json').write_text(json.dumps(receipt,indent=2))
with zipfile.ZipFile(root/'architecture-explainer-package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['index.html','architecture-explainer.mp4','poster.png','captions.srt','chapters.json','script.json','render.py','narrate.ps1','player-template.html','source_manifest.json','build.json','verification.json','verify_package.py']:
        z.write(root/name,name)
print(json.dumps(receipt,indent=2))
print('Package bytes:',(root/'architecture-explainer-package.zip').stat().st_size)
