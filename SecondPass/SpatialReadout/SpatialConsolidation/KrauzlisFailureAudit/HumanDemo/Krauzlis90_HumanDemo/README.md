# Ten slowed motion trials

Open **index.html** directly in Chrome, Edge, Firefox, or Safari after extracting the ZIP. No server, installation, internet connection, or account is required. Keep index.html, data.js, and videos/ together. Press **Ready — play**, watch the briefly ring-cued side, then answer Yes or No. Answers appear only after your first guess. Replay and Next trial are manual. Do not inspect audit/ or data.js before solving if you want an unspoiled attempt.

The viewer defaults to lossless PNG-frame playback. Choose **MP4 recording** for the corresponding H.264 video, or open any neutral trial01.mp4 through trial10.mp4 in videos/. No audio, pre-roll, answer overlays, interpolated frames, or automatic repeats were added. The task's final fixation-only report frame remains on screen while you decide. A desktop displays the original 100×100 scene at 600×600 using nearest-neighbor enlargement; a small screen uses 300×300. The MP4 files are 600×600.

## What to look for

A thin white circular ring appears on the left or right for the first two frames, around the future dot patch location. A small dark central plus/cross remains as the fixation mark. The ring then disappears for five fixation-only frames. Two pale dot groups subsequently move; judge the mean direction of the group at the cued location. Yes means that group changed direction. No means only the other group changed, or neither changed. Changes are ±90°. The central cross is not an arrow, and this renderer does not draw corner instruction glyphs. Individual dot turnover is expected and is not itself the event.

## Timing and fidelity

This is the existing `Krauzlis90Fresh.stimuli.SpatialBatteryStream`, sampled directly with new deterministic demo seeds. These are ten distinct trials, not ten new task definitions. All three baseline conditions are represented. The native frame sequence is preserved: two cue frames, five fixation-only frames, one dot-reference frame, the original 12/20/28 baseline transitions, eight postevent transitions, and one report frame: 29/37/45 frames total. Playback is exactly 10 encoded MP4 frames per second, ten times slower than the renderer's nominal 100 Hz clock. The clips last 2.9/3.7/4.5 seconds. Native time intervals are not selectively changed.

The default PNG playback preserves every pixel after deterministic `round(native_float * 255)` conversion. Original float32 TCHW arrays are included in native/ as NPZ, and lossless uint8 sprite sheets in sprites/. Pixel-preserving browser rendering is not a color-calibrated display. Browser scheduling can jitter; the authoritative video timestamp checks are in verify.json and audit/*.framecrc.txt. PNG playback visits every frame in order without skipping, interpolating, or adding frames. Keep the tab visible: switching away interrupts the attempt and requests a full replay.

H.264 MP4 is **lossy**. Every decoded frame was compared against the exact 6× nearest-neighbor uint8 image. Aggregate maximum error was 7/255 and mean absolute error about 1.998/255; per-frame errors and decoded hashes are in audit/manifest.json. The PNG sprites are the authoritative exact uint8 pixels. No stimulus geometry, contrast, dot trajectories, cue timing within the sequence, or labels were edited to make trials easier.

This is an informal viewing/self-test, **not formal psychophysics**, reaction-time measurement, or evidence about trained-model ability. Slow playback changes apparent speed and cue visibility. Trials were bounded-curated using only event-category/side metadata to cover the task, not selected for ease, and shuffled. This small demonstration does not estimate population accuracy. No training, model, checkpoint, cloud, or GPU workload was used.

## Local answers and privacy

First-answer score lives in page memory only. Reload clears it. There is no telemetry or network code. Correct answer, cued side, and changed side are revealed only after guessing. Offline assets necessarily contain the key; its Base64 encoding prevents accidental front-page exposure, not deliberate inspection. The audit manifest also contains full native metadata and should be viewed only after solving.

## Reproduction and verification

The renderer-only exporter is `../generate.py` in the repository's isolated HumanDemo directory. From the repository root, run:

```
PYTHONDONTWRITEBYTECODE=1 /Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/HumanDemo/generate.py
```

It requires NumPy, PyTorch (CPU only), Pillow, and imageio-ffmpeg 0.6.0; the latter was installed in HumanDemo/.deps without modifying the shared runtime. FFmpeg 7.1 encoded H.264 CRF 10, yuv420p, 10 fps, sixfold nearest-neighbor upscaling. The exporter limits numerical/encoder threads to one. Source hashes, new seed namespaces, chosen ordinals, and shuffle seed are recorded. Rerendering each selected trial from a fresh stream seed was bit-identical. A reproduction copy of the exporter is under audit/; it must be run from its original repository location (or adjusted explicitly for that root), not as an offline viewer dependency.

verify.json records aggregate renderer/codec assertions. audit/manifest.json contains every native float frame hash, uint8 frame hash, decoded MP4 frame hash, original metadata, PNG/video file hashes, conversion error, and all-frame compression error. The exporter persists records after each trial and re-reads them to assert ten unique trials and condition coverage. Browser QA evidence is in browser_qa.json and audit/ screenshots. The portable ZIP is separately extracted and opened via file:// in a real browser with networking disabled before delivery.
