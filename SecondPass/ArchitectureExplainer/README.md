# Attention, memory and decision: animated architecture walkthrough

Source for a narrated 3-minute, 12-second explanation of the spatial-memory-query architecture. Nine chapters cover visual evidence, spatial memory, CLS memory, the learned CLS constant, queries/keys/values, shared attention, temporal comparison, recurrent readout, and training objectives.

The animation is a teaching schematic. Animated activations and belief bars are illustrative. Connections, dimensions and losses were checked against the files recorded in `source_manifest.json`. Rendering does not load weights, run model inference, or start training.

## Build locally on Windows

Requirements: Python with Pillow and NumPy, FFmpeg and FFprobe on PATH, Windows Segoe UI fonts, and the Microsoft Zira Desktop speech voice available through Windows PowerShell's System.Speech assembly.

From the repository root:

```powershell
powershell.exe -NoProfile -File SecondPass/ArchitectureExplainer/narrate.ps1
python SecondPass/ArchitectureExplainer/render.py
python SecondPass/ArchitectureExplainer/verify_package.py
```

Open the generated `index.html` for video playback and chapter navigation. `architecture-explainer.mp4` includes synchronized on-screen captions and synthesized narration; `captions.srt` supplies a separate subtitle file. The verification script fully decodes the video, checks codecs, timing, narration presence, caption fit and chapter structure, and creates a portable ZIP containing the player, video and source.

The published player, MP4, narration WAV, poster, subtitles, chapter metadata and verification receipts are included in Git. Intermediate frames, per-sentence audio and ZIP packages remain ignored. The original export was 1280 × 720 at 15 fps and passed a complete media decode, caption-fit checks, and visual inspection of all nine chapter stills. The verification script checks media properties; its historical visual-review statement is not a substitute for inspecting any revised render.

## The memory distinction

- Spatial recurrent memory: **100 vectors of 64 numbers**, arranged as a 10 × 10 spatial grid (6,400 numbers).
- CLS recurrent memory: **one evolving 64-number state**, updated from attention-pooled values.
- Learned CLS token: **64 trainable parameters**, fixed during each trial.
- Temporal-comparison memory: **735 numbers**, updated by `0.8 * old + 0.2 * selected_comparison`.
- Beliefs: task output probabilities computed from the readout; distinct from the memory states.

Every frame updates the spatial grid and the CLS state in parallel. Spatial memory receives the full value grid through a spatial GRU, with attention-modulated write gates and 3 × 3 neighborhood computations. CLS receives an attention-weighted sum of the values through a separate GRU. The new architecture also projects the previous 6,400-number spatial state directly into its single 64-number query. The earlier architecture supplies the query from CLS memory plus the learned token.

The readout combines CLS and comparison memory. Additional temporal state includes a motion-population trace and previous flow; lagged pixel frames provide motion evidence.

## Editing

Edit `script.json` for narration, `render.py` for diagrams, and `player-template.html` for the chapter player. Regenerate narration before rendering after any script change. Preserve the distinction between the current architecture and the earlier checkpoint used in the neuroscience analyses.
