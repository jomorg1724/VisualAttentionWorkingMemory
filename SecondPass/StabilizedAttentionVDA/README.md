## VDA explicitly resumed — October 9, 09:16 PDT

User requested continuing training. SecondPass/StabilizedAttentionVDA/LocalRuntime/continuation01 resumes run01/latest1280 with exact model/all86Adam states/scheduler/native streams/allRNG state. Original run01 best/latest1280 and frozen source remain unchanged; best1280 stays validation-eligible. Target remains16000 cumulative VDA updates,14720 remaining at restart; no architecture/loss/lr/clip/batch/data changes. Two CPU resume tests passed, next16 task/condition choices and native draws exact; disposable update identical after reload. CUDA RNG restoration verified. Worker44620/supervisor23228 independently live-verified, saved1296/live1310; all86 finite shape-aligned Adam states advanced correctly (shared+16, each head+1). Receipt continuation01/independent_production_verified.json. Only unused original allocation26607seconds is carried forward from activation09:15:14PDT, hard deadline16:38:41PDT October9; no automatic extension/cloud. Next full validation1600, every320/terminal, final256percondition. New stop request belongs in continuation01. Doom remains prepared and unstarted. Previous pause/live entries below are historical.

## Paused at user request — October 8, 21:59 PDT

VDA continuation run01 stopped cleanly after validation1280. Latest1280 and best1280 preserved with86 finite aligned Adam states each, optimizer/scheduler/native streams/all RNG state saved, zero unsaved updates. Worker44924 and supervisor58400 exited0, GPU lock independently free, original checkpoint/source hashes unchanged. No automatic restart or budget extension; resume only on user request. Doom remains prepared and unstarted. Evidence: SecondPass/StabilizedAttentionVDA/LocalRuntime/run01/pause_verified.json. Earlier running descriptions below are historical.

# VDA-4, VDA-9 and VDA-16 suite extension

Continuation launched October 8, 2026 at 21:23 PDT from the successful model.
Saved update16 was independently verified with86 finite aligned Adam states.
See `LocalRuntime/run01/independent_production_verified.json`. The completed
13-task suite, its selected checkpoint 54,808, and its analysis remain intact.

## What was recovered

The source is the local RViT checkout:
`C:/Users/jomor/OneDrive/Desktop/RViT_plus_paper_jepa_grid9-20260718T193411Z-1-001/RViT/KimiDecayVDAConvRecurrentTF/envs`.
`provenance.json` pins the checkout revision and exact copied source hashes.
The task is **value-directed attention**, not delayed visual matching alone.

| Variant | Native array / raster | Cue positions | Displayed validity |
|---|---|---|---|
| VDA-4 | 2×2 / 50×50 | All four | .25, .50, .75, 1 |
| VDA-9 | 3×3 / 75×75 | Top-left or bottom-right | 0, .25, .50, .75, 1 |
| VDA-16 | 4×4 / 100×100 | All sixteen | .25, .50, .75, 1 |

Half the native trials have a change. Conditional on change, the cued location
changes with exactly the displayed probability; otherwise a non-cued location
is sampled uniformly. A 0-validity cue is an exclusion cue, not an uninformative
cue. Any actual change is reportable, including an uncued change. This differs
from the Krauzlis task's ignore-the-foil rule.

Cue red/green/blue denotes correct-response value 5/3/1. Ring completeness denotes
validity. All active Gabors have fresh 5-degree orientation noise each frame and
native pixel noise. Change size is uniform on [-theta, theta]. We expose fixed
theta maxima 8, 26 and 65 degrees as three conditions per task, rather than use
an unrecorded adaptive curriculum. This makes **16 tasks / 44 conditions**.

## Preserved timing and disclosed adaptations

The corrected `blank_first_v1` collector returns seven observations:
blank, cue, blank, sample, sample, test, test (physical times 0–6).
It includes `reset()` and excludes the terminal returned image at time 7.
The original discarded-reset supervised results are not a valid reference.

Native rasters are rendered unchanged, then mapped by x -> (x + 1.11)/2.22 and
nearest-neighbor resized to 100×100 for this backbone. The affine map keeps the
native noise extremes without clipping and puts blank pixels at .5. VDA-9's
noninteger resize changes its spatial sampling; VDA-4 is doubled; VDA-16 is not
resized. These are explicitly versioned input adaptations, not byte-identical
100×100 native tasks or a pure set-size/capacity comparison.

The suite uses a **terminal supervised decision**, so it does not reproduce the
original sequential wait/declare RL policy or reaction times. Its VDA loss is

    L = mean[(value / 3) * CE(change_anywhere, prediction)] + .05 L_pixel

The value division is the native mean value, a fixed scale rather than a
batch-dependent normalization. This preserves 5:3:1 optimization weights and
gives the color cue a consequence. It is a utility-weighted classification
surrogate, not an assertion of identical learning to the original reward rule.
Report accuracy/AUC as well as the fraction of available reward earned, and
stratify by value, validity, realized location and magnitude.

## Model and proposed run

The shared encoder, spatial/CLS memory, attention, and 64D readout are unchanged.
Three new binary heads add 390 parameters: 381,654 parameters / 86 tensors.
Default initialization copies all old weights and their 80 Adam states from
the successful checkpoint. New heads initialize independently and obtain Adam
states on first use. `--fresh` selects a whole-model fresh start instead.
The new 16-task scheduler and independent 201xxx streams are a new experiment;
this is not an exact continuation of the old 13-task data schedule.

Each task gets one batch per 16-update cycle; conditions rotate within task.
Batch 32 / microbatch 4, Adam 3e-4, clip 100, FP32. Validate before training,
after the first complete 16-task cycle, then every 320 updates,
select by equal-task mean AUC, and test selected weights on separate streams.
No original task, artifact or source file is changed by the extension.

Launch command used from the repository root (do not launch a duplicate):

```powershell
C:/Python310/python.exe -m SecondPass.DeadlyCorridor.launch vda --run SecondPass/StabilizedAttentionVDA/LocalRuntime/run01 --steps 16000 --wall-seconds 28800
```

The launcher owns a finite local cap and does not restart automatically. The
target is an upper bound, not a measured promise that all 16,000 updates fit. A cap
exit is incomplete training. Put `stop_requested.json` in the run directory for
a checkpointed stop between updates. The old suite's GPU lock is shared.

## Verification

`test_contract.py` checks exact native-renderer observation alignment, invalid
location labels, isolated/restorable random streams, 16-task scheduling, source
hashes, exact old-head outputs after migration, old Adam preservation, finite
new-head updates and unchanged original checkpoints. Preparation smoke weights
are discarded. `preparation_verified.json` records the checks.
