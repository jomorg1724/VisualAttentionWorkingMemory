## Deadly Corridor cloud training active — October 9, 17:22 PDT

User explicitly authorized starting Doom training on RunPod. Separate cloud run01 on RTX 4090 24GB pod jcfmgww5ubnvax (EU-RO-1, $0.89/hour), network volume hb9ruanesc (20GB, about $1.40/month). Parent is the prepared original suite best54808; VDA/suite originals remain untouched. Prepared discrete SAC with 54-action actor/twin32-quantile critics sharing the exact64D belief representation; encoder trains through critics, actionrepeat4, native skill5 rewards, Adam3e-4/clip100 unchanged. Independent CPU reload verified saved2354 decisions/37updates/35episodes, 62critic+4actor+1temperature finite aligned Adam states, 54 encoder tensors advanced, all34 source/checkpoint hashes verified. Worker945/supervisor944/guard779 live; GPU active. One eight-hour allocation from pod creation17:09:53PDT, hard deadline October10 01:09:53PDT; one-million decision ceiling, no automatic extension. Guard requests clean save then pod shutdown using provider-issued pod-scoped credentials; network volume persists. Four cloud contract tests and production-sized CUDA update passed. Initial runner stopped before learning due duplicate steps key in logging; fixed only log naming, archived failed attempt, deadline unchanged. PyTorch2.8/Python3.12 cloud runtime migration documented; WAD exact and config equivalent after CRLF normalization. No held-out performance claim yet. Evidence: SecondPass/DeadlyCorridor/CloudRuntime/run01/independent_production_verified.json; protocol CloudRuntime/README.md. Previous statements that Doom is unstarted are historical.

## VDA explicitly resumed — October 9, 09:16 PDT

User requested continuing training. SecondPass/StabilizedAttentionVDA/LocalRuntime/continuation01 resumes run01/latest1280 with exact model/all86Adam states/scheduler/native streams/allRNG state. Original run01 best/latest1280 and frozen source remain unchanged; best1280 stays validation-eligible. Target remains16000 cumulative VDA updates,14720 remaining at restart; no architecture/loss/lr/clip/batch/data changes. Two CPU resume tests passed, next16 task/condition choices and native draws exact; disposable update identical after reload. CUDA RNG restoration verified. Worker44620/supervisor23228 independently live-verified, saved1296/live1310; all86 finite shape-aligned Adam states advanced correctly (shared+16, each head+1). Receipt continuation01/independent_production_verified.json. Only unused original allocation26607seconds is carried forward from activation09:15:14PDT, hard deadline16:38:41PDT October9; no automatic extension/cloud. Next full validation1600, every320/terminal, final256percondition. New stop request belongs in continuation01. Doom remains prepared and unstarted. Previous pause/live entries below are historical.

## Paused at user request — October 8, 21:59 PDT

VDA continuation run01 stopped cleanly after validation1280. Latest1280 and best1280 preserved with86 finite aligned Adam states each, optimizer/scheduler/native streams/all RNG state saved, zero unsaved updates. Worker44924 and supervisor58400 exited0, GPU lock independently free, original checkpoint/source hashes unchanged. No automatic restart or budget extension; resume only on user request. Doom remains prepared and unstarted. Evidence: SecondPass/StabilizedAttentionVDA/LocalRuntime/run01/pause_verified.json. Earlier running descriptions below are historical.

# VDA continuation and separate Deadly Corridor preparation

October 8, 2026. User requested VDA-4/9/16 in the suite and a separate prepared
Deadly Corridor experiment, then selected continuation from the successful model.

## VDA: production verified

Recovered the corrected native `blank_first_v1` tasks from the local nested
RViT project; exact sources and revision are pinned in
[provenance](../SecondPass/StabilizedAttentionVDA/provenance.json).
VDA means value-directed attention: color denotes reward value, ring completeness
denotes cue validity, and any changed Gabor is reportable, including uncued sites.
Seven inputs include the initial blank and exclude terminal time7.

New experiment: 16 tasks / 44 conditions, including VDA-4/9/16 at theta maxima
8/26/65. Native changes are uniform within +/-theta with per-frame orientation
noise5degrees. Fixed conditions replace adaptive curriculum. Native rasters
are affinely normalized and nearest-resized to100x100, so this is not a pure
set-size manipulation. Terminal value-weighted classification is a disclosed
adaptation of the sequential reward task, not a reaction-time/RL replication.

All80 old tensors and Adam states exactly matched suite best54808 at startup.
Three fresh heads add390 parameters; shared architecture unchanged. New201xxx
streams and16-task scheduling make this a separate transfer experiment.

Supervisor58400/worker44924 launched at21:23:14 PDT. Target16000 additional
updates (512000 presentations;1000 updates/task), eight-hour cap ending05:23:14
PDT October9, no cloud/automatic extension. Batch32/micro4, Adam3e-4, clip100,
CE+.05pixel loss; VDA CE weighted by value/3. Baseline validation, then16,
every320 and terminal. Select equal-task mean AUC; final256 trials/condition.

Saved16 independently CPU-verified: all86 finite shape-aligned Adam states,
all80 old tensors advanced, source hashes match, original checkpoint unchanged.
[Production receipt](../SecondPass/StabilizedAttentionVDA/LocalRuntime/run01/independent_production_verified.json).
Original13-task training and analysis artifacts remain intact.

## Deadly Corridor: prepared, not started or queued

Discrete SAC with twin32-quantile critics and54-action categorical actor, all
reading the same post-GELU64D tensor as the belief heads. Critic gradients
train the shared encoder; actor gradients train its own head. This combines
established methods but is not a published benchmark reproduction or result.

Memory persists within episodes. Pixel replay reconstructs the full real
episode prefix separately under current online/target weights. Action repeat4
skips3 intermediate frames. Native scenario/skill5/reward, headless RGB, no
game-variable features. HUD/weapon/crosshair disabled; full320x240 view resized
to100x100. Scenario hashes recorded. Finite-cap launcher ready, not invoked for
Doom. [Mathematical and operational details](../SecondPass/DeadlyCorridor/README.md).

## Verification

Ten tests passed: native VDA timing/pixels/labels, RNG restoration, model/Adam
migration, finite new-head updates, streaming/full-prefix equivalence, identical
actor/critic input tensors, distributional and terminal targets, replay alignment,
and actual ViZDoom engine/disposable optimizer updates. All44 conditions also
passed finite forward checks. Smoke weights discarded; original source and
checkpoint hashes unchanged. Both folders contain preparation receipts.
Separate illustration seeds produced the
[actual seven-frame VDA previews](../SecondPass/StabilizedAttentionVDA/previews/vda_trials.png).
