# Faster-dot weighted-image transformer

## Final result: completed October 5, 4:36 PM PDT

All25,024updates completed, with391,000unique movies and782,000 retained-trajectory
presentations. Recovery took28min20sec; the overnight shutdown is excluded from
that duration. The original200uncheckpointed updates remain recorded as rollback.
MinimumvalidationBCE selected10000; latest25024 retained. Final3072presentations/
1536pairedcontexts: **BA49.772%, AUC0.498358, BCE0.693149**. Allsixspeed/anglecells
remain near chance (BA49.02–50.98%). Increasing speeds fourfold did not resolve
change detection in this architecture/training setup.

Root CPU-verified best10000/latest25024: all24Adamstates aligned and finite,
allweightsfinite. Recovery worker15684, supervisor33412 andguard20572 exited;
guardreceipt normal_stop. No training is active. See
`LocalRuntime/run02_speed4/final_test.json`, `completion_verified.json` and
`root_final_verification.json`. Launch/recovery entries below are historical.

## October 5: explicit shutdown recovery

The user reports the computer was shut down and explicitly requests continued
training. No original worker remains live. Latest/best checkpoint10000 is intact:
all24 finite Adam states align to10000, with finite weights and CUDA/global/shuffle
RNGs saved. Saved sampler is pool156/start156000, epoch0, cursor512 with its exact
1000-item permutation. Last live10200 contains200 uncheckpointed updates; recovery
rolls those back and records them separately rather than adding fresh exposure.

Original four-hour wall deadline expired while powered off. Explicit user recovery
gets one finite3600-second local window, including final evaluation/reporting,
without changing the cumulative25024 target. Preserve original budget/config/source
and previous checkpoint selection. Runtime/source migration is recorded separately
in `LocalRuntime/run02_speed4/Recovery/resume01`. Preparation is not proof that a
worker is running; saved progress after restoration will provide that evidence.

**Actual recovery launched4:07:42PM PDT October5:** supervisor33412, worker15684,
independentguard20572, finite deadline5:07:42PM. Resume receipt proves exact
CPU/CUDA/global/shuffle RNG restoration and originalepoch/order/cursor; first
resumed10001 saved withall24 Adamstates aligned. Root subsequently independently
CPU-loaded latest10250 withfiniteweights/Adam, liveworkeridentity, original
scientificconfig/budget unchanged and current source matching recoverymanifest.
Focusedactualproduction CPU tests atmid-epoch/endpool yieldedbitwiseidentical
model/Adam/sampler/RNG to uninterrupted execution. This is runtime correctness
evidence, not a learning-success claim. Final25024target and futurevalidation/
test selection unchanged. Source migrationversion `shutdown_exact_state_resume_v1`.

October 4, 2026: after the failed original transformer and its read-only audit,
the user explicitly requested increased dot speed and a rerun. This is one speed
intervention, not an attention modification or a new architecture comparison.

Speeds are multiplied by four, from 0.375/1/2 to **1.5/4/8 pixels per frame**.
The 26/28-degree turns, same six-frame continuous movie, Gaussian renderer,
labels, nuisance draws and evaluation pairing remain unchanged. At the slowest
speed, changed versus unchanged last-frame dot positions differ by approximately
1.35/1.45 pixels instead of 0.337/0.363. At the fastest speed, those differences
are approximately 7.20/7.74 pixels. Increased visibility is a rationale, not a
promised solution.

The exact same model trains from scratch using the same initialization and
shuffle seeds: chronological `(oldest+2*middle+3*newest)/6`, 10x10 patches,
width128, one four-head transformer layer, CLS vectors concatenated into the
existing nonlinear classifier. Same BCE, Adam3e-4, FP32, batch32/micro8 and
1000-pair/two-epoch fresh pools. No inherited checkpoint or optimizer state.

The requested 25,024 updates represent 391,000 unique movies and 782,000 pair
presentations. The same prospective profile pins feasible exposure before
production. One new finite four-hour local cap begins at profiling, covers
evaluation/reporting and cannot extend automatically. Validation selection and
final paired held-out evaluation are unchanged. Previous run/checkpoints/source
snapshots remain preserved.

Focused CPU checks reproduce the original frames exactly at default scale1,
confirm both velocities scale4 across train/validation/test, and confirm matched
nuisance variables, labels, angles and paired evaluation. Independent review of
the actual delta confirms architecture and all other training settings unchanged.

Runtime: [run02_speed4](../SecondPass/WeightedSumTransformer/LocalRuntime/run02_speed4).
The native profile pins the full25,024 updates with measured update0.04378sec,
pool generation1.92031sec and evaluation0.003455sec/pair. Supervisor35380
launched October4,10:06:35PM PDT; profile starts10:06:37.834PM, hard cap
October5,2:06:37.834AM PDT. Actual production progress and outcomes require
separate saved receipts; no final score exists at launch.

**Actual production verified:** worker41532 progressed beyond256updates.
Root independently CPU-loaded best64/latest250: all24 Adam states align to their
saved steps, all weights/optimizer values finite, speed_scale4 persisted and
frozen source/profile match. Owned worker identity is live. These establish
execution, not learning success. Next planned validation5,000.

[Original experiment and audit](weighted-sum-transformer.md) · [Current status](CURRENT_STATUS.md)
