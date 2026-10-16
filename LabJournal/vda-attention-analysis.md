# VDA completion and frozen attention experiments — October 9

VDA training completed16000 updates (512000 added presentations) with selected
best16000. Final11264 test trials across44 conditions completed; equal-task
meanAUC .946502. VDA4/9/16 mean BA across theta conditions:.712232/.690985/.694117.
Best/latest CPU reloaded with86 finite aligned Adam states. Training worker
44620 and supervisor23228 exited0; GPU lock free. Receipt:
`SecondPass/StabilizedAttentionVDA/LocalRuntime/continuation01/independent_completion_verified.json`.

User requested the previous full analysis package for the three new tasks,
then specified the central test:100% bottom-right cue, top-left change,
bottom-right inhibition during the change time only. New frozen analysis lives
in `SecondPass/StabilizedAttentionVDA/Analysis/run01`. Protocol and source:
[analysis README](../SecondPass/StabilizedAttentionVDA/Analysis/README.md).

One new finite7200-second local analysis allocation launched16:34:05PDT,
supervisor65640/worker66524, no optimizer/training/cloud/automatic extension.
Prespecified131616 presentations:8064 primary,4608 native,18144 psychometrics,
100800 exploratory interventions. Three VDA sizes, dose/site/timing controls,
mirrored layouts,90 every-timestep attention groups planned. Exact native
pixels, zero-angle and zero-bias controls passed. The100%-cue/uncued-change
combination deliberately violates the native cue; an uncued change remains
a positive VDA response, not a false alarm.

## Primary result — independently paired,384 scenes

At18degrees, bottom-right inhibition−4 during frames5–6 raises top-left
detection48.177%→67.969% (+19.792pp; paired95% interval15.618–24.219pp).
Bottom-right detection falls86.979%→6.510%. Catch false alarms3.646%→1.823%.
Top-left/catch balanced accuracy72.266%→83.073%; dprime1.733→2.530.
The benefit survives Holm correction across8 planned benefit/cost tests
(p=5.657e-18). Bottom-right inhibition−8 gives68.75% top-left detection;
top-left stimulation+4 gives84.635% while lowering bottom-right detection
to39.063%. Opposite-direction controls also show the reciprocal tradeoff.

Every primary scene has identical pixel hashes across arms. Attention through
frame4 is unchanged. At finalframe6, mean attention at bottom-right changes
36.146%→1.596%, and at top-left43.128%→65.760% under bottom-right inhibition−4.
These are model priority-logit interventions on a single frozen checkpoint,
not biological stimulation or training-seed replication.

The psychometric curves show strong evidence-strength dependence and cued-site
benefits. Displayed-validity effects are comparatively modest; calibrated
probability understanding is not established. Color/value effects are
stratified, not separated by a counterfactual color-value remapping.

## Completed analysis and audit

All 131,616 presentations finished in 1,884 seconds. Worker and supervisor
exited normally; the GPU lock is free. Source hashes, checkpoint hashes and
all learned tensors stayed unchanged. Audit verified identical paired pixels,
unchanged pre-intervention attention, the analytic first-frame softmax effect,
5,760 global-shift control presentations and every one of 1,080 exported sprites.

The independent 48-scene mirrored sweep shows opposite-location detection
under cued-site inhibition −4 increasing 43.75%→68.75% for VDA-4,
33.33%→45.83% for VDA-9, and 52.08%→54.17% for VDA-16. Corresponding
opposite/catch balanced-accuracy gains are +13.54 pp (95% CI 9.38–17.71),
+6.25 pp (3.65–9.38), and +2.08 pp (0–5.21). These are exploratory,
scene-clustered pointwise intervals, not training-seed replication.

The gallery contains nine PNG/PDF figures, three seven-frame GIFs, and
90 condition groups with every timestep, individual examples and condition
means. Open [the offline gallery](../SecondPass/StabilizedAttentionVDA/Analysis/run01/gallery/index.html).
All 31 local HTML references resolve and JavaScript syntax checks pass.
Browser interaction testing was blocked by app security policy; no workaround
was attempted. See completion_verified.json and viewer_verification.json.
No further training or analysis is queued; Doom remains prepared only.
