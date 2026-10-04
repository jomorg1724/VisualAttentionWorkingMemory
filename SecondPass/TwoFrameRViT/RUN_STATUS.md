# Cancelled online RViT run — 2026-10-03 UTC

User requested replacement by a fresh1000-movie/10-epoch replay run. Graceful stop
at2075 updates /66400 fresh movies. All89 scientific artifacts transferred and
verified; terminal checkpoint CPU-reloaded with92 complete Adam states. Terminal
SHA25690d8e2ac887cc33f04f8d174d9a10af201485910ecd5680ca433fac330cf0f7f.
No final test was run. Latest validation2000: BA50% in all conditions, meanAUC0.497246.
Remote files archived as /workspace/vawm_two_frame_rvit_01/cancelled_online_2075;
local files remain CloudRuntime/artifacts. Same pod retained for explicitly requested
replacement, with original cap unchanged. [New run](../TwoFrameRViTReplay/RUN_STATUS.md).

Earlier status records below are historical.

# Two-frame RViT — training on a separate A40

The user explicitly authorized this model on a separate pod beside the active
sixteen-head KDA and requested faster validation. Pod **7f27p6jxpitihn** is running
fresh whole-model training. Adam1e-4/no clipping, effective32/micro4, FP32/full
BPTT with CNN activation checkpointing. All7,270,290parameters/92tensors train;
no old weights, Adam/RNG/streams or disposable-profile state is inherited.
Native B12/B20/B28 teaching, cues/rendering/labels and event proportions unchanged.

Native profiling prospectively pinned **4216updates /134912episodes**,
including the cost of **18validation looks** and both final tests.
Profile peakGPUallocation **1173951488bytes**. First validation at
**100updates**, then **250,500,750,...,terminal**;100trials/condition with per-cell
BA/AUC/targethit/foilFPR/catchFPR. All these looks participate in validation-only
selection by mean3conditionAUC, thenBA, thenearlier. Repeated validation draws
are not independent samples. Selected/terminal final tests are paired on200fresh
trials/condition. Early scores are not a stop criterion or final performance.

Checkpoint **3/96episodes** downloaded/hashverified/CPUreloaded. All92parameters
changed; all92named Adam states advanced. Direct fresh constructor and empty
initial optimizer/native streams were verified. SHA256
`1362c2beae385992817e3e47b10c7da26d140cc7e935385718beb1f21b6b4bf4`.
[Production evidence](CloudRuntime/production_verified.json) ·
[Independent CPU reload](CloudRuntime/downloaded_checkpoint_verified.json).

A separate NEW8h/$5 cap uses the same bounded envelope as preceding runs:
**2026-10-02 21:16:04UTC → 2026-10-03 05:16:04UTC**, hardstop
**October2 10:16:04PM PDT**, scientific cutoff600seconds earlier.
A40 quote$0.49/hour plus$1storage reserve; no automatic renewal. The parent-owned
frozen deployment permits only exact existingKDApodpa0ko8f2qirisy beside this new
rental; unrelated active/unknown pods block creation. Independent authenticated
shutdown guard and mirror99603/PPID1 are active. Final artifact hashes and both
checkpoint CPUreloads are verified before stopped-pod deletion. Local wake
assertion supports retrieval through this new deadline.

[Measured allocation](CloudRuntime/artifacts/allocation.json) ·
[Immutable cap](CloudRuntime/budget.json) · [Guard](CloudRuntime/guard_verified.json) ·
[Current training](CloudRuntime/artifacts/live_status.json) ·
[Design](README.md) · [Research journal](../../LabJournal/krauzlis-two-frame-rvit.md).

Existing sixteen-head KDA and local CNN-GRU continue. Additional CPU-only early
checkpoint validations run outside their active run directories and leave
optimization, RNG/streams and official selection unchanged. Those are live
validation snapshots, not new model arms or final tests. The cancelledthree-layer
KDA stays deleted/stopped; no old run is restarted or its cap extended.

## First live validation

At update100,100validation trials/condition gave B12/B20/B28 balanced accuracy
50% each; AUC0.330069/0.495308/0.501020, mean0.442132. Decisions were all-positive
(targethit/foilFPR/catchFPR100%). This is an early validation result, not a final
held-out result or reason to stop. Training continues; update160observed21:31UTC.
[Validation100](CloudRuntime/artifacts/validation_000100.json).
