# Krauzlis RViT: repeat 1,000-movie pools for ten epochs

The user requested this sampling change on October 2 PDT / October 3 UTC. The
previous online RViT was stopped at 2,075 updates / 66,400 fresh movies. All 89
scientific artifacts were retrieved and verified, including the terminal model
and 92 Adam states. Latest validation at update 2,000 had 50% balanced accuracy
in each condition and mean AUC 0.497246. These are validation results; cancellation
skipped final tests.

The replacement starts the same whole architecture from scratch. Generate 1,000
native movies, split 334/333/333 across B12/B20/B28, rotating the extra movie among
conditions in successive pools. Shuffle movies within each length and condition
order within each batch round. Every movie appears exactly once in each of ten
epochs, then the entire pool is replaced. Maximum batch 32 / microbatch 4, with
13/14-movie tails retained and normalized by actual batch size. Each epoch is 33
updates; each pool is 330 updates / 10,000 presentations. Count unique movies
separately from repeated presentations. Native 26/28-degree changes, cues, event
proportions and final binary loss remain unchanged. Adam 1e-4, FP32 full BPTT,
no clipping, every learned parameter trainable.

Two focused sampler tests passed: exact tenfold reuse, no premature regeneration,
partial-batch normalization and checkpoint cursor/RNG restoration. The A40 profile
measured pool generation at 79.93 seconds. Measured training and validation costs
pin seven complete pools: 2,310 updates / 70,000 presentations / 7,000 unique movies.
All disposable profile state was discarded before production.

Production checkpoint 3 / 96 presentations was downloaded and CPU reloaded. All
92 parameter tensors changed and all named Adam states advanced; the initial
checkpoint matches the fresh constructor with empty Adam. Live snapshot at
00:21:00 UTC: update 15 / 480 presentations, first epoch of the first pool. Latest
batch loss 0.721914; no validation results yet. This verifies optimization, not
acquisition of the task. [Run evidence](../SecondPass/TwoFrameRViTReplay/RUN_STATUS.md).

Same A40 pod and original eight-hour/$5 allowance; hard deadline 05:16:04 UTC
October 3 / 10:16:04 PM PDT October 2, scientific cutoff ten minutes earlier.
Guard, parent-owned mirror and verified retrieval/deletion remain active.
KDA and local CNN-GRU training continue unchanged.


**Live snapshot, 00:33:31 UTC:** update 182/2,310; first pool, epoch 6/10;
5,544 presentations of 1,000 unique movies. Mean loss over the last100 updates
0.683764; latest batch0.724392. First validation100: BA50% for B12/B20/B28,
mean AUC0.476540, all-positive predictions. This is live validation, not final
held-out testing. Next validation250; no failure or cap renewal.


**2026-10-03 00:41 UTC — All three runs continue.** RViT replay289/2310,
first pool epoch9/10,8800presentations/1000unique; KDA16heads2970/4216,
95040freshmovies; localCNN-GRU2333/2631,74656freshmovies. Recent100-update mean
losses0.681589/0.683380/0.683749 respectively. Latest validations: RViT250meanAUC
0.463484, KDA2108meanAUC0.537502, GRU1315meanAUC0.537774; BA50%allconditions
for every model. These are live validation snapshots, not final held-out tests.
No failure or cap renewal; independent guards/mirrors remain active.


**Final results recorded 2026-10-03T03:12:48.794988+00:00:** 2,310 updates; final 100-update mean training loss 0.49881. Selected fresh mean BA 50.00%, AUC 0.4672. Terminal fresh mean BA 47.83%, AUC 0.4922. [Final report](../SecondPass/TwoFrameRViTReplay/FINAL_REPORT.md).
