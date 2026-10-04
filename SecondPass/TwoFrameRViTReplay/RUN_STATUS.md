# Completed — older cloud RViT did not acquire the task

All **2,310 updates / 70,000 presentations / 7,000 unique movies** completed.
Final 100-update mean training loss **0.49881**. Validation selected update 1,250.
Fresh tests: selected checkpoint **50% balanced accuracy in every condition**,
mean AUC **0.4672**; terminal checkpoint mean BA **47.83%**, mean AUC **0.4922**.
Terminal BA: B12 **47.11%**, B20 **50.60%**, B28 **45.79%** (200 trials each).
The decline in training loss did not demonstrate generalization.

All 63 final manifest artifacts were verified, selected/terminal checkpoints
CPU reloaded with 92 active Adam states, and pod `7f27p6jxpitihn` was stopped and
deleted after retrieval. No continuation or cap renewal was launched.

[Final report](FINAL_REPORT.md) · [Retrieval](CloudRuntime/retrieval_verified.json)
· [Deletion](CloudRuntime/cleanup_verified.json).

Earlier entries below are historical.

# Fresh RViT replay training — RUNNING

**Live snapshot, 00:33:31 UTC:** update 182/2,310; first pool, epoch 6/10;
5,544 presentations of 1,000 unique movies. Mean loss over the last100 updates
0.683764; latest batch0.724392. First validation100: BA50% for B12/B20/B28,
mean AUC0.476540, all-positive predictions. This is live validation, not final
held-out testing. Next validation250; no failure or cap renewal.


Confirmed on the existing A40 pod `7f27p6jxpitihn`. Fresh model, Adam, RNG,
streams and counters; all 7,270,290 parameters train. No profile or predecessor
weights transferred. The architecture, native stimuli and final binary loss
are unchanged.

Generate 1,000 movies, train for ten shuffled epochs, then generate the next pool.
Effective batch 32 / microbatch 4, including correctly normalized partial tails.
Each epoch has 33 updates; each pool has 330 updates / 10,000 presentations.
Profile pins **seven complete pools: 2,310 updates, 70,000 presentations and
7,000 unique movies** within the original allowance.

Production checkpoint 3 / 96 presentations was downloaded and CPU reloaded:
all 92 named Adam states advanced, all 92 parameter tensors changed, and the
initial checkpoint matches the fresh constructor with empty Adam state.
Checkpoint SHA256: `a280654e77c8c6ba0c74fee32c9406163a64f65123dc0b19901b20b156b01eb0`.
[Startup evidence](CloudRuntime/production_verified.json).

Live snapshot: 2026-10-03T00:21:00.194982+00:00, update 15, 480 presentations,
first pool / first epoch. Latest batch loss 0.721914.
No validation results yet. Independent fixed validation 100/cell at update 100,
every 250 updates and terminal; fresh final tests 200/cell.

Original hard deadline remains **05:16:04 UTC October 3 / 10:16:04 PM PDT October 2**;
scientific cutoff 05:06:04 UTC. No cap renewal. Independent guard, detached mirror
83662 and bounded wake 83663 remain active for retrieval and verified deletion.

The cancelled online run stopped at 2,075 updates / 66,400 fresh movies. Its 89
artifacts and terminal checkpoint are preserved in the original runtime; no final
held-out test was run for that cancelled model. KDA and local CNN-GRU continue.

[Journal](../../LabJournal/krauzlis-rvit-replay.md) · [Implementation](worker.py).
