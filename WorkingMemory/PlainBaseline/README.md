# Plain baseline (ladder rung 1 and 2)

The comparison model the 2026-09-16 handoff asks for before any architecture work: a per-frame CNN, one GRU, and per-task linear heads, everything trainable from scratch, one learning rate, no clipping, no fixed priors. It exists to answer one question per task: can a standard model learn this task at all, and how fast. Its numbers are the floor every later component must beat on the same seeds.

## Model

For one frame of shape (3,100,100): four stride-2 convolutions (32, 64, 96, 128 channels; 5x5 then 3x3), each with GroupNorm(8) and ReLU, giving (128,7,7); flatten to 6272; Linear to 256; ReLU. A GRU with hidden 256 reads the 256-vector per frame; its final hidden state feeds the task's Linear head. About 2.4M parameters, of which 1.6M are the flatten-to-256 projection. Position is kept (no global pooling) because the cue and the target are defined by location.

## Recipe

- Adam, lr 1e-3, default epsilon, no weight decay, no gradient clipping (norm logged).
- Batch 64 episodes, processed in two chunks of 32 with gradient accumulation (same gradient, half the peak memory).
- One task per run. Rung 1 cells: D0 for orientation, binding, motion; the three Krauzlis baselines cycled; recognition load 4 with holds 3/4/5 cycled. Rung 2 (`--cells delay`) cycles all delays or loads.
- Train stream seed 61973001, validation 63973001 (256 trials per cell every 200 updates), test 64973001 (512 trials per cell, best-by-validation and terminal checkpoints both reported).
- 100,000 episodes per task (1,563 updates). Sequential on the laptop RTX 3070.

## Options added after the first rung-1 run collapsed

- `--stack K`: early fusion. The CNN sees frames t-K+1..t as 3K input channels (zero-padded before the first frame). With K=1 this is the per-frame model. With K>=2 a change between adjacent frames is a first-order feature of the first convolution, instead of an interaction the GRU gates must discover.
- `--feature-norm layernorm`: LayerNorm on the 256-d frame feature after the ReLU, so the network cannot reduce output variance by scaling the feature towards a constant.
- `--workers N`: N producer processes, each with its own train stream seeded 61973001 + 7919*(worker+1), cycling cells in interleaved order. Removes the generator bottleneck (about 0.6 s per batch single-threaded). `--workers 0` keeps the exact single train stream.

## Files

- `baseline.py`: model, producer thread, training and evaluation, receipt with source hashes.
- `summarize.py`: collects a run directory's receipts into `summary.md`.
- `runs/<rung>_<timestamp>/<task>/`: `receipt.json`, `metrics.csv` (per update), `validation.json`, `best.pt`, `terminal.pt`. `runs/latest_rung1.txt` names the current rung-1 directory.

## Results

See `runs/*/summary.md` and [experiment 26](../../LabJournal/experiments/26-plain-baseline-rung1.md). In one table (test BA on 512 trials, seed 64973001):

| Run | Recipe | Result |
|---|---|---|
| rung1 record runs, five tasks | as specified (lr 1e-3, per-frame) | all at chance; constant-output collapse within 150 updates |
| orientation_cued sweeps (10 recipes) | stack, LayerNorm, centring, zero head, lr 1e-3 to 3e-5 | all at chance in 100k episodes |
| orientation_cued scratch, 1M budget | centred, stack 3, lr 1e-4 | chance through 544k episodes (stopped) |
| ladder: single / ring / sign | centred, stack 3, lr 1e-4 (and 3e-4 zero head) | 1.000 / 1.000 / 1.000 |
| curriculum: real task from ring or sign parent | same | 1.000 / 0.998 within 12.8k episodes |
| delays from the curriculum model | same, cells D0/4/12/24 | D0 0.996, D4 0.498, D12 0.512, D24 0.490 |

Extra tools: `variants.py` (ladder rungs), `trajectory.py` (binned training curves), `gradient_snr.py` (gradient consistency probe; negative result, see the journal page).
