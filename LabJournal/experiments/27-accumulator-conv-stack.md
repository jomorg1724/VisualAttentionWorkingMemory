# 27 — Accumulator states inside the conv stack: gate, curriculum and delay ladder against the plain baseline

[Journal index](../README.md)

Status: completed. Last updated: 2026-09-17T12:10-07:00.

## Question and reason for this test

Experiment 26 found the plain CNN+GRU learns the cued comparison at D0 by curriculum but cannot hold the cued angle across even four blank frames. Its only memory is a 256-d GRU after a flatten. The user asked whether the lineage's temporal-integration component, the conv layers with an accumulator state after each scale, does better. That component's earlier positive results were on two-frame tasks with a frozen encoder and warm starts, so they do not answer this; the state being spatial at 25x25, 13x13 and 7x7 is a specific mechanism for holding a local pattern, which is what the delay failure needs.

## Design and ancestry

No parent checkpoints from the old lineage. Model (`WorkingMemory/PlainBaseline/accum.py`): the plain baseline's four conv blocks; after each of the three coarser blocks a 1x1 projection to 32 channels feeds an accumulator whose 32-channel output is concatenated to the block output for the next block; the deepest [H, O] (160x7x7) is flattened to the same 256-d feature, GRU and heads as the plain model. Three update rules, each an arm: `convgru` (SpatialConvGRU from the lineage), `opponent` (the lineage's opponent accumulator with the fast/slow retentions made learnable per channel, initialised at 0.25/0.75), `kda` (SpatialKDA). Control arm `plain` is the experiment-26 model on the same code path. Every parameter trainable, one learning rate, no clipping, centred input, three stacked frames, Adam 1e-4, batch 64.

Program per (arm, seed), sequential (`program.py`): ring rung from scratch (100k episodes; gate: test BA >= 0.90 or the arm stops); real task D0 from the ring terminal (100k); delay ladder from that model: D0/D1/D2 (100k), D0/D2/D4 (100k), D0/D4/D12/D24 (200k). The last stage's test at the four delays is the arm's delay curve. Seeds 1 and 2. Two lanes in parallel on one GPU: plain+convgru, opponent+kda.

Local smoke test at 0.4% scale passed for all four arms through all five stages. Cloud: one RunPod community pod, 6-hour deadline, bundle and dataset hash-pinned, results pulled every 10 minutes by `cloud/watch.py`, which stops and deletes the pod after verified final retrieval.

## Dated progress

- 2026-09-17T01:04-07:00: pod `hy2m2tjf1awuhf` created (RTX 3090 community, $0.22/h; no 4090 available). Deadline 6 h from creation.
- 2026-09-17T01:15 to 01:35-07:00: two attach attempts failed locally (Windows `ssh-keyscan` returns nothing for this host; Git's works, so `known_hosts` was written by hand; then the launcher crashed on a None stdout after the remote setup had already succeeded). Remote setup verified by ssh: bundle and dataset hashes checked, venv built, both lanes running as pids 1190/1191 with 3 producer workers each, GPU 3.1 GB. Receipts written by hand from that inspection.
- 2026-09-17T01:41-07:00: detached watcher (pid 20760) started; first pull verified.
- 2026-09-17T03:07 and 03:46-07:00: lane 1 and lane 2 finished all 20 stages each (program receipts; 87 and 126 minutes of training). The watcher's last pull was at 02:51; it died when the Claude session ended, so the pod idled until it was finalised by hand.
- 2026-09-17T11:55-07:00: `cloud/finalize.py`: remote processes stopped (GPU 2 MiB), results archive pulled and hash-verified, both lanes `completed`, pod stopped (200), deleted (204), lookup 404. The checkpoint fetch returned zero files (the remote `find` listing was empty; not reproducible now), so the terminal weights were lost with the pod. All metrics, validation looks and test receipts are local. Cost: about 10.8 h at $0.22/h, of which about 8 h were idle.

## Results (test BA on 512 trials, seed 64973001, terminal checkpoint of each stage)

Gate and curriculum: ring rung from scratch 0.998-1.000 for all four arms and both seeds; real task D0 from the ring parent 0.986-1.000 in every case. No accumulator arm loses anything at D0, and none failed the gate.

Delay ladder, first stage (D0/D1/D2, 100k episodes from the D0 model):

| arm | seed | D1 | D2 |
|---|---:|---:|---:|
| plain | 1 | 0.998 | 0.523 |
| plain | 2 | 0.994 | 0.580 |
| convgru | 1 | 1.000 | 1.000 |
| convgru | 2 | 1.000 | 1.000 |
| opponent | 1 | 1.000 | 0.977 |
| opponent | 2 | 1.000 | 0.990 |
| kda | 1 | 1.000 | 1.000 |
| kda | 2 | 1.000 | 1.000 |

Delay curve (final stage, D0/D4/D12/D24, 200k episodes):

| arm | seed | D0 | D4 | D12 | D24 |
|---|---:|---:|---:|---:|---:|
| plain | 1 | 0.998 | 0.867 | 0.867 | 0.848 |
| plain | 2 | 0.992 | 0.875 | 0.883 | 0.875 |
| convgru | 1 | 0.998 | 1.000 | 1.000 | 1.000 |
| convgru | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| opponent | 1 | 0.998 | 1.000 | 0.992 | 0.551 |
| opponent | 2 | 0.996 | 1.000 | 0.990 | 0.996 |
| kda | 1 | 0.994 | 1.000 | 1.000 | 1.000 |
| kda | 2 | 1.000 | 1.000 | 1.000 | 1.000 |

Stage wall time on the RTX 3090 (seconds, seed 1, stages ring/cued/delayA/delayB/delayC): plain 125/122/149/148/489; convgru 148/144/195/219/851; opponent 203/200/280/347/1334; kda 185/181/155/176/716. Full table: `runs/cloud_20260917_010358/final_tables.md`.

Reading. With the delay ladder the plain GRU does learn retention (0.85-0.88 at every delay on both seeds), so the experiment-26 delay failure was a curriculum problem as well. Both gated spatial states, ConvGRU and KDA, are at ceiling (0.994-1.000) at every delay on both seeds, and they reach D2 at 1.000 in the first ladder stage where the plain model sits at 0.52-0.58: the spatial state learns each new delay immediately, the plain model needs the later stages to recover. The opponent traces with learned retention are at ceiling through D12 on both seeds and split at D24 (0.551 and 0.996): retention across 24 blanks is reachable for this arm but not reliably. The candidate mechanism, that the sigmoid-parametrised retention initialised at 0.75 must move to about 0.95 and does so slowly at lr 1e-4, could not be checked because the terminal weights were not retrieved.

## What this shows, what it does not show, and the next decision

Measured: on the orientation family, with everything trainable from scratch and the same recipe, a gated spatial accumulator state inside the conv stack (ConvGRU or KDA) turns the delay ladder into a ceiling result at every delay on two seeds, where the plain CNN+GRU control reaches 0.85-0.88 and the fixed-form opponent traces are seed-dependent at D24. The accumulator arms cost nothing at D0 and pass the from-scratch gate. This is the first component in the project shown to beat a plain baseline under the rules (one factor, same seeds, plain model in the table, no frozen parts, no inherited learning rates).

Not shown: whether the advantage holds on the other four families (motion, binding, recognition, Krauzlis), whether it survives without the delay ladder (no arm was trained on D24 directly), whether it is the spatial placement or the gating that matters (a per-frame GRU with more capacity was not run), or anything about the E/I, attention or priority-readout components of the old lineage. Only two seeds; the opponent split shows seed variance is real. Terminal weights were lost, so no mechanism analysis of the learned states is possible for these runs.

Next decision: adopt the ConvGRU-in-the-conv-stack model as the working baseline (KDA ties it; its state is 256 numbers per site against 32). Then (1) run the same program on motion, binding, recognition and Krauzlis with ladder rungs built per family, plain control beside it; (2) an ablation that isolates gating from spatial placement; (3) re-run one convgru seed locally and keep its checkpoints for probing the state.

## Evidence and closure

- `WorkingMemory/PlainBaseline/runs/cloud_20260917_010358/`: `cloud_provisioning.json`, `launch_receipt.json`, `bundle_manifest.json` (source hashes), `pulled/results/lane{1,2}/` (per-stage `receipt.json`, `metrics.csv`, `validation.json`, `program_receipt.json`), `final_tables.md`, `finalize_receipt.json` (stop 200 / delete 204 / lookup 404), `watch.log`.
- Code: `WorkingMemory/PlainBaseline/accum.py`, `program.py`, `cloud/`.
