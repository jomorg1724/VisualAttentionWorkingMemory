# James Herman: RViT weight-edit "existence proof" for where value acts (Nov 4–6, 2026)

**This is James Herman's work, not ours.** It was done by James Paul Herman (hermanj@gmail.com), shared with jomorg1724@gmail.com through Google Drive, and copied here unchanged on 2026-11-07. We did not run, edit or verify any of it.

## The experiment in one paragraph

In RViT trained on the Luo & Maunsell reward-value detection task (SENS-C table), reward value ends up changing the readout (the actor), not the per-location memory code. James asked whether the architecture forces that, or whether learning chooses it.
- **The code can carry value.** He hand-edited the LSTM's output-gate weights with one rank-1, position-gated edit at the low-value location (Part C). That produces a Luo & Maunsell-like value effect as a gain on that location's code: d′ 2.70 → 2.16 there, the other location unchanged, information intact.
- **Training doesn't put value there.** With the actor frozen, 50k episodes give no sensitivity value effect (Part A).
- **Training removes it when handed it.** Starting from edited models, the standard learner undoes about 60% of the edit, at the same rate whether the edit helps or hurts reward (Part D).
- **The edit is disarmed, not undone (Part E).** The token projection re-maps the position signal that triggers the edit. Either the critic's or the actor's loss alone is enough to do this.

His conclusion is that the readout locus is a property of the learning rule, not a limit of the architecture. See `SUMMARY_2026-11-rvit-existence-proof.md` for his own summary, numbers and caveats.

## What's here

| Path | What it is | Source on James's Drive |
|---|---|---|
| `SUMMARY_2026-11-rvit-existence-proof.md` | His one-page wiki note: answer, design, results, caveats | `RViT/documents/wiki/experiments/2026-11-rvit-existence-proof.md` |
| `analysis/` | His analysis folder as-is: `README.md`, `stats/`, `figs/` (Parts A, C, D, E, locus), `data/runs/` | `RViT/analyses/2026-11-04_rvit-existence-proof/` |
| `plans/` | `RVIT_EXISTENCE_PROOF_PLAN.md` (rev 3) and its independent review | `RViT/documents/plans/` |
| `code/zk/` | Experiment scripts: `ep_train`, `ep_edit`, `ep_make_edit`, `ep_stats`, `ep_geom`, `ep_hybrid`, `ep_figs`, `ep_e0`, `tests/test_ep_train.py` | `code/tree_zk/zk/` |
| `code/slurm/` | Cluster launchers: `run_ep.sh`, `ep_cells.slurm`, `ep_edit.slurm` | `code/slurm/` |

`MANIFEST.sha256` lists a SHA-256 hash for every copied file: 4,050 files, about 400 MB. Nearly all of the size is in `analysis/data/runs/`, which holds 3,063 JSON result files, 640 CSVs and 320 per-cell training logs in about 340 run folders. There are no model checkpoints in it.

## What's not here

- **The trained checkpoints.** They're on his CRC cluster (`/ix/jpherman/hermanj/rvit_zk/runs/existence/`), not in Drive.
- **The rest of his `tree_zk` codebase.** That includes `agent_orig`, `train_orig.py`, `lm_assay`, `lm_readout_probe` and the task code. The scripts here import from it, so they won't run on their own.
- **His other analyses.** Only this experiment was copied.

## How it was copied

Drive shortcuts to James's shared folders were added to jomorg1724's My Drive, and Google Drive for desktop synced them to `G:\`. Files were copied from there, skipping Drive's `desktop.ini` files. The three slurm scripts came byte-exact from the Drive API instead, and their sizes match Drive. You can remove the shortcuts from My Drive at any time; that doesn't affect James's originals.
