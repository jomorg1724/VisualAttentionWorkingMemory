# Structure discovery on native Krauzlis (October 24)

> **October 25 follow-up:** [the same pipeline on spatial binding, cued orientation and memory-guided search](tasks/README.md). Binding generalizes (100%), cued orientation is partial (about 71% of a 75% ceiling without the cue sign) and memory search fails. The scripts now take a task directory and field; Krauzlis behavior is unchanged.

**Question.** The adversarial review of the "beyond gradients" proposals found that every large sample-efficiency gain came from built-in structure, not from avoiding gradient descent. So: can that structure be *discovered automatically* and handed to an ordinary gradient-trained network, end to end on native pixels?

**Answer on this task: yes.** An ordinary GRU trained with Adam goes from chance (50–54% balanced accuracy) to **94–96% from 200–3,000 labeled movies** once it receives two pieces of structure that were found by the pipeline, not specified by hand:

1. **Nuisance symmetry removal**, found without labels: flow channels that remain phase-coherent share one unknown rotation, so motion is re-expressed relative to its own causal history.
2. **An attention gate**, found by closed-form probes over 58 candidate appearance maps: the winner is the cue-onset frame, exactly the known answer.

Each piece alone does little (canonicalization only: 58–60%; gate only: 53–54%). Together they reproduce the hand-built upper bound to every digit. The repository's best engineered network reached 85.7% after 38,400 presentations; more than 15 fresh architectures stayed at chance after up to ~147k movies.

CPU only, no GPU; about 7 minutes of rendering and about 75 minutes of training in total.

## Pipeline

| Stage | Script | Uses labels? | What it does |
|---|---|---|---|
| 0 Fields | `features.py` | no | Renders movies with the repo renderer; per 9×9 grid cell: local optical flow `z = u + i v` (multi-lag Lucas–Kanade) and appearance energy. Left-padded so movies end together. |
| 1 Nuisance | `discover.py` | no | Coherence `|E z_i z̄_j|` between flow channels; degree-1 moments `E z` ≈ 0 show a random rotation; coherent channels form one sector. Canonical coordinates `w = z·conj(R)/|R|`, with `R` the causal running sum of the sector's past. |
| 2 Oracle | `discover.py` | train split | Label correlation of feature families and the samples needed to detect each at 5σ (25/ρ²). |
| 3 Gate | `discover.py` | train split (400 trials) | Every appearance map at frames anchored to movie onset (0–28) or to the report (last 29) is a candidate gate. Score = 5-fold CV balanced accuracy of a shrinkage LDA on gate-weighted pooled canonical motion. |
| 4 Networks | `train_arms.py` | yes | GRU (one shared per cell, or one flat GRU over all cells), Adam 3e-3, batch 32, 24,000 sample passes, threshold from training data; 1,500 held-out movies, 2 seeds. |

## Results

**Nuisance discovery** (`discovery.json`; 3,000 training movies, labels unused):
- Degree-1 moment |E z|/rms = 0.052: flow phase is uniformly random across trials, so a rotation nuisance exists.
- Within-cell coherence by lag 1/2/4/8/12 frames: 0.92 / 0.88 / 0.86 / 0.81 / 0.77.
- Across-cell coherence among the 12 most active cells: mean 0.88, so **one global sector**. The search recovered a renderer fact: the two patches' mean directions are always 90° apart, so a single rotation is shared.
- Canonical coordinates are invariant to an arbitrary per-cell rotation (max difference 0.0097, float32 rounding).

**Plateau oracle** (largest |correlation| with the label in each family; samples to detect at 5σ):

| Feature family | Correlation | Samples |
|---|---|---|
| Raw flow, best single channel | 0.036 | ~18,800 |
| Raw flow pooled over cells | 0.022 | ~51,800 |
| Raw degree-2 rotation-free products (change signal) | 0.085 | ~3,500 |
| Canonical motion, pooled, no gate | 0.072 | ~4,800 |
| **Canonical motion, gated** (best gate = cue onset frame) | **0.297** | **~280** |

**Gate discovery.** The two top candidates are the cue frames (onset+0 and onset+1, CV balanced accuracy 0.566) out of 58; the ungated baseline scores 0.436. The mean gate map puts its mass on the two patch locations. The probe's absolute accuracy is low because it is linear, but its ranking is correct.

**Networks** (held-out balanced accuracy, mean of 2 seeds, range in brackets; AUC):

| Arm | 200 labels | 1,000 labels | 3,000 labels |
|---|---|---|---|
| Flat GRU on raw input (repo default style) | 50.7% [50.5–50.9], 0.515 | 53.9% [53.4–54.4], 0.560 | 51.5% [50.6–52.3], 0.511 |
| Shared per-cell GRU, raw | 50.0%, 0.498 | 50.5%, 0.501 | 51.8%, 0.518 |
| + discovered gate only | 52.8% [49.9–55.7], 0.539 | 53.7%, 0.549 | 52.7%, 0.522 |
| + discovered canonicalization only | 58.0% [55.1–60.8], 0.612 | 57.6%, 0.611 | 59.6% [55.9–63.3], 0.614 |
| **+ both (fully discovered)** | **94.2% [93.6–94.8], 0.986** | **95.8% [95.4–96.1], 0.990** | **95.2% [94.9–95.5], 0.990** |
| + both, literal global-sector canonicalization | 94.0% [93.0–95.0], 0.982 | 95.1% [94.9–95.3], 0.988 | — |
| + canonicalization and **hand** gate (upper bound) | 94.2% | 95.8% | 95.2% |

The discovered gate is identical to the hand gate: the cue is drawn identically in its two frames, so the discovered onset+1 map equals the hand-coded average of onset+0 and onset+1.

## What is and is not established

- **Established.** On native Krauzlis pixels, label-free symmetry discovery plus closed-form gate search supplies exactly the structure an ordinary gradient network needs. Both pieces are necessary, and the oracle's numbers ordered the arms correctly before any network was trained.
- **Still hand-chosen.** The Lucas–Kanade flow front end and its complex encoding; the 9×9 cell grid with weight sharing across cells; end alignment of movies (the change is always 8 frames before the report); the two time anchors and the gate-candidate family (appearance maps).
- **Scope.** One task, 2 seeds, 1,500 held-out movies, CPU only. Not yet tested on other tasks or as a predictor of the archived runs.
- **Novelty.** The components are known (circular statistics, symmetry quotients, gating, linear probes). The candidate contribution is the end-to-end result: automatic structure discovery turns a task that gradient descent fails on into one it learns from ~200 labels, and the oracle predicts which arms will learn. That claim needs the next tests before it is a finding.

## Next tests

1. Validate the plateau oracle against the archive: compute it for the archived task/representation pairs and check it predicts which runs stayed at chance.
2. Run the same pipeline unchanged on spatial binding, cued orientation and memory-guided search.
3. Remove remaining hand choices: discover the alignment (onset vs report anchoring) and test a learned or generic front end in place of Lucas–Kanade flow.

## Files

`features.py`, `discover.py`, `train_arms.py`; `discovery.json`; `results_A.json`, `results_B.json`, `results_C.json`, `results_summary.json`; logs `log_*.txt`. Data files (`data_*.npz`, `features_*.npz`) are regenerated by the scripts (seeds 41 train, 42 val).
