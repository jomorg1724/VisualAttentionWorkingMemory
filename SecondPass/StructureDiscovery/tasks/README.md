# Structure discovery on three more tasks (October 25)

[Krauzlis study](../README.md) · [Journal entry](../../../LabJournal/structure-discovery.md)

**Question.** The same pipeline (unlabeled nuisance discovery → plateau oracle → closed-form gate probe → ordinary Adam-trained GRU arms) took native Krauzlis from chance to 94–96%. Does it generalize to spatial binding, cued orientation and memory-guided search?

**Answer: partly.**
- **Spatial binding:** fully generalizes. Discovery finds the retro-cue frame as the gate, and canonicalization takes a GRU from 64–67% to 100% from 200 labels.
- **Cued orientation:** discovery finds the right gate, and the network reaches the ceiling available without the cue's sign (about 71% versus a 75% ceiling). The pipeline cannot represent that sign.
- **Memory-guided search:** fails. Every discovered structure leaves the networks at chance, while a plain flat GRU learns to 55% (chance 25%).

> **Update (October 25b, [below](#october-25b-sign-nuisance-and-cross-location-references)):** two new discovery stages fix much of both failures:
> - discrete sign nuisance, selected jointly with a displaced gate;
> - cross-location references from unlabeled coherence with other cells' history.
>
> **Cued orientation now reaches 100% from 200 labels.** Memory search rises from chance to 37–47% for the per-cell networks: better than the flat GRU at 200 labels, below it at 3,000.
>
> **Update (October 25c, [below](#october-25c-richer-field-for-memory-search-binding-and-krauzlis-rerun)):**
> - A richer field (three orientation scales, colour opponency, 5 px stride) lifts memory search to 54 / 56% at 1,000 / 3,000 labels, the best result on that task, with large seed variance.
> - Rerun with the full pipeline, binding stays at 100%.
> - Krauzlis improves to 96.8% with cross-location references.

CPU only, 2 threads per process. The user's pod_to_local08 GPU run was left untouched. About 10 minutes of rendering and about 1.5 hours of training.

## Changes from the Krauzlis pipeline (disclosed)

1. **Orientation field.** Flow is nearly empty for static Gabor and object frames, so `features.py` also computes a per-cell doubled-angle structure-tensor phasor `o = (sxx − syy) + 2i·sxy`. Every task was run on both fields. The flow runs are the literal unchanged pipeline, with a reduced set of three arms.
2. **Short movies.** Movies here have 3–29 frames, where Krauzlis has 29–45. Discovery statistics now use only trials valid at the frames involved. An anchored gate candidate whose frame does not exist in a movie falls back to a uniform gate. The flow front end skips Lucas–Kanade lags longer than the movie. On Krauzlis these changes are no-ops.
3. **Multiclass.** Memory search is 4-way:
   - balanced accuracy is mean per-class recall, and AUC is macro one-vs-rest;
   - the oracle correlation is the best one-vs-rest correlation;
   - the network outputs 4 logits per cell, summed over gated cells, trained with cross-entropy;
   - each cell's input also carries its normalized (x, y), so a location answer can be expressed.
4. **Upper-bound arm.** The Krauzlis hand gate (cue frames) is replaced by an **oracle gate**: a Gaussian (σ = 10 px) on the true target location. In memory search the target location *is* the label, so that arm reads the answer. Its 100% is **invalid** and is not an upper bound.
5. **Memory-search sampling.** The repo stream reuses scenes across timing conditions for equal trial ids, and assigns targets as a permutation over each block of 4 consecutive ids. Each trial therefore gets its own id, in complete blocks of 4. This gives 3,000 unique scenes with balanced targets, spread over all 288 conditions. A first render without this had about 70 unique scenes and unbalanced labels; it was discarded.

Data: 3,000 training movies (seed 41) and 1,500 held-out movies (seed 42) per task, from the repo renderers:
- `WorkingMemory.SpatialTaskBattery` for binding and orientation, delays 0/4/12/24, balanced;
- `SecondPass.MemoryGuidedSearch` for memory search.

Two seeds per arm.

## Discovery (labels unused for stages 1; train split only for 2–3)

| Task, field | Sectors (selected) | Best gate (probe CV BA; ungated) | Best label correlation → samples at 5σ |
|---|---|---|---|
| Binding, orientation | 2 (per-cell) | **end-1 = retro-cue frame** (0.995; 0.535) | 0.855 → ~34 |
| Binding, flow | 1 (global) | end-1 (0.801; 0.580) | 0.568 → ~78 |
| Orientation, orientation | 5 (per-cell) | **start+0 = ± cue frame** (0.642; 0.495) | 0.091 → ~3,000 |
| Orientation, flow | 1 (global) | end-15 (0.536; 0.491) | 0.036 → ~19,000 |
| Memory search, orientation | 3 (per-cell) | start+8 / sample frames (0.314; chance 0.25) | 0.042 → ~14,000 |
| Memory search, flow | 1 (global) | end-13 (0.286) | 0.025 → ~40,000 |

The oracle ordering matches the network results below: binding is easy, orientation is middling and memory search is hard for the structured arms.

## Networks: held-out balanced accuracy, mean of 2 seeds, at 200 / 1,000 / 3,000 labels

**Spatial binding (chance 50%)**

| Arm | Orientation field | Flow (literal) |
|---|---|---|
| Flat GRU, raw | 51.0 / 60.7 / 59.5 | 50.7 / 63.3 / 67.2 |
| Per-cell GRU, raw | 67.1 / 65.7 / 63.6 | — |
| + canonicalization only | **100.0 / 99.5 / 100.0** | — |
| + discovered gate only | 92.2 / 99.7 / 100.0 | — |
| + both (fully discovered) | **100.0 / 99.7 / 100.0** | 92.6 / 95.2 / 95.7 |
| + both, global-sector canonicalization | 94.0 / 99.2 / 99.8 | 62.2 / 74.4 / 72.9 |
| + canonicalization, oracle gate | 100 / 100 / 100 | — |

Unlike Krauzlis, canonicalization alone suffices here. A likely reason: the retro-cue ring appears in the target's own cell one frame before the probe, so each cell's GRU can form "I was cued and I changed" from its own appearance input without a gate. In Krauzlis the cue came 20–40 frames earlier. With flow, discovery chose the global sector, but the per-cell arm is the one that works (95–96%). With the orientation field it chose per-cell, which is right.

**Cued orientation (chance 50%; ceiling without the cue sign 75%)**

| Arm | Orientation field | Flow (literal) |
|---|---|---|
| Flat GRU, raw | 49.8 / 49.7 / 50.2 | 50.5 / 51.3 / 51.5 |
| Per-cell raw / canon only / gate only | 48.9–51.5 at every n | — |
| + both (fully discovered) | **66.6 / 65.7 / 71.0** (AUC 0.72–0.80) | 50.1 / 50.4 / 49.5 |
| + both, global-sector canonicalization | 50.8 / 49.7 / 50.2 | 50.3 / 48.6 / 47.9 |
| + canonicalization, oracle gate | 70.4 / 73.4 / 74.4 | — |

- The label is 1 only if the cued item rotates in the *sign shown by the ± glyph*. Label 0 is half "no rotation" and half "opposite rotation".
- A detector that only sees "the target rotated" therefore tops out at 75% balanced accuracy (from the renderer code).
- Both the discovered-gate and the oracle-gate arms sit just under that ceiling.
- The missing piece is the sign: a discrete ±1 that multiplies the label. The glyph is drawn in a different cell from the target, so a per-cell GRU never sees it.
- Both pieces are again necessary: each alone is at chance.
- Flow carries no signal for a static rotation.

**Memory-guided search (4-way, chance 25%)**

| Arm | Orientation field | Flow (literal) |
|---|---|---|
| Flat GRU, raw | 27.9 / 47.8 / **54.8** (AUC 0.82) | 25.4 / 30.8 / 28.1 |
| Per-cell raw / canon / gate / both | 25.2–26.0 at every n | 24.8–25.1 |
| + both, global-sector canonicalization | 26.1 / 27.1 / 28.5 | 24.4 / 24.8 / 25.1 |
| Oracle gate (target location = label) | 100 (**invalid: leaks the label**) | — |

The answer requires comparing each array patch with a sample shown at a *different* place (the center) and time. Per-cell canonicalization references each cell to its own past, so it cannot make that comparison. The global reference can make it only through one orientation number per cell. Only the flat GRU, which sees every cell together, learns. Here the discovered structure is the wrong structure, and the oracle correctly predicted the failure (correlations 0.025–0.042).

## What this establishes

- **Generalizes:** label-free canonicalization plus gate discovery is sufficient on a second cued change task (binding). There, the cue is local and the nuisance is a per-location orientation.
- **Discovered correctly but not representable:** cued orientation needs a discrete sign nuisance that lives in another cell. The pipeline found the right frame but has no way to transport the sign to the target.
- **Fails:** memory search needs comparison across locations (store-then-match by translation). The pipeline discovers only same-location references.
- **The oracle predicted all three outcomes before any network was trained.** That is the most reusable part of the result.

## October 25b: sign nuisance and cross-location references

The two failure modes above became two new discovery stages in `discover.py`, plus one change to gate search. Each was rerun on cued orientation and memory search with the orientation field; binding and Krauzlis were not rerun. The original stages reproduce exactly: identical gate scores and oracle values.

**New stages**
- **1b. Cross-location references (labels unused).**
  - For every cell, the covariance of its current field with every cell's causal history, pooled over the last L steps (mean-removed).
  - Pairs above the same 4/√n threshold define a transport `R_i = Σ_j β_ij P_j`, with regression weights β_ij that include phase.
  - Canonical coordinates are then `w = z·conj(R)/|R|`. With only self-pairs significant, this reduces exactly to the per-cell canonicalization.
  - Variants: `Wx` includes the self-pair; `Wxo` excludes it.
  - Invariant to a global rotation (max difference 2e-9).
- **3′. Displaced gates.** A cue may point at a location rather than mark it. Every anchored map is also tried translated by up to ±3 cells, scored by the same probe.
- **3b. Sign nuisance, chosen jointly with the gate.**
  - Candidates: appearance energy at each anchored frame present in every movie, measured under the gate or under its unshifted cue map.
  - Unlabeled screen: keep a candidate only if a two-cluster split explains at least 75% of its variance (a discrete ±1).
  - Probe: each discrete sign acts on the canonical field by negation or by complex conjugation (reversal of rotation).
  - It is searched jointly with the 20 best gates. A sign hides the gate's value: without it, the right location does not score best (0.640 versus 0.642 here).
  - A (gate, sign) pair is kept only if it beats the best unsigned gate by at least 0.03 CV BA.
- **Location-aware per-cell probe.** Reported only, not used for selection.

**Label budget.** As in every stage-3 result in this study, including Krauzlis, gate and sign selection use 400 labeled training trials. The network label counts below are additional to those 400.

**Discovery**
- **Cued orientation.** Selected: the ± cue frame (start+0) shifted 2 cells down onto the target locations. Sign: a perfectly binary energy feature (separation 0.99996, 50/50), acting by **conjugation**, i.e. rotation reversal.
  - Probe CV BA: 0.642 → **1.000**.
  - Label correlation of the gated signal: 0.091 → **0.841**.
  - The source was measured under the shifted gate, i.e. blank cells whose contrast against the frame mean depends on how many glyph pixels are drawn. It tied with energy measured on the glyph itself (both 1.0).
- **Memory search.**
  - No sign passes the margin, which is correct.
  - Cross-location coherence: 3,170 significant cross-cell pairs, maximum 0.66. Same-time across-cell coherence was 0.087.
  - Best single-channel label correlation: 0.042 → **0.194**, i.e. about 14,000 → about 660 samples to detect.
  - The location-aware cell probe did *not* show this gain: 0.329 versus 0.336 raw.

**Networks: held-out balanced accuracy, 2 seeds, at 200 / 1,000 / 3,000 labels**

*Cued orientation (chance 50%):*

| Arm | Result |
|---|---|
| Previous fully discovered (cue-frame gate, no sign) | 66.6 / 65.7 / 71.0 |
| Displaced gate, no sign | 68.3 / 74.0 / 74.5 (the no-sign ceiling is 75) |
| **Displaced gate + discovered sign (conjugation)** | **100.0 / 99.7 / 100.0**, every delay ≥ 99% |
| Displaced gate, discovered sign broadcast as an input | 99.9 / 99.7 / 100.0 |
| Sign + oracle target gate (upper bound) | 100.0 / 99.9 / 100.0 |
| Displaced gate + cross-location reference, no sign | 76.8 / 78.5 / 80.9 |

The last row exceeds the no-sign ceiling. The transported reference likely carries some of the glyph's orientation history from neighboring cells; this is not established.

*Memory-guided search (4-way, chance 25%):*

| Arm | Result |
|---|---|
| Flat GRU, raw (best earlier arm) | 27.9 / 47.8 / 54.8 |
| Any per-cell arm, earlier | 25–28.5 |
| **Per-cell GRU, cross-location reference** | **37.2 / 39.1 / 45.8** |
| Per-cell GRU, cross-location only (self excluded) | 37.6 / 38.1 / 47.3 |
| Cross-location + discovered gate | 29.9 / 30.1 / 34.5 |
| Cross-location only + discovered gate | 30.9 / 30.0 / 34.9 |

- Cross-location references lift the per-cell networks from chance to 37–47%. At 200 labels they beat the flat GRU (37 versus 28%).
- They plateau below it at 3,000 labels (46–47 versus 55%).
- The discovered gate is weak (probe 0.35 against chance 0.25) and hurts. The 0.03 margin rule is applied only to signs, not to gates.
- Likely ceilings, untested:
  - one complex orientation number per cell is a poor object descriptor;
  - the 25 px sample-to-patch offset is not a multiple of the 10 px cell stride, so the transported reference compares different crops of the object.

**What this adds**
- **Cued orientation is now solved from 200 labels.** Every structural element was discovered: the cue frame, its displacement to the target, the discrete sign, and the correct group action (conjugation over negation).
- **Memory search is partly fixed.** Unlabeled cross-location coherence finds the sample → array transport, and the oracle predicted the improvement. The front end (one orientation phasor on a coarse grid) now limits it.
- **Sequential selection fails when a nuisance hides the gate's value.** Gate and nuisance had to be selected jointly.

## October 25c: richer field for memory search; binding and Krauzlis rerun

**Rich field** (`features.py ... rich`, folder `memory_guided_search_rich/`):
- four complex channels per cell:
  - orientation phasors of the structure tensor at Gaussian scales σ = 0, 1.5 and 3 px;
  - a colour-opponency phasor (R−G) + i(B − (R+G)/2);
- a 5 px stride (17 × 17 = 289 cells), so the 25 px sample-to-patch offset is exactly 5 cells.

Same 3,000 / 1,500 movies as before.

**Pipeline changes for multi-channel fields** (no effect on single-channel tasks):
- Units are (channel, cell); gates live on spatial cells and are shared across channels.
- Each channel is scaled to unit RMS on the training split. Raw RMS: colour 0.039, orientation 1e-3 / 2e-4 / 9e-5.
- Gate shifts move in 10 px steps.
- The cross-location covariance is estimated on a 1,000-trial subset, with threshold 4/√1000.
  - A first run used 4/√3000 by mistake; it is kept as `discovery_rich_tau3000.json`.
- Large arrays are computed in chunks and stored as memory-mapped `.npy` files.
- Each cell's GRU sees all four channels of that cell.

A first, unchunked discovery attempt brought free RAM to 0.6 GB and was stopped. Only my own process was stopped; the user's GPU run was unaffected.

**Discovery (rich field)**
- Gate: start+7, probe 0.328 (chance 0.25). No sign selected.
- Cross-location: 1,006 self-pairs and 342,047 cross pairs significant. The transport is dense; the 75% overlap between neighbouring 5 px cells accounts for much of this.
- Per-cell probe: cross-location 0.42 against raw 0.30 (orientation-only field: 0.31–0.33).
- Best single-channel label correlation: 0.215, about 540 samples to detect (orientation-only field: 0.194).

**Memory search networks (4-way, chance 25%; 200 / 1,000 / 3,000 labels; 2 seeds, range in brackets)**

| Arm | Rich field | Orientation-only field |
|---|---|---|
| Per-cell, cross-location | 36.2 [31–41] / **54.4** [51–58] / **56.5** [49–64] | 37.2 / 39.1 / 45.8 |
| Per-cell, cross-location, self excluded | 36.3 / 54.1 / 56.3 | 37.6 / 38.1 / 47.3 |
| Flat GRU, raw | 26.8 / 33.8 / 35.1 | 27.9 / 47.8 / 54.8 |
| Per-cell GRU, raw | 23.8 / 24.9 / 24.9 | 25.2–25.7 |

- The richer field raises the discovered-structure arm by 10–15 points at 1,000–3,000 labels. It is now the best memory-search result: 54 versus 48% for the flat GRU at 1,000 labels; at 3,000, 56.5 versus 54.8%, within seed noise.
- It gives nothing at 200 labels.
- Seed variance is large, ±7 points at 3,000 labels.
- The same field *hurts* the flat raw GRU (35 versus 55%). It has about 3,200 inputs per frame, and only the discovered references make the extra channels usable.
- Memory search remains far from solved. The obvious remaining limits are the dense, overlapping transport and the 20 px windows, which are smaller than the 28 px objects.

**Binding rerun with the 25b pipeline (orientation field)**
- Original stages reproduce exactly.
- Gate: retro-cue frame shifted 1 cell (a trivial gain). No sign.
- Only 16 self-pairs pass the cross-location threshold, so most cells get no reference.
- Networks are unchanged at the ceiling. Every arm (discovered gate; cross-location with and without gate; cross-only with gate) scores 99.8–100% at every label count.

**Krauzlis rerun with the 25b pipeline (flow field)**
- Original stages reproduce exactly.
- Gate: cue onset frame, unshifted. This is the hand gate, so the gated arm reproduces the original 94.2 / 95.8 / 95.2% exactly.
- A sign *was* selected: cue-frame energy acting by conjugation, probe 0.635 against 0.580.
  - I have no mechanism for it; Krauzlis change directions are random in sign.
  - It was the best of about 500 (gate, sign) candidates on 400 trials, so it is probably selection noise passing the 0.03 margin.
  - It is harmless: 94.4 / 94.5 / 95.8% with the sign, 94.5 / 93.9 / 95.7% with it as an input.
  - The 0.03 margin is too permissive for a search this wide.
- **Cross-location reference with the discovered gate: 95.3 / 96.7 / 96.8%**, AUC up to 0.995. This is the best Krauzlis result, 1–1.5 points above the original study, and every baseline length is at least 95%.

## Next tests

1. Tighten selection: scale the sign and gate margins to the number of candidates searched (Krauzlis's sign shows the current 0.03 is too loose), and apply the margin to gates too.
2. Memory search: sparser transport (stricter threshold or a top-k number of sources per cell), and windows matched to object size.
3. Field selection by oracle (orientation won on binding, cued orientation and memory search; flow on Krauzlis), and oracle validation against the archived runs.

## Files

`render_all.sh`, `train_task.sh`, `summarize.py`, `results_summary.json`; per task: `discovery_{orient,flow}.json`, `results_{orient,flow}_main.json`, `log_*.txt`. October 25b: `results_orient_v2.json`. October 25c: `spatial_binding/results_orient_v2.json` and `discovery_orient_v1.json`; `krauzlis/` (copies of the original data; `discovery_flow.json`, `results_flow_v2.json`); `memory_guided_search_rich/` (`discovery_rich.json`, `results_rich_main.json`; large `.npy`/`.npz` files are gitignored); `discover_v25b_snapshot.py` is the script version that ran the binding and Krauzlis reruns. For orientation and memory search, `discovery_orient.json` now includes the new stages, and `discovery_orient_v1.json` keeps the first run. Those tasks' `features_orient_*.npz` now carry the selected v2 gate, so the v1 orient results are records of the earlier gate. The `data_*.npz` and `features_*.npz` files are regenerated by `features.py` / `discover.py`.
