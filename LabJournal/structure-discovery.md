# Structure discovery on native Krauzlis — October 24

[Journal index](README.md) · [Full write-up and code](../SecondPass/StructureDiscovery/README.md)

## Question

A literature sweep, 26 inventor proposals and an adversarial review (October 23–24, scratch work outside the repo) all pointed the same way. The large sample-efficiency gains of closed-form, Bayesian and physics-inspired learners over this project's networks came from built-in structure: symmetry removal, multiplicative gating, alignment and front end. Avoiding gradient descent was not the source. Given the same structure, Adam-trained networks matched those learners or came within about 2–10× on labels. This experiment asks whether that structure can be discovered automatically and then handed to an ordinary gradient-trained network, on native pixels.

## Implementation

CPU only, no GPU and no training of repo models. 3,000 training and 1,500 held-out native Krauzlis movies (seeds 41/42; B12/B20/B28 balanced).
- Per 9×9 cell, local optical flow `z = u + i v` and appearance energy.
- **Unlabeled discovery** of the rotation nuisance via phase coherence and vanishing degree-1 moments, then canonical coordinates relative to causal history.
- A **plateau oracle**: label correlation and samples-to-detect per feature family.
- **Closed-form gate probes** over 58 candidate appearance maps.
- Seven Adam-trained GRU arms, from raw input to fully discovered structure, plus a hand-gate upper bound.

## Results

- Discovery found one global rotation shared by both patches (across-cell coherence 0.88). The renderer keeps target and foil directions 90° apart, so this is correct.
- Oracle, best label correlation: raw single channel 0.036 (~18,800 samples to detect at 5σ). Canonical, gated: 0.297 (~280 samples). The best gate is the cue onset frame.
- Gate probes ranked the two cue frames first out of 58 candidates.
- Networks, held-out balanced accuracy (2 seeds) at 200 / 1,000 / 3,000 labels:
  - Flat raw GRU: 50.7 / 53.9 / 51.5%.
  - Per-cell raw: 50.0 / 50.5 / 51.8%.
  - Gate only: 52.8 / 53.7 / 52.7%.
  - Canonicalization only: 58.0 / 57.6 / 59.6%.
  - **Both, fully discovered: 94.2 / 95.8 / 95.2% (AUC 0.986–0.990).** Identical to the hand-gate upper bound; the literal global-sector variant gives 94.0 / 95.1%.

For reference, the repo's best engineered network reached 85.7% after 38,400 presentations.

## Interpretation and limits

Automatically discovered structure is sufficient on this task, both pieces are necessary, and the oracle ordered the arms correctly before training. Still hand-chosen: the optical-flow front end, the cell grid with shared weights, end alignment of movies, the onset and report time anchors, and the gate-candidate family. Single task, 2 seeds. Not yet a general finding.

## October 25: three more tasks

[Full write-up](../SecondPass/StructureDiscovery/tasks/README.md). Same pipeline, CPU only; 3,000 train / 1,500 held-out movies per task, 2 seeds.

Disclosed changes:
- an orientation-phasor field next to flow (flow is empty on static frames);
- mask-aware statistics for short movies;
- multiclass scoring with cell-position inputs for 4-way memory search;
- an oracle target-location gate in place of the hand gate;
- unique scenes for memory search.

Results, held-out balanced accuracy at 200 / 1,000 / 3,000 labels:
- **Spatial binding: generalizes.**
  - Discovered gate: the retro-cue frame (probe 0.995).
  - Raw per-cell GRU: 67 / 66 / 64%. Fully discovered: **100 / 99.7 / 100%**.
  - Canonicalization alone also reaches 100%. The cue is in the target's own cell one frame before the probe.
  - Literal flow pipeline: 93 / 95 / 96%.
- **Cued orientation: correct gate, partial result.**
  - Discovered gate: the ± cue frame.
  - Fully discovered: 67 / 66 / 71%. Oracle gate: 70–74%.
  - Ceiling without the cue sign: 75%. Every other arm is at chance.
  - The ± sign is a discrete nuisance drawn in another cell, and the pipeline cannot carry it to the target.
- **Memory-guided search: fails** (chance 25%).
  - Every per-cell or discovered arm: 25–28.5%.
  - A flat raw GRU on the orientation field: 28 / 48 / 55%.
  - The task needs a comparison across locations, which the pipeline does not discover.
  - The oracle-gate arm's 100% is invalid: there the gate is the label.
- The plateau oracle ordered all three tasks correctly before training.

## October 25b: sign nuisance and cross-location references

[Write-up](../SecondPass/StructureDiscovery/tasks/README.md#october-25b-sign-nuisance-and-cross-location-references). Requested by the user after the failures above. CPU only; orientation field; 2 seeds.

New stages in `discover.py`:
- unlabeled cross-location references (covariance of each cell with every cell's causal history, regression-weighted transport);
- gate candidates displaced by up to ±3 cells;
- a discrete sign nuisance, screened for discreteness without labels and chosen *jointly* with the gate by the probe, acting by negation or conjugation.

Selection still uses 400 labeled training trials, in addition to the network's labels.

- **Cued orientation.** Discovered the cue frame shifted 2 cells onto the targets, plus a binary sign acting by conjugation (rotation reversal); probe 0.642 → 1.000. Networks: **100 / 99.7 / 100%** at 200 / 1,000 / 3,000 labels, matching the oracle-gate upper bound. Choosing the gate first and then the sign failed: without the sign, the right location did not score best.
- **Memory-guided search.** Unlabeled cross-location coherence reached 0.66, and the best single-channel correlation rose from 0.042 to 0.194. Per-cell networks went from chance to **37 / 39 / 46–47%** (chance 25%). The flat raw GRU gets 28 / 48 / 55%, so this is better at low label counts and worse at high ones. The discovered gate hurt (30–35%). Likely limits are the coarse orientation-only front end and the misalignment of the grid with the object offset.

## October 25c: richer memory-search field; binding and Krauzlis rerun

[Write-up](../SecondPass/StructureDiscovery/tasks/README.md#october-25c-richer-field-for-memory-search-binding-and-krauzlis-rerun). Requested by the user. CPU only.

- **Memory search, rich field.** Field: orientation phasors at σ = 0 / 1.5 / 3 px plus a colour-opponency phasor, 5 px stride, 289 cells × 4 channels, each channel scaled to unit RMS.
  - Per-cell cross-location arm: **36 / 54 / 56.5%** at 200 / 1,000 / 3,000 labels (orientation-only field: 37 / 39 / 46%). Best memory-search result so far; seed spread is ±5–7 points.
  - The flat raw GRU gets *worse* on the rich field (27 / 34 / 35%, against 28 / 48 / 55% on orientation only).
  - Engineering: chunked, memory-mapped discovery. A first unchunked attempt drove free RAM to 0.6 GB and was stopped (my process only).
  - A threshold bug for the subset covariance was found and fixed before training.
- **Binding rerun:** every arm 99.8–100%.
- **Krauzlis rerun:**
  - Original stages reproduce exactly.
  - The cross-location reference with the discovered gate gives **95.3 / 96.7 / 96.8%**, 1–1.5 points above the original study.
  - A spurious-looking sign was selected (the best of about 500 candidates, +0.055 on the probe). It is harmless, but it shows the 0.03 margin is too loose for wide searches.

## Next decision

1. Validate the oracle against the archived runs.
2. ~~Run the unchanged pipeline on spatial binding, cued orientation and memory search.~~ Done October 25.
3. ~~Add sign-nuisance discovery and cross-location references.~~ Done October 25b.
4. ~~Give memory search a richer field; rerun binding and Krauzlis.~~ Done October 25c.
5. Scale selection margins to the number of candidates searched; apply them to gates.
6. Memory search: sparser transport and object-sized windows.
7. Discover alignment and replace the hand front end.
