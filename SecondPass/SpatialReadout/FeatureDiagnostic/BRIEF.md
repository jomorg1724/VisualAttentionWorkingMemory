# Authorized frozen feature/comparator diagnostic

User: "okay, proceed with the tests" following the proposal to decode cue location/sign and patch orientations from frozen intermediate features and test a small diagnostic comparator on the same native D0 task.

## Scope

- Actual terminal6760 from `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt`; expected SHA256 `1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841`.
- Native orientation_cued and spatial_binding, D0. No modified teaching, stimulus laws, deployed readout or main-model parameter updates. Analysis readouts may fit independently; they are not deployed competitors.
- Decode instruction location/sign and all four actual axial sample/probe orientations (cos2theta/sin2theta), at specified early/late visual, ConvGRU and final readout layers. Distinguish present-frame evidence from retained sample information. Preserve spatial information in feature extraction; disclose pooling/projections and their limitations.
- Small task-label diagnostic comparator receives only extracted activations, never ground-truth angles, cue identity, selected true target, labels or privileged metadata as inputs. Metadata may supervise diagnostic probes and evaluate strata. If a structured decoder uses inferred cue/angles, all inputs at evaluation must be predicted, with trained-fit separation; no oracle substitution in operational score.
- Independent base-episode train/validation/test streams; keep every timestep/location and any paired edit for one base in the same split. Train-only scaling/projection/feature selection, validation-only fit selection, single final test after selection. Save seeds, per-split counts, frozen feature files, fitted probe state and predictions.
- Compare comparator with frozen deployed decisions on identical held-out examples, include a simple label-prior/cue-only or shuffled-association control, and report location/sign/label strata. Angular precision in axial degrees, and relative to actual change sizes, not scalar latent levels. Negative probes are inconclusive about erasure.

## Bounded execution

One researcher owns implementation and execution in this directory. One local MPS extraction worker, CPU threads at most2, no cloud. New finite1800-second wall cap persisted immediately before first accelerator profile/extraction; includes feature extraction, CPU fitting, final evaluation and report. No automatic renewal. Target1024/256/512 independent episodes per task for fit/validation/test, subject to one small extraction-cost measurement; pin lower counts before production if needed, never based on test performance. Reserve time for reporting. Main checkpoint/state and source hashes must be unchanged; direct model/extraction-wrapper logits must match.

No additional training or architecture experiment, gradient-conflict campaign, full-suite sweep or endless review gates. Save a concise actual-result report with limitations and candidate next decision. Parent verifies artifacts/results and updates journal.
