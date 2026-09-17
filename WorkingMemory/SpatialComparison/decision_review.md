# Independent review of the spatial comparison

Reviewed on 2026-09-13 before the first production run. This was one read-only review of `model.py`, `train.py`, `sweep.py`, `stimuli.py`, `README.md`, and `STIMULI.md`, together with the existing `model_checks.json` and `stimulus_checks.json`. The reviewer ran no training, inference, or additional test campaign and changed no runtime source.

**Outcome: no substantive issue requiring a change before the authorized profiling and production.**

## Computation and optimization

- Both models form their comparison from the previous firing-rate state and current sensory representation before updating memory. The comparator is evaluated on every physical frame, without metadata, phase gates, or an externally frozen memory state. Only the final output receives the task loss.
- The dense model uses its native vector representation in a 384-to-128-to-128 MLP. The spatial model compares concatenated spatial fields locally before mean/max pooling. Neither arm is forced into the other arm's unsuitable input geometry. Both retain a current sensory/current-memory output route and begin with identical small, nonzero comparison-output projection weights.
- Spatial recurrent weights apply fixed excitatory/inhibitory signs on the presynaptic input-channel dimension, including all kernel offsets. The 51 excitatory and 13 inhibitory channels use nonnegative rates, local 3-by-3 recurrence, zero-padding boundaries, and channel-wise rate/adaptation coefficients. Initial excitatory and inhibitory magnitudes each sum to 0.6 over both channels and kernel offsets. Rate and adaptation states are not normalized; input normalization is across channels at each position.
- No recurrent-state detach or no-gradient sensory boundary is introduced. Non-reentrant sensory checkpointing preserves the full temporal graph, including gradients into the trained sensory pathway. The supplied CPU checks show nonzero encoder, early-state, and recurrent-weight gradients in both arms; those checks establish connected computation, not adequate learning or stability over the eventual run.
- Compatible weights and Adam states transfer by parameter name. The spatial input/core/output tensors are explicitly excluded where incompatible. New components receive fresh optimizer histories. Resume replaces counters and restores model, optimizer, sampler, and RNG states. Existing sensory components use the lower learning rate; both arms retain the specified Adam epsilon, decay exclusions, and clipping policy.

## Stimuli, exposure, and evaluation

- Preserve/swap binding keeps the orientation inventory unchanged. Sample assignment and label are balanced across four-case blocks. Frequency and amplitude are shared by the two patches, while phase and pixel noise are independently refreshed across frames; these nuisances do not identify an item that moves during a swap.
- The same checkpointable task-local streams, batch size, and seeded shuffled schedule give both arms matching fresh examples and exposure. Delay evaluation restores the full stream and checks identical evidence/query/probe images and labels. The unseen-location test uses a disjoint interpolation grid, not extrapolation outside the trained region.
- Four-case assignment balancing induces dependence despite independently drawn visual nuisances. Binding uncertainty must retain those blocks along with the repeated delay/model views, as already planned by the coordinating agent.
- A single 14,400-second ledger covers both arms, their sequential profiles, training, evaluation, and reporting. Profile updates are saved separately and discarded as production initialization. Equal exposure is pinned before production from measured costs. Four validation looks select by mean AUC over the eight primary orientation/binding cells; motion remains descriptive, and held-out locations do not select checkpoints.

## Limits of the conclusion

This is a practical comparison of two designs. The dense arm retains its trained input, recurrent core, and memory-output projection; the spatial arm initializes those components anew. The supplied checks report 797,474 versus 528,538 parameters and 512 versus 21,632 recurrent state entries, respectively. Comparator geometry, normalization, state size, parameter sharing, and initialization history differ. Equal new examples do not make this a causal test of locality alone or equalize lifetime training exposure.

The current sensory field includes opponent traces, so it is not an isolated current-image representation. A gain cannot by itself identify a unique memory route. The earlier orientation diagnostic already demonstrated useful pre-probe orientation in the dense firing rates; this experiment tests whether the complete spatial design learns better comparison/binding behavior, not whether the old model had erased orientation. Performance on unfamiliar positions should be reported separately from trained-position performance. Chance-level early learning is not an architecture-failure criterion or a reason to restart the agreed run.

## Final interpretation of the completed run

Reviewed the completed `report.md`, with `analysis.json` and `parent_reference.json`, as the interpretation follow-up to this same review. No further model inference, tests, or experiments were performed.

Both arms completed 35,200 new episodes and selected their terminal update 4,400. The spatial design is a useful feature-location association candidate: binding D24 reaches 93.36% versus 49.61% for dense (paired gain 43.75 percentage points, 95% interval 39.84–47.86), and its 94.53% result at unseen centers supports interpolation across the tested positions. Since swap trials change both locations, remembering only one location suffices; these scores do not establish two-item storage capacity.

The paired unchanged-parent reference makes the single-item D12 gain meaningful beyond dense deterioration: spatial reaches 79.69%, compared with parent 63.28% and dense 46.48%. All three remain near chance at single-item D24. Binding and single-item tasks differ in change sizes, rendering, and query content, so their scores are not a controlled memory-load comparison.

The candidate is not a general replacement. Spatial motion D24 reaches only 25% balanced accuracy, versus 67.58% for dense and 77.54% for the preserved parent; its AUC of 0.762 also prevents interpreting chance decisions as complete information loss. The schedule gave motion only 10% of updates and excluded it from selection. Preserve the spatial candidate and the old parent, report the trade-off, and retain the practical-design rather than isolated-locality interpretation. The proposed report does so accurately; no additional computation is requested by this review.
