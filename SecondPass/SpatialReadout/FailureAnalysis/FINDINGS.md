# Why the spatial/sequence tasks are failing

## Conclusion

The strongest localized behavioral failure is **cue-conditioned use of task-relevant evidence**, not simply insufficient long-delay memory. The model responds to cue location, but largely fails to combine location, instruction meaning, local change and prior evidence into the correct answer. This is a functional diagnosis, not proof of one broken internal module or of a required architecture.

Audits and frozen interventions used actual terminal6760. Main model weights and checkpoint remained unchanged; no training or fitted probes. Frozen experiment covered15 conditions with32 observations each,160 unique base episodes, within its900-second allowance. Complete records: [DIAGNOSIS.md](DIAGNOSIS.md), [task audit](task_audit.md), [training audit](training_audit.md).

## Direct evidence

1. **Failures precede long retention.** Native matched D0/D24 trials give orientation50%/50%, binding46.875%/50%, cued-duration25%/25%. Ring/orientation/binding D0 have their relevant sample and probe concurrently in the final raw3-frame stack. Motion D0 still requires8-transition integration, so it is not a memory-free condition.
2. **Cue location is sensed but used poorly.** In the32 native motionD0 draws, positions0/3 always produced left and1/2 always up, independent of the balanced correct winner. Relocating the cue changed23/32 choices but only2/31 answer-changing pairs were correct in both conditions. This is measured spatial response bias, not useful target selection.
3. **Instruction/evidence conjunction is missing from behavior.** Flipping only orientation cue sign changed29 labels but2 choices;0/29 label-changing pairs were both correct. Reversing rotation signs with cue fixed also changed2/32 choices. Moving the cue altered probabilities substantially more than either sign/evidence edit. Weak output use does not establish absent internal information.
4. **No demonstrated renderer or broken-gradient explanation.** All35 cells matched native adapters; checked labels/cues/membership were coherent. Three nonlearned pixel observers solved32/32 D0 ring, signed orientation and binding samples using visible cues plus known geometry. Diagnostic backward paths reached early frames and all major modules; no accidental freezing/detach/routing error was found. This does not establish neural acquisition ease.
5. **Acquisition is failing, not merely generalization.** Recent failing-task train losses remain near class-prior baselines, including D0. During the last650 updates, sensory tasks contributed98.15% of squared logged whole-model gradient norms (independently recomputed). These norms include heads and precede Adam scaling: they are NOT a98% share of learning. Small frozen gradient checks show balanced spatial cancellation and one negative sensory/spatial alignment; causal interference requires a controlled test.

## Why the successful tasks do not settle this

The sensory tasks permit direct two-frame comparison within the raw stack. They have different geometry and photometry: central broad gratings versus small peripheral patches, or one large moving-dot field versus multiple small fields requiring temporal counts. Their success does not prove trained local patch encoding, cue-rule combination, binding or list memory.

The sign difference occupies8 spatial pixels and phase distinctions can be4 pixels, creating plausible acquisition burdens. Their disappearance in learned features is unmeasured; ringD0 fails despite a128-pixel cue, so tiny glyphs alone cannot explain all failures.

Recognition is a distinct failure: almost one quarter of its branch examples are always-negative empty lists. Those controls are learned, while nonempty recent train losses remain near chance. Exact study/probe rasters and labels are correct. Encoding multiple image identities, separating study from probe, and avoiding probe-period interference remain unlocalized; neither spectral-detail success nor empty specificity demonstrates membership memory.

## Next discriminating decision—not launched

Do not select a bigger memory module from these results. First localize the native D0 spatial comparison: can frozen intermediate features support cue location/sign and local sample/probe orientation, and can a small independent diagnostic comparator combine them? Use grouped train/validation/test splits, balanced cue/label controls and no ground-truth target as deployed-model input. Positive decoding would localize recoverable information; negative decoding alone would not prove erasure.

If ingredients are accessible but deployed use fails, investigate relational readout/optimization. If local evidence is poorly recoverable, prioritize spatial sensory encoding before retention. Separately replicate shared-parameter gradient/Adam alignment before attributing the failure to joint-task interference. Any subsequent single-task or architecture training comparison needs its own explicit scope and authorization; none was started.

## Concrete engineering caveats

The inherited diagnostic grouping misnames new spatial recurrence/readout tensors as inactive heads. It does not exclude them from Adam or affect total logged gradient norm. One imported renderer helper was omitted from the historical frozen manifest; current copies match but past identity cannot be proved from that snapshot. Neither issue explains the measured behavior. Prior parent3393 was jointly trained on all13 tasks, not sensory-only.

All findings are single-checkpoint exploratory evidence. Small paired samples do not establish population mechanisms or architectural necessity. The strongest result is differential use of cue location versus instruction/evidence, not significance of small accuracy differences.
