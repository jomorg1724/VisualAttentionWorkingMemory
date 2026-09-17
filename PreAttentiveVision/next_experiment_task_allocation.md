# Proposed next experiment: contour-focused training with all-task retention

Status: user authorized execution. Researcher is implementing the schedule comparison and will launch within the remaining compute allowance; run receipts, rather than this status sentence, establish actual launch and completion.

## Question

Can reallocating training toward the remaining weak task bring a shared encoder to high performance across all seven current task distributions?

The late-SE checkpoint at total step1512 scored 100% motion and orientation,99.3% contrast,100% spatial frequency,99.6% chromatic discrimination and99.8% natural-spectrum discrimination, but63.4% contour accuracy. Its contour validation BA rose51.3→56.3→63.8% during the last three looks. Both the weak score and rising trajectory make training allocation a concrete next development question. They do not demonstrate an exhausted architecture or prove task interference.

## Two arms from the same parent

Use the trained late-SE checkpoint as the shared starting point for both arms, preserving all learned weights and compatible optimizer state. Retain the seven stimulus laws, ordered decoder, task heads, optimizer settings, batch size32, and model architecture.

1. **Uniform continuation:** one batch for each of seven tasks in every seven updates.
2. **Contour emphasis with retention:** six contour batches plus one batch from each other task in every12 updates. Thus half of updates train contours; each other task receives one twelfth.

This is a change in task-sampling allocation, not a claim that static loss scaling is equivalent. At equal optimizer updates the focused arm receives3.5 times as many contour examples and7/12 as many examples from each other task. Those unequal per-task exposures are the intervention and must be reported, not hidden as a matched-exposure architecture test.

Use shared task-specific fresh example streams so corresponding draws match across arms up to their common per-task prefixes. Establish these streams explicitly as the new experiment's sampler protocol: a changed schedule cannot be described as bit-for-bit continuation of the prior single-stream presentation order.

## Exposure and assessment

A concrete target is1008 additional updates per arm, which is divisible by both7 and12. Each arm sees32,256 new pairs. Uniform:4,608 pairs/task. Focused:16,128 contour pairs and2,688 pairs for each other task. Confirm this target fits the remaining approximately1008 seconds of compute, including sequential training and fresh paired evaluation, before launch; reduce the equal update target in84-update increments if required. The intended budget is a finite exploration, not a guaranteed acquisition horizon. No automatic cap extension.

Inspect validation curves at three planned points and choose checkpoints by the minimum task balanced accuracy; break ties by mean task AUC. Report all task accuracies and AUCs rather than only the selection scalar. Minimum accuracy is an engineering criterion here; different task difficulty and chance levels remain explicit.

The ambitious engineering target is at least95% balanced accuracy on every current task family, including the aggregate contour task. Also report contour accuracy by jitter, so an aggregate result cannot hide difficulty dependence. If that target is not reached, a useful improvement would be a clear contour gain over uniform continuation with the other six tasks retaining strong performance; show uncertainty rather than equating one sample percentage with a guaranteed population level.

Use fresh shared evaluation pairs, with source-image clustering for natural stimuli and disclosure that the BSDS source-photo pool is unchanged. These are development tests of the defined distributions, not proof of universal visual competence or biological preattention.

## What the outcomes would tell us

- Focused training improves contours while preserving the other domains: the allocation is a useful training change. It does not by itself identify gradient conflict as the cause.
- Both arms improve comparably: continued training is sufficient to explain the gain; the allocation change has not earned adoption.
- Contours improve but other tasks regress: the next question is maintaining shared skills under specialization, with actual tradeoff evidence.
- Contour scores remain low: inspect the trajectory before inferring a plateau. A spatial contour-grouping readout becomes a specific next candidate, not an automatic conclusion that the encoder discarded contour information.

Task balancing is an established concern in multitask optimization: [Chen et al.,2018, GradNorm](https://proceedings.mlr.press/v80/chen18a.html). That paper studies adaptive gradient balancing; the proposed fixed sampling comparison is simpler and does not claim to reproduce GradNorm or establish its mechanism in this project.
