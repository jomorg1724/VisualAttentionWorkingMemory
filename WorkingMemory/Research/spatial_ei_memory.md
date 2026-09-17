# Spatial E/I memory: rationale and proposed test

Research/design note, 2026-09-13. No architecture has been implemented or trained by this note.

Keeping the 64 x 13 x 13 sensory field is a reasonable way to give recurrent memory explicit spatial organization. The strongest new hypothesis is that this helps preserve feature-location binding across several objects. It is not established that compression caused the current single-item orientation failure.

## What the current evidence says

The implementation in `../RecurrentComparison/model.py` applies a 64-to-8 pointwise convolution, flattens the resulting 8 x 13 x 13 tensor, and projects its 1,352 entries to 128 input features for a 256-unit E/I population. Flattening alone is an invertible rearrangement: it does not remove location information. The channel reduction and dense projection impose compression; the dense projection mixes locations and does not enforce shared local processing.

The latest [frozen orientation diagnostic](../RecurrentComparison/OrientationDiagnostic/report.md) found, after 24 blank frames:

- Sample orientation decoded from pre-probe firing rates with 4.25 degrees mean absolute error.
- A comparator trained only on same/different labels, using pre-probe firing rates and an independently encoded probe, reached 73.44% balanced accuracy versus the deployed model's 50.00%.
- Comparing independently predicted sample/probe angles reached 87.30%, but its diagnostic decoders received additional angle supervision during fitting.
- The label-only comparator was worse than the deployed model at zero delay; it is not a demonstrated general replacement. Small 7.5-degree changes remain difficult.

Thus useful sample information survives. Processing/comparing the probe and using the stored representation are immediate development targets. A spatial state could also preserve more precision, but the diagnostic has not measured a benefit from removing compression.

## A concrete convolutional version of our E/I model

Input field and the two persistent states:

\[
H_t,R_t,A_t\in\mathbb R^{B\times64\times13\times13}.
\]

Here R is nonnegative firing-rate state; A is its adaptation trace. Use a learned 1 x 1 input projection that retains all 64 channels, and a learned 3 x 3 recurrent convolution with padding one:

\[
J_t=K_x*H_t+K_r*R_{t-1}-g\odot A_{t-1}+b,
\]
\[
R_t=(1-\alpha)\odot R_{t-1}+\alpha\odot\operatorname{ReLU}(J_t),
\qquad
A_t=(1-\beta)\odot A_{t-1}+\beta\odot R_{t-1}.
\]

The two updates use the old state synchronously. Each location receives current sensory input plus recurrent input from all 64 channels at itself and its eight neighbors. Parameters are shared across positions and time; states differ at each position. A retained channel count does not guarantee lossless encoding through learned dynamics.

Shapes: K_x is [64,64,1,1]; K_r is [64,64,3,3]; b, g, alpha and beta broadcast from [1,64,1,1]. Initialize R and A to zero. Retain the current bounded learned timescales as an initial design: tau_r in (1,32), tau_a in (4,128), g in (0,0.5), alpha = 1-exp(-1/tau_r), beta = 1-exp(-1/tau_a). These are model-frame units, not physiological milliseconds. Adaptation is continuity with our existing model, not evidence that adaptation always improves retention.

For approximately 80% excitatory / 20% inhibitory units, use 51 E and 13 I channels:

\[
(K_r)_{o,i,u,v}=\operatorname{softplus}(\Theta_{o,i,u,v})\,d_i,
\qquad d_i\in\{+1,-1\}.
\]

The sign belongs to the presynaptic INPUT channel i, across all output channels and offsets. This imposes Dale-style recurrent output signs. The arbitrary signed CNN features, input projection and task decoder are computational interfaces; the entire network would not thereby become a Dale-constrained biological circuit.

Use zero padding as an explicit finite visual-field boundary approximation; circular padding would connect opposite image edges. Balance initial E/I input magnitudes over both channels and kernel offsets. Preserve nonnegative rates rather than applying centered LayerNorm to R or A. If afferent normalization is needed, normalize channels locally at each position; normalization over the whole map adds global coupling. Preserve full BPTT and gradient clipping, and monitor state/gradient scale in any later implementation. These are design choices, not completed stability measurements.

## Compare while the spatial structure is still available

Storage alone is insufficient. A learned convolutional comparator could receive previous memory and current sensory input:

\[
C_t=\phi\left(K_c*[R_{t-1};H_t]\right)
\in\mathbb R^{B\times64\times13\times13},
\]

where concatenation gives 128 channels and K_c is [64,128,1,1], optionally followed by a spatial convolution. The task head can produce a local change map, or pool AFTER comparison for a global decision. A nonlinear comparator is necessary; a single affine projection alone is not an expressive equality test.

This exposes old memory and new input separately before the new input updates memory. It can operate every frame with fixed state size, without an oracle probe flag, selective memory freeze, or attention mechanism. H_t already includes opponent sensory history, so it is not a purely current-image representation. Comparing the analogous vector-state interface first would target the diagnosed failure directly; moving the same functional interface onto a grid would test the additional spatial hypothesis.

## Scientific motivation and limits

- Sprague, Ester and Serences (2014) reconstructed remembered locations from population activity in occipital, parietal and frontal regions, with degraded representations under greater memory load. This supports spatial mnemonic content, not a literal convolutional memory mechanism. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4181677/).
- Oldenburg et al. (2024) found that spatial arrangement and feature preference jointly govern recurrent activation/suppression in mouse V1. This motivates local feature-dependent E/I connectivity. Their experiment concerns visual cortical circuits, not validation of our working-memory model. [Primary paper](https://www.nature.com/articles/s41593-023-01510-5).
- Linsley et al. (NeurIPS 2018) used horizontal gated recurrent units to learn spatial contextual dependencies. This is an established ML precedent for recurrence within feature maps; it does not establish delay-period storage in the proposed rate model. [Primary paper](https://papers.nips.cc/paper_files/paper/2018/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html).
- Park, Zhang and Choe (2025 preprint) study convolutional lateral recurrence with E/I-inspired weight regularization. Their study uses image classification and repeatedly processes a static input; it is not a visual working-memory demonstration. We would retain hard presynaptic sign constraints instead of adopting their soft E/I regularizer. [Primary manuscript](https://arxiv.org/html/2509.15460v1).

The feature-map lattice is only loosely analogous to retinotopy. Our multiscale encoder has overlapping receptive fields; its 169 positions are not 169 independent image patches or memory slots. Its channels are learned coordinates, not literal cortical columns. Exact weight sharing and homogeneous 3 x 3 connectivity are engineering approximations; cortex also has heterogeneous and longer-range connections. A single recurrent update propagates information one grid hop, so local communication can take extra time and interfere or blur nearby memories. More recurrent steps are more computation as well as more delay. No architecture label guarantees stable storage or accurate comparison.

## Analytical resource estimate

These counts exclude the shared sensory network and final comparison/readout, and are not runtime measurements. Input projection includes the current compression path in the dense model.

| Quantity | Current vector E/I | Proposed spatial E/I |
|---|---:|---:|
| Rate entries per example | 256 | 10,816 |
| Rate + adaptation entries | 512 | 21,632 |
| Rate + adaptation fp32 bytes | 2,048 (2 KiB) | 86,528 (84.5 KiB) |
| Recurrent kernel weights | 65,536 | 36,864 |
| Input projection + core MACs/frame | 357,888 | 6,922,240 |

The proposed input/core has 41,216 parameters with a bias-free input projection, recurrent bias, and three learned channel-wise timescale/adaptation vectors. Convolution reduces distinct recurrent weights but applies them at 169 positions. Explicit memory state is 42.25 times larger, and input/core MACs are about 19.3 times larger. Neither ratio is an end-to-end speed estimate. Sensory traces are additional persistent state in both models. Inference state size and per-frame computation remain constant with sequence length; total computation scales with the number of frames, while full-BPTT training must also store or recompute earlier activations.

## Recommended next experiment

Keep the trained dense model as a reference. First evaluate a learned old-memory/current-probe comparator on the existing orientation task; that directly follows the diagnostic result. Then test ONE spatial E/I candidate on a small spatial-binding extension, using the same functional comparator arrangement so the spatial model is not uniquely given a better comparison interface.

Show two spatially separated Gabors, insert blank frames, and then show a comparison pair. On one half the two orientations remain at their original positions; on the other half they swap. Counterbalance A-left/B-right versus B-left/A-right so the orientation inventory and probe alone cannot predict the answer. Independently vary nuisance phase/noise so literal raster equality is not the task. Include zero delay to measure perception/comparison acquisition and a trained nonzero delay to measure delayed performance. Vary locations during training and reserve fresh locations for a separate generalization test. This tests which orientation was where without adding a selection cue or attention module.

Retain the centered single-item orientation task as an anchor. If basic binding works, later vary item count and spacing to measure capacity/interference; those are follow-up questions, not a precommitted sweep. A spatial-vs-vector result also changes state size and parameter sharing, so it would establish practical benefit of the design package rather than isolate locality causally. The spatial core cannot inherit the dense core's incompatible weights/optimizer tensors; reuse compatible sensory weights and explicitly initialize new components. New stimuli must be trained, not presented only at evaluation with an expectation of spontaneous generalization.

Recommendation: spatial memory is a justified next capability to investigate. Couple it to an explicit comparison interface, and do not interpret it as an already established cure for the current single-item error.
