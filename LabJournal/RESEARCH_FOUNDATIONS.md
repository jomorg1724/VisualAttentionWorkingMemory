# Research foundations and what was actually adopted

[Index](README.md) · [Architecture](ARCHITECTURE.md)

This page organizes the existing repository's literature rationale. It does not claim a new literature search or independently revalidate every paper. Follow the linked research files for primary-source references, mathematical choices and disclosed adaptations.

| Research area | Adopted computational idea | Where documented | What is not established |
|---|---|---|---|
| Guided Search6.0 | Separate early vision, selection, memory and later integration as functional components | [PAV research](../PreAttentiveVision/research.md), [root scope](../README.md) | The complete GS6 system has not been implemented; treating activated LTM as weights is the user's approximation |
| Modern convolutional networks | Residual local processing, depthwise spatial kernels, channel mixing/normalization, multi-kernel branches and SE modulation | [Encoder rationale](../PreAttentiveVision/research.md) | Tiny100-pixel adaptations are not reproductions of published benchmark results |
| Krauzlis-style random-dot neuroscience | Controlled cardinal mean motion with noise/coherence adaptations | [Exact stimulus sources](../PreAttentiveVision/krauzlis_stimulus.md) | Two-frame100-pixel task does not reproduce every psychophysical/neural condition |
| Motion-energy computations | Quadrature spatial filters combined with unequal temporal traces and opponent energies | [Temporal design](../PreAttentiveVision/TemporalIntegration/README.md) | A useful motion-energy-inspired module is not a fitted V1/MT population model |
| Recurrent WM and E/I dynamics | State-dependent recurrent updates, signed presynaptic influence, leaky firing rates and adaptation | [Recurrent research](../WorkingMemory/Research/recurrent_memory_without_attention.md) | Dale-like signs alone do not validate biological implementation; bounded leak does not ensure full-loop stability |
| Spatial mnemonic representations and lateral recurrence | Preserve feature-location fields and local convolutional interaction | [Spatial rationale](../WorkingMemory/Research/spatial_ei_memory.md) | Shared convolutional weights are an engineering assumption; high swap accuracy does not establish item capacity |
| Selection and maintenance | Let remembered content influence which sensory/memory information drives the next update | [Attention implementation](../WorkingMemory/PreUpdateAttention/README.md) | Literal dot-product attention is not established as the brain's algorithm |
| Multi-task optimization and residual paths | Test allocation before architecture; preserve useful short paths when motivated | [Allocation/residual design](../WorkingMemory/Research/training_exposure_and_temporal_residual.md) | Allocation benefits do not uniquely prove gradient conflict; a residual could help for several reasons |

The existing research discusses modern ML precedents such as xLSTM and gated recurrent/associative systems alongside neuroscience on persistent activity, adaptation, practice-dependent representations, and selecting remembered information for use. These motivate mechanisms to consider, not a requirement to clone a paper wholesale. The implemented normalized LSTM is not an xLSTM reproduction. Later attention authorization supersedes earlier design-stage “no attention yet” restrictions only for the specified experiments.

The project's strongest claims are presently computational and behavioral within its generated tasks: a particular schedule improves contour; particular frozen states support useful readouts; a particular trained attention path is required during blanks. Biological correspondence would require additional predictions and comparisons to neural/behavioral data. Those were not part of the reported training runs.
