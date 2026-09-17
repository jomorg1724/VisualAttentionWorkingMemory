# Identity-initialized ConvNeXt output additions

Implemented 2026-09-12 following the user's authorization to compare two hybrids with continued ConvNeXt. Mathematical motivation, primary literature and limitations remain in [component_combination_research.md](component_combination_research.md). This document describes implementation and the focused CPU check; training results belong to the experiment report.

`hybrid_models.py` exports `build_hybrid(name)`, `HYBRID_NAMES`, `NEW_PARAMETER_PREFIXES` and `NEW_STATE_PREFIXES`. Both subclasses inherit the original `SpatialEncoder` with its `frontend` and `stages` unchanged. Consequently existing `encoder.stages.*` and all `decoder.*` names survive, supporting parent weight and named optimizer-state transfer. New optimizer state is the trainer's responsibility. The factory constructs additions at identity; it does not load a checkpoint automatically.

| Name | Designated returned field | New trainable parameters | Encoder total |
|---|---|---:|---:|
| `convnext_gabor_residual` | 24 channels at 50×50 | 1,296 | 263,328 |
| `convnext_se_residual` | 96 channels at 13×13 | 4,728 | 266,760 |

The Gabor branch reuses the current deterministic chromatic bank unchanged: 24 rectified simple responses plus 24 quadrature energies, frequencies 0.12 and 0.25 cycles/pixel, four orientations and three color axes. The duplicate raw-RGB channels are excluded from this branch; RGB still enters the complete original base encoder. Binomial blur and factor-two decimation precede per-location channel LayerNorm and a learned 48→24 pointwise projection. A learned 24-channel scale starts at zero while the projection starts nonzero:

\[
\widetilde H_1=H_1+\alpha\odot P\bigl(\mathrm{LN}_{C}(\mathrm{BlurDown}(G(x)_{:48}))\bigr).
\]

Only the returned first field changes. Both deeper fields are computed through the original unmodified path. Stored additional fixed coefficients total 12,096: 11,664 Gabor coefficients and 432 blur coefficients. No stochastic neural noise is added. The previously derived approximately120.6 million additional MAC/frame is an analytical operation count, not measured timing.

The separate SE-style variant applies a 96→24→96 pointwise MLP to the final field's channel means:

\[
\widetilde H_3=H_3\odot[1+0.5\tanh(W_2\operatorname{ReLU}(W_1\operatorname{mean}_{h,w}(H_3)+b_1)+b_2)].
\]

The last weights and biases start at zero, giving initial gain exactly one. Gains remain between0.5 and1.5. Earlier returned fields are unchanged. This is the documented centered SE-inspired adaptation; it does not copy MobileNet weights or its literal gate.

The existing ordered decoder, two-frame shared weights, three spatial resolutions, original model file and pinned earlier experiment sources are untouched. Initial identity preserves the trained parent's function; subsequent updates can still change motion and other task performance. These additions investigate useful engineering components, without attributing the previous Mobile/VOne advantage to a single mechanism.

Focused verification used `python -m unittest PreAttentiveVision.test_hybrid_models -v` on CPU with two threads and the actual ConvNeXt step756 parent checkpoint. One test passed in1.297seconds. It checked exact parent feature equality and all seven task logits at initialization, compatible state-key loading, parameter counts, and exact equality of non-designated fields after opening each addition. At identity, the Gabor scale and SE final weights received finite nonzero gradients. After opening them, the Gabor projection and SE first weights also received finite nonzero gradients. Zero downstream gradients at the exact identity initialization are expected and are not treated as a defect. No GPU execution or training occurred in this implementation check.
