# Three-layer whole-sequence patch KDA

The user requested three KDA layers and a new fresh GPU-pod run. This version preserves the original single-layer experiment in `SecondPass/SequenceKDA`; it inherits its architecture conventions and verified FP32 backend equations, never trained weights or optimizer state.

Native `[B,T,3,100,100]` movies become 100 nonoverlapping 10-by-10 RGB patches per frame. Each centered 300-value patch passes through a shared Linear300→128, GELU and LayerNorm. Learned row/column embeddings and fixed frame-index sine/cosine features are added. Tokens enter in chronological frame order, then row/column order; one learned CLS token with time index T is appended last.

Exactly **three independently parameterized pre-LayerNorm residual KDA blocks** follow. Every block has width 128, two heads with key/value width 64, dense q/k/v/decay projections, a two-head write projection and an output projection. The first two blocks return every token representation. The third block computes its final CLS representation, then the existing final LayerNorm and Linear128→256/GELU/Linear256→2 classifier produce the decision. There are no extra feedforward blocks or memory modules.

The constructed model has **324,488 parameters in 55 trainable tensors**. CLS queries in layers two and three depend on representations produced by preceding layers, giving the final read a contextualized query. Causal KDA ordering, zero per-trial state and full temporal gradients remain unchanged.

Each head in every layer initializes decay-only half-life to 32 or 128 frames, accounting for 100 patch updates per frame; initial write strength is 0.01. Gates remain learned. All parameters and arithmetic are FP32, with TF32 and autocast disabled. The existing chunk32 triangular-solve backend is reused; it carries state and gradients through all tokens. See [BACKEND.md](BACKEND.md).

Training preserves native Krauzlis B12/B20/B28, 29/37/45 frames, 26/28-degree changes and 57 target / 29 foil / 14 catch draws per 100. One fresh whole model, Adam 1e-4, zero counters and independent fresh train/validation/test streams; batch 32 with microbatch 4, full BPTT and no clipping. The target is 4,216 updates / 134,912 episodes, reduced before production if native profiling requires it. Two validation looks use 100 trials per condition and select by mean AUC, then BA, then earlier checkpoint. Selected and terminal models receive the same fresh 200 trials per condition, with target/foil/catch results reported separately.

Parent owns the new single-pod eight-hour/$5 cap, setup/profile/evaluation/retrieval allocation, absolute stop guard, off-pod status and mirror. Production launch requires persisted optimizer progress. No state from the disposable profile may enter production. These source files alone provide no launch evidence.

Focused CPU engineering checks exercise the actual three-block stack, full parameter gradients and updates, early-frame gradient, trial reset, layer output shapes and short-stack reference parity. Native-length costs and feasible exposure are measured on the pod; there is no repeated exhaustive local backend campaign.
