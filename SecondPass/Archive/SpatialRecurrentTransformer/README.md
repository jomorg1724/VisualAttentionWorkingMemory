# Spatial recurrent convolutional transformer, recurrent CLS — v1

One warm-start architecture, not a second comparison arm. CNN + three spatial KDAs and the existing 160→64 spatial projection are retained by name/shape. Two pre-norm 64-channel, two-head transformer blocks replace both ConvGRU and comparator. At each frame, 50 memory queries (49 spatial plus persistent CLS) attend jointly to 49 current visual and 50 memory keys/values. Spatial Q/K/V/output projections and spatial FFNs are convolutional; CLS projections/FFN are linear. Learned position/source embeddings are features, not logit biases. The H residual is carried through attention and FFN. Both new spatial memory and CLS are recurrent across frames, reset at trial boundaries. Only final CLS→learned64→256→ReLU→inherited task head. No flatten-field readout, raw sensory bypass, metadata input, EI/GRU, phase gate, locality/source-logit bias or dropout.

Initialization: PyTorch default conv/linear parameters; learned spatial/CLS initial state, position and source embeddings normal(0,.02); LayerNorm default scale1/offset0. Full BPTT, no detach. All weights trainable, one unchanged Adam1e-4 (betas .9/.999, eps1e-8, weight_decay0), fp32, no clipping or TF32.

## Source and state

Production source is **completed comparison terminal**, not comparison validation winner and never early117:
- SHA256 `e47031f3c66fefdb174384ad442688554966f3ce2cc86a9562e2ec99640ae72c`.
- Actual checkpoint branch2600 + parent6760 = cumulative9360; inherited global scheduler12753.
- Compatible CNN/KDA/spatial_input/heads weights and named Adam moments copied exactly. Old spatial_gru, comparator and flatten readout removed; `memory.*` and `cls_readout.*` explicitly fresh, including fresh Adam state.
- Native train streams, scheduler/condition queues and CPU/NumPy/Python RNG preserved. New-architecture CUDA transition is explicitly freshly seeded; old device RNG retained as provenance. Current CUDA state checkpointed/replayed thereafter.
- Actual Session first snapshot is compared with migration input (including Torch2.8 `decoupled_weight_decay=False` adapter). Fresh Session next draw is checked, not merely two migration dictionaries.

## Exposure, selection and budget

Fixed **2600 new updates / 83200 episodes**, batch32/micro4, unchanged13 tasks/35 conditions and official BSDS source identity splits. This adds200 shared-task updates /6400 episodes per task; condition allocation comes from the inherited queues. One bounded13-task profile plus all35 validation-cell timings on this architecture; all profile state discarded. No stale model timings or same-model baseline reuse. Allocation fails closed if2600 does not fit; never silently shrinks or extends. Conservative per-task cost uses its measured update inflated by its slowest/observed per-cell evaluation ratio; runtime first-cycle timing comparison is persisted.

Two validation looks, at1300 and2600, with native early-cap terminal fallback occupying an unused look only. No optional initial baseline. Fresh validation namespace96792763, final namespace96892763. Selection: eight orientation_cued/spatial_binding cells' mean AUC, then suite equal-task mean AUC, earlier ties, validation only. Reports retain all35 cells, N0 specificity/FPR and Krauzlis event strata. Validation64/cell (Krauzlis100); final128/cell (Krauzlis200), selected and terminal deduplicated only after identical model verification.

Parent-owned immutable budget is copied from recovered `cloud_comparison_03/artifacts/budget.json`, with unchanged hard deadline2026-09-28T21:46:03.902995Z, work cutoff10minutes earlier. No budget defaults, renewal, pod-control API or credentials exist in this adapter. Parent must enforce provider-side billing stop and $12 total ceiling separately; worker wall deadline is not proof of billing shutdown.

## Local checks and package

```bash
/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python -m pytest SecondPass/SpatialRecurrentTransformer/test_transformer.py -q
/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python -m SecondPass.SpatialRecurrentTransformer.bundle --output /Users/jonathanmorgan/VAWMRuntime/spatial_recurrent_transformer_01
```

Builder verifies actual terminal9360 digest, source hashes, all500 BSDS images and every archive member. No historical evaluation files/metrics, fixture6760 or credentials are included. Requirements pin Torch2.8; install using the pod's documented interpreter/environment. Preserve source versions; no edits to predecessor sources/checkpoints.

## Exact remote launch (parent only)

Extract archive using `tar --no-same-owner`. Set `DEPLOY` to the extracted directory containing `repo`, `assets`, `deployment_manifest.json` and requirements. Then:

```bash
cd "$DEPLOY/repo"
python -u -m SecondPass.SpatialRecurrentTransformer.worker verify-source "$DEPLOY/run" --source-pointer "$DEPLOY/assets/source_pointer.json"
python -u -m SecondPass.SpatialRecurrentTransformer.worker supervise "$DEPLOY/run" --source-pointer "$DEPLOY/assets/source_pointer.json" --budget "$DEPLOY/assets/budget.json"
```

Run the supervisor using the parent's already verified durable launcher and independent billing backstop. It profiles, pins allocation and trains sequentially, without another gate. It refuses an existing activation/production migration rather than silently restarting. Outputs include migration_integrity, source receipt, pinned allocation, per-update fsynced progress, all-new-parameter gradient/update evidence, immutable full checkpoints, latest pointer, persisted progress verification, full evaluation JSON, report and completion manifest. No cloud work is performed by CPU tests or package building.
