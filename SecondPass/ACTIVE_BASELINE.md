# Active architecture: original CNN + spatial KDA + global GRU

## User-directed rollback — 2026-09-29 UTC

The no-CLS recurrent-transformer experiment is cancelled. Return to the original
`WorkingMemory.PlainBaseline.accum.AccumulatorBaseline` with `stack=3`,
`center=True`, `accumulator='kda'`, hidden size 256 and native task heads.

Forward path: centered causal three-frame stack → four convolutional blocks,
with spatial KDA states at 25×25, 13×13 and 7×7 → flatten 160×7×7 → learned
256-dimensional feature → global 256-dimensional GRU → task head.
No final spatial ConvGRU, comparator addition, recurrent transformer, CLS token,
or pooled convolutional-map decoder belongs to this active reference.
The original implementation is intact; no destructive Git rollback is needed.

## Initialization rule

**Every new run starts entirely from scratch unless the user specifically
permits inherited weights.** Architecture reuse, a request to revert, and a
request to train do not authorize checkpoint migration. Do not load any model,
optimizer, stream, RNG or selection state by default. Do not freeze a pretrained
encoder. Existing historical continuation permissions do not authorize a new run.

The existing fresh-only constructor is
`SecondPass.JointTraining.worker.fresh_model(seed, device)`; it constructs the
whole model, all parameters trainable, and empty Adam at a single LR of 1e-4.
No training, profiling, checkpoint loading or cloud restart is authorized by
this rollback. No trained checkpoint has been installed as a new parent.

## Why this architecture qualifies

- `WorkingMemory/PlainBaseline/runs/local_kda_program_20260917/program_receipt.json`
  records the original KDA ring stage with `init: null` and terminal ring-D0
  BA/AUC 1.0. Later cued/delay curriculum stages inherited the earlier stage;
  they are **not independent from-scratch task acquisitions**.
- `SecondPass/JointTraining/worker.py:33` constructs this same architecture
  freshly for the complete native 13-task / 35-condition suite. Its original
  production seed is 94182763; no older checkpoint was loaded.
- `SecondPass/JointTraining/RESULTS.md` preserves measured joint-suite results,
  including sensory acquisition after same-architecture continuation. This is
  not a claim that all spatial/memory tasks were solved.
- `SecondPass/SpatialReadout/protocol.py` explicitly labels the later final
  ConvGRU branch `kda_final_convgru_warm_start_v1`. That branch, the comparison
  addition and the CLS transformer inherited weights and are excluded as
  evidence of from-scratch acquisition by those changed architectures.

This restores the last verified working **architecture**, not a retrospectively
selected best checkpoint or a claim of superiority on every task. Preserve all
historical source, checkpoints and results as separately labelled evidence.

## Cancelled cloud run

RunPod `7ij62e571pln8w` was stopped at the user's request and independently read
back as `EXITED`. Its launchd mirror was removed; no matching mirror process
remains. Local checkpoint 2899 was SHA-256 verified after cancellation in
`/Users/jonathanmorgan/VAWMRuntime/cloud_convdecoder_fresh_02/artifacts/`.
This is the last locally verified checkpoint, not a claimed terminal step.
The stopped disk is retained to avoid destroying unmirrored artifacts and
continues to incur storage charges. No GPU restart or deletion was performed.
No final-test completion or architecture-incapacity conclusion is claimed.
