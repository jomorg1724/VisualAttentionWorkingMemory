# AV-context v2 — five-change scratch arm, local run

Base: `attention_context_comparator_scratch_v1`
([model](../model.py)) → `attention_context_comparator_v2` ([model.py](model.py)).
Same five-task battery, same stimuli, losses, cues, seeds and batch recipe as
the v1 scratch arm; 12,000 updates / 480,000 episodes; validation every 800
updates; a pre-registered stopping rule at 2,400.

All v1 source files stay byte-identical. v2 is a subclass layer so that the
two arms still training on the pinned v1 hashes are unaffected and so that the
identity gate below can compare v2 against the real v1 code.

## The five changes

| | Change | Where | Identity at init |
|---|---|---|---|
| A | `energy_channels` also returns the raw opponent (4) and total magnitude (1); accumulator features 72 → 77; `output = Conv2d(77,32,1)` with the five new columns zero | `OpponentAccumulatorV2`, `_emit` | exact |
| B | per-scale `cat(avg_pool, max_pool)` to 13×13 (32 → 64 per scale); `readout.fusion = ConvNormAct(192,64)`, max-half columns zero | `_sensory` | exact |
| C | head 1: λ = 0.02 (σ = 5 cells; penalty 0.18 at 3 cells, 0.72 at 6) and source-neutral `[0,0]`; head 0 unchanged (λ = 4, `[2,0]`) | `raw_locality`, `source_bias` | changes output |
| D | priority selection maps `xavier_uniform(gain=.5)`, zero bias, seed `model_seed+200` | `priority_readout.selection` | changes output |
| E | one learning rate: recipe sets `parent_lr = new_lr = 3e-4`; v1 trained 390,152 of 534,239 parameters at 3e-5 | [protocol.py](protocol.py) | optimizer only |

The magnitude in A is `(positive + negative).mean(channels, pairs)`, one map
`[B,1,H,W]`, which is what the spec's shape comment states (a `keepdim` mean
over channels alone would be `[B,1,4,H,W]` and could not be concatenated).

Parameters: 534,239 → 590,015 (+480 from A, +55,296 from B), all at 3e-4.

## Gates and pre-flight (all forward-only, no training)

[check.py](check.py) → [construction_checks.json](construction_checks.json):
v2 with C/D/E disabled is **bit-identical** to v1 on CPU on fixed motion,
orientation and binding batches (max |Δlogit| = 0). On CUDA the same
comparison differs by 3.1e-5 because cuDNN selects different kernels for
77/192-channel convolutions; the CPU equality is the hard gate. Gradients
reach every new zero column, all five selection maps, `raw_locality` and
`source_bias`.

[preflight_attention.py](preflight_attention.py) →
[preflight/preflight_attention.json](preflight/preflight_attention.json), on
the hash-verified local scratch motion-only v1 step-3200 checkpoint
(`a76157…5447`), 64 cued motion D0 validation trials, every timestep:

| model | λ by head | mass beyond 3 cells by head | patch-query → cue-key mass (visual bank) |
|---|---|---|---|
| v1 step 3200 | 4.023, 4.027 | 1.8e-17, 1.7e-17 | 2.0%, 2.0% (adjacent-cell leakage only) |
| v2 init | 4.0, 0.02 | 2.3e-17, 0.716 | 2.0%, 6.5% |

v1 keeps 83% of each query's mass on its own cell and 5.6% on neighbours; the
2.5–3.5-cell bin holds 2.5e-14. The locality argument is now a measurement.

[preflight_probe.py](preflight_probe.py) →
[preflight/preflight_probe.json](preflight/preflight_probe.json), same
checkpoint, 768 fit / 256 test cued motion D0 trials, standardized multinomial
logistic probes with C chosen by 3-fold CV. Every probe is at chance: H_T, R_T,
C_T and last-evidence-frame H_9, as full 10,816-d fields, 128-d mean+max pools
and 576-d cued 3×3 regions. Best is `C_T_cued_3x3` at 28.1% (Wilson 95%
23.0–33.9%); its permuted-label control scores 25.8%. Under the spec's rule
this means the field is not decodable, so A and B stay in and the launched arm
is the full A–E combination.

## Run layout

`runs/v2_local_<timestamp>/` holds `aggregate.json`, `budget.json`,
`live_status.json`, `fixed_config.json`, `profile_result.json`,
`training/metrics.csv`, `training/live_diagnostics.jsonl` (every 32 updates),
`training/checkpoint_*.pt` (every 256 updates and at each validation target),
`validation_<step>/summary.json` and, on completion, `selected_test/` and
`terminal_test/`. `gate_receipt.json` records the step-2,400 verdict.

Live diagnostics per task: priority entropy as a fraction of ln 169, attention
mass beyond 3 cells per head, spatial variance of the evidence maps and of H_T,
softplus(raw_locality) per head, source bias, selection weight norm and
priority/attention gradient norms. Validation summaries carry the same
quantities measured on the validation trials.

Monitor:

```bash
python WorkingMemory/AttentionContextComparator/V2/status.py
```

## Pre-registered stopping rule

At validation 2,400 (96,000 episodes) both must hold, measured on the
validation trials: priority entropy below 98% of ln 169 on at least one of
`orientation_cued`, `motion_duration_cued`, `krauzlis_cued_motion`; and head-1
attention mass beyond 3 cells above 1%. If either fails the supervisor stops
with status `stopped_by_preregistered_gate`, no held-out test and no extension.
For reference the v1 cloud arm's validation-3200 entropy fractions were
99.1% / 99.6% / 99.9% on those tasks.

## Cloud overnight arm

`cloud/launch.py --hours H --updates N` builds a hash-pinned bundle
(`cloud/prepare_bundle.py`), creates one RunPod community pod (RTX 3090, else
4090), uploads bundle, BSDS500 payload, config and `cloud/setup_remote.sh`,
starts `cloud_sweep.py` remotely and a detached local
`cloud/lifecycle_monitor.py`, which runs `cloud/watch_remote.py` every 45 s,
mirrors validation/test artifacts under `runs/cloud_<ts>/incremental/`, and at
completion, failure or deadline retrieves everything (checkpoints included),
verifies hashes, then stops and deletes the pod. The remote supervisor stops
training 45 minutes before the deadline and validates/tests the latest
checkpoint; the 2,400 gate is recorded but does not stop this arm.

No pod is running. The overnight pod `vqpgk21cpi53b6` was user-stopped at
step 10,621 and deleted after verified retrieval; its complete results and 49
checkpoints are in `runs/cloud_20260915_222845/retrieved_user_stop/` (an
earlier mid-run pull through 10240 is in `pulled/`).
Run the launcher and lifecycle from PowerShell (native OpenSSH); Git's MSYS
`ssh` rewrites backslashes in remote arguments.

To stop a pod by hand with retrieval first:

```bash
python WorkingMemory/cloud_shutdown.py WorkingMemory/AttentionContextComparator/V2/runs/cloud_20260915_222845/cloud_provisioning.json
```

It refuses to delete when retrieval fails unless `--force` is added.

## Attribution

Five changes in one arm: a positive result shows the combination works, not
which part did it. Leave-one-out ablations are deferred until a positive
result exists. Change E is the non-architectural change and touches 73% of
the parameters; it is named here rather than folded in.
