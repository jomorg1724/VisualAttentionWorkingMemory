# Spatial priority-map readout

**Live status:** version `spatial_priority_readout_scratch_v2` is training from
scratch on Palladio RunPod `ce00y2ooosl7wc` (community RTX 3090, 24 GB, listed
$0.22/hour). The absolute deadline is
`2026-09-15T11:47:20.740518Z`. Metrics here are launch-health snapshots, not
validation results.

## Lineage correction

The first launch incorrectly inherited attention8400. When the user corrected
the lineage, local monitor PIDs 30480/38436 were stopped and pod
`mbi39b005jp884` was stopped (HTTP 200), deleted (HTTP 204), and confirmed
absent (HTTP 404). It had logged 320 updates / 12,800 fresh episodes and no
validation. Its artifacts remain under `runs/priority_20260915_025605/` as a
cancelled attempt and are not scientific results.

The replacement has no parent checkpoint in its source bundle, loads zero
model tensors and zero Adam states, begins at step 0, and initializes all
579,423 trainable parameters from documented seeds. The remote seed-repeat
check passed. The initial remote tensor-state digest is
`9e2a507a4bc9920eb5ca7d0c288943a56867579eeaffc15eda0886734d249a45`;
checkpoint 0 is
`97b7e6ce513592fc094724854ea1d4d329c93fd1ae0d86aa20ed50e4fd653f76`.
Local and pinned-cloud tensor-byte digests differ and are recorded separately;
determinism is established within each runtime, not claimed bitwise across
PyTorch/platform versions.

## Architecture

The full model freshly constructs the existing opponent encoder and temporal
emission, original `JointAttention`, source/locality biases, spatial E/I
memory, and comparator. Prospective-query `gamma` is absent.

The terminal decision path contains no global mean/max pooling:

1. concatenate final `H_T`, updated `R_T`, and comparator `C_T` into
   `[B,192,13,13]`;
2. shared 1×1 convolution `192→96`, GroupNorm/SiLU;
3. padded 3×3 convolution `96→64`, GroupNorm/SiLU, without downsampling;
4. task-specific 1×1 selection logits and local class-evidence maps;
5. 169-location spatial softmax and weighted local-evidence sum into logits.

Priority maps are captured with evaluation predictions. Task identity only
selects the task head; no cue label, coordinate, phase oracle or global
classifier bypass is introduced.

## Fixed training and evidence

The historical five-task stimuli, cues, labels, losses, train/evaluation
seeds and condition schedule are unchanged. Each update averages five
task-specific microbatches of eight before one clipped Adam step. The target
is 4,000 updates / 160,000 episodes with validation at
800/1600/2400/3200/4000 and unchanged held-out selection.

Three RTX 3090 profile updates averaged 3.4266 seconds, with 1.014 GB peak
allocated and 1.122 GB reserved. The conservative training plus
evaluation/retrieval estimate is 5.51 hours / $1.21; the eight-hour listed
ceiling is $1.76.

- [`construction_checks.json`](construction_checks.json): deterministic
  scratch initialization, empty optimizer, component/invariant, map and
  gradient tests.
- [`lineage_correction_receipt.json`](lineage_correction_receipt.json):
  cancelled-run cleanup and replacement lineage.
- [`launch_receipt.json`](launch_receipt.json): verified cloud launch,
  hashes, profiling and initial metrics.
- [`runs/scratch_20260915_034720/live_status.json`](runs/scratch_20260915_034720/live_status.json):
  roughly 45-second metrics/GPU/checkpoint mirror.
- [`report.md`](report.md): scientific and operational live report.

Detached lifecycle PID 36128 and watcher PID 36812 own mirroring, final
hash-verified retrieval, stop and deletion. Historical comparisons have
different initialization and therefore are not initialization-controlled
ablations.

## Complementary local frozen-core diagnostic

The authorized [LocalDiagnostic](LocalDiagnostic/README.md) completed locally
without querying, modifying or interrupting the separate scratch cloud run. It
froze the motion-only step-12200 encoder, opponent traces, original attention,
spatial E/I memory and comparator, then cached identical final `H_T/R_T/C_T`
tensors for two post-hoc output probes.

On 256 independent held-out base movies presented at each D0/4/12/24, a
74,303-parameter pooled probe scored 23.83% balanced accuracy /0.4805 macro
OVR AUC, while the 74,373-parameter spatial-priority probe scored 24.32% /
0.4832. The paired BA difference was +0.49 percentage points (95% grouped
bootstrap CI -1.76 to +2.64). Neither probe exceeded the 25% task chance
level, and no delay showed a reliable spatial advantage.

The learned map nevertheless put 13.13% of mass in an evaluation-only 3×3
target region versus 5.33% under a uniform map; this target alignment did not
yield direction decoding and is not causal attention evidence. See the
[final report](LocalDiagnostic/report.md) and
[immutable run artifacts](LocalDiagnostic/runs/diagnostic_20260915_0350/).
