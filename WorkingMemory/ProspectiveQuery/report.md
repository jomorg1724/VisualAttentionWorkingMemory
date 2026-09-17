# Prospective sensory-conditioned query experiment

**Final status:** user-stopped partial run. Training ended at logged step 9430
/ 41,200 fresh episodes; the latest durable checkpoint is step 9216 / 32,640
episodes. All 29 indexed artifacts were retrieved and verified. RunPod pod
`dqi13o2x3qkvos` was stopped, deleted and confirmed absent; account spend is
$0/hour. See the [completion report](completion_report.md), [completion
receipt](runs/prospective_20260914_175602/completion_receipt.json) and [cleanup
receipt](runs/prospective_20260914_175602/cleanup_receipt.json).

## Question and controlled intervention

The experiment asks whether attention routing improves when pre-update queries can use current sensory/cue information prospectively instead of depending only on prior memory:

`Q_t = W_Q(LN(R_{t-1}) + P + e_m + gamma * LN(H_t))`.

`gamma` is the sole new architectural parameter: one trainable scalar initialized to exactly zero. The implementation reuses the existing query normalization. At initialization, this produces the original query and output exactly while allowing gradient-based adoption of the sensory residual. Keys, values, inherited Q/K/V/O tensors, position/source embeddings, learned source and locality terms, spatial E/I recurrence, comparator, pooling/readout, stimuli, task definitions, losses, optimizer hyperparameters, seeds, training exposure and evaluation protocol are unchanged.

The parent is the original intact attention8400 checkpoint, SHA256 `e37602aa20ccfc400ea8fe9d98c11f29c508069388897c55803b97f2ccdf1bc9`. Migration explicitly rejects the wrong hash, wrong version/step, and already migrated checkpoints. It retains all 138 parent tensors and 123 compatible Adam states. The only new tensors are `attention.gamma` and the same five deterministic semantic heads used by the earlier original-bias five-task migration.

## Construction evidence

[`construction_checks.json`](construction_checks.json) records:

- `gamma=0` maximum absolute differences of 0.0 for query tokens, attention output and inherited full-model output;
- nonzero gamma gradient at zero, 1.9900609;
- sensory perturbation changing queries by 0.1256902 and attention routing by 0.0290360 maximum absolute difference;
- unchanged source/locality terms and fixed 4,000-update / 160,000-episode protocol;
- rejection of remigration and incorrect parent hashes.

The pinned runtime source hashes are in the remote manifest and [`launch_receipt.json`](launch_receipt.json). The immutable source bundle SHA256 is `a747feef2097a25217f068d60444a05b8fe6112df5d5e113e2be077282a9afa9`; the remote manifest SHA256 is `3e64d82d9ad64078627ebd4de39e2fd61b3e8b2f036615012b8cff6e6ad1f4e0`; and the exact train job SHA256 is `803b27a3cc996d03762e3092a7e12112831f46b1cd5e393525722de57eee436d`.

## Fixed training and comparison boundary

Each of 4,000 updates presents one batch of eight examples from each of the five existing tasks, averages the five cross-entropies, clips once and takes one Adam step: 160,000 fresh episodes. Validation is scheduled at steps 9200, 10000, 10800, 11600 and 12400 with the prior selection rule.

The historical original-bias control provides matched validation points only through 11600 because it was user-stopped before step 12400 and final held-out evaluation. Therefore only 9200/10000/10800/11600 are controlled historical comparisons. Prospective step-12400 and final held-out results will be unmatched exploratory endpoints unless a matched control is later authorized.

## Cloud execution evidence

The fresh secure-cloud pod uses one A100-SXM4-80GB (80 GiB, driver 580.126.16) at $1.59/hour. The pinned environment is Python 3.10.12, PyTorch 1.13.1+cu117, NumPy 1.23.1, SciPy 1.8.1 and Pillow 9.1.1. A100 was preferred over H100 for compatibility with the inherited CUDA 11.7 runtime.

Two setup defects were corrected without changing the scientific run or renewing its deadline. First, the Windows CRLF setup script was normalized remotely. Second, profiling exposed that the source archive did not contain the existing BSDS500 fixtures required by the unchanged recognition task. No training had started. The exact 500-image local fixture set was transferred in a separate payload (SHA256 `990e2bd405f6bcc4c940299f437685fc08765dcb56923ab0ab639f5f37c1883e`), its manifest and file count were verified, and the same source-hash ledger and original deadline were resumed. Profiling then completed successfully.

At the launch verification snapshot, GPU utilization was 59% with 2,099 MiB allocated, the worker had written metrics through step 8419, `gamma` had moved from 0 to 0.00198899, and the hash-indexed initial training checkpoint at step 8400 had SHA256 `54189b0e27a9baa0080b72884de6fa47e9d027994b4fa6d47ddf2aa1b073de16`. The bounded watcher incrementally retrieves only hash-indexed checkpoints and will verify the complete final artifact index before cleanup. Automatic stop at 08:56:34 UTC and termination at 10:56:34 UTC are billing backstops; successful retrieval still requires explicit pod deletion.

## Completed analysis-only diagnostics

The local diagnostics completed against frozen original-bias checkpoint10000 (`35281f…942`) with its hash unchanged. [Full paired rows](runs/prospective_20260914_175602/diagnostic_results.json) and a [compact interpretation receipt](runs/prospective_20260914_175602/diagnostic_summary.json) are preserved.

Across 32 independent paired movies, changing only the valid cue rings changed the frozen model's predicted class in 15 cases: 46.88%, Wilson 95% interval 30.87–63.55%. Mean absolute attention-routing change was 0.0003545, with a maximum element change of 0.6364. This establishes that the frozen baseline is sensitive to cue identity at routing/output. It does **not** show successful use of the duration rule: original/counterfactual accuracies were 21.88%/34.38%, and mean correct-class probabilities were 0.2494/0.2492, near four-class chance.

The split linear probe used 256/64/128 independent train/validation/test base episodes, four patch examples each, train-only standardization and validation-only ridge selection. Held-out accuracy was 25.98% overall, 23.44% at target patches and 26.82% at foils versus 25% chance. This gives no evidence that the chosen local 3×3 final-moving-frame sensory summary linearly exposes longest-duration direction. It does not establish absence from other sensory times, spatial summaries or nonlinear representations, and signal availability would not by itself establish causal model use.

## Live validation snapshot: step 9200

The first planned validation completed in 56 seconds and training continued as
`train_10000`. Its 1,836 predictions match the historical control's ordered
examples exactly on task, condition, label, paired/trial IDs and metadata.
All six remote artifacts were mirrored with matching SHA256 values; see the
[mirror receipt](runs/prospective_20260914_175602/validation_9200_mirror_receipt.json)
and [full condition table](runs/prospective_20260914_175602/validation_9200_report.md).

Prospective versus control task-level normalized BA / mean AUC is:
recognition 0.2014/0.6746 versus 0.2812/0.7205; Krauzlis
0.0000/0.4964 versus 0.0000/0.5172; motion duration
-0.0260/0.4946 versus 0.0000/0.4955; orientation 0.0703/0.5603
versus -0.1016/0.4875; and binding 0.8047/0.9849 versus
0.4844/0.8723. The selection rank improves from
`[-0.1016, 0.6186]` to `[-0.0260, 0.6422]`.

This first look favors prospective queries on orientation and binding, but
recognition is mixed and weaker at load 24, while both motion tasks remain near
chance. It is a live validation result, not selected or final held-out
performance.
