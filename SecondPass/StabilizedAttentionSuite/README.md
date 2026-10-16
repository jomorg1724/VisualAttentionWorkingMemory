# Stabilized attention across the complete task suite

October 7 authorization: train the successful stabilized architecture jointly on the full existing suite, changing as little as possible. This tests joint learnability across tasks, not zero-shot transfer to untrained tasks.

## Model and training

The original stabilized Krauzlis backbone is imported unchanged: fixed local pixel correlations, RGB appearance pathway, one shared spatial-attention map, bounded normalized query/key attention, normalized spatial/CLS GRUs, attention-dependent memory writes, local temporal comparisons, and shared 64-unit readout FFN. Only its last binary linear layer is replaced by 13 task-specific linear layers. Task identity selects an output layer and does not enter attention or memory. There are 381,264 fresh trainable parameters in 80 tensors. No prior checkpoint is loaded.

All 13 tasks and 35 conditions use the existing TaskSuite renderers, native sequence lengths, stimuli, labels, difficulty ranges, and BSDS500 photo splits. New train/validation/test namespaces are 199101/199102/199103. The suite's BalancedScheduler gives each task one update per 13-update cycle and balances conditions within each task. Each update uses 32 fresh trials, microbatch 4, FP32 full BPTT, Adam 3e-4, terminal cross-entropy plus the unchanged 0.05 pixel-prediction objective, and global gradient-norm clipping at 100. There is no gradient renormalization. Fresh batch sampling replaces the single-task run's 1,000-movie/two-epoch replay schedule; this is an explicit training-adapter difference.

Short sensory trials retain their native two frames. The long-window motion comparison cannot contribute on those trials; the existing appearance and recurrent CLS pathway must solve them. No extra shortcut, extra processing steps, task-specific encoder, auxiliary label, or architecture rescue is added.

## Runtime and evaluation

One sequential local GPU run, with a single eight-hour cap beginning at profiling; no cloud, automatic extension, or second arm. Requested 17,160 updates (1,320 per task); profiling all 35 conditions prospectively reduces this to complete 13-update cycles if necessary. Disposable profile weights are discarded before fresh production. Actual target and deadline are in LocalRuntime/run01/profile.json, and actual activation requires persisted optimizer verification.

Full-suite validation at update 130, then every 520 and the terminal update: 64 trials per ordinary condition, 100 per Krauzlis condition, total 2,348. Additional 300-trial Krauzlis monitoring every 130 updates. Validation always reuses its separate fixed stream; it selects best weights by equal-task mean AUC, then equal-task mean chance-normalized balanced accuracy. Conditions contribute equally within tasks. Empty-list recognition conditions report specificity and are excluded from AUC/BA selection.

Final untouched test: 256 trials per ordinary condition and 1,000 per Krauzlis condition, total 11,192. Report every task, delay/load/hold condition, difficulty stratum, and Krauzlis target/foil/catch separately. One best and one latest checkpoint preserve model, optimizer, scheduler, native stream states and RNG. Original Krauzlis checkpoints and analysis are preserved. A prior model comparison must disclose architecture, training exposure and evaluation differences; historical claims are not automatically a matched control.

CPU checks passed exact original-vs-adapted Krauzlis logits and internal details, native stream replay, all 13 task updates, inactive head isolation, and all 80 finite optimizer states. These are implementation checks, not learning results.

Launch: `C:\Python310\python.exe -m SecondPass.StabilizedAttentionSuite.launch --run SecondPass/StabilizedAttentionSuite/LocalRuntime/run01`.

## Activated run01

All35 GPU profile conditions passed. Pinned13,702 updates:1,054 per task,438,464 fresh training presentations. Deadline October7,2026 at4:19:41PM PDT. Independent CPU readback confirmed all80 aligned finite Adam states at saved13, all13 heads trained, worker41092 live and advancing, source hashes correct, original checkpoints unchanged. See LocalRuntime/run01/independent_production_verified.json. No held-out result at activation.
