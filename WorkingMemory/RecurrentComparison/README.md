# Focused recurrent memory comparison

This user-authorized experiment compares one LSTM and one excitatory/inhibitory adaptive rate network after the same learned opponent sensory computation. It contains no attention, slots, key/query addressing, routing, sensory feedback, or interleaved architecture.

**Completed:** both models reached 5,000 updates/40,000 episodes and selected their final checkpoint. All training and evaluation workers exited successfully. The remote artifacts were retrieved and verified, and the paid pod was deleted at approximately04:07 UTC on September13. Read [the final report](report.md), [HTML evidence](report.html), and [completion receipt](completion_receipt.json).

**User-directed execution update:** after local LSTM training began, the user moved the E/I arm to the verified Palladio RunPod account. Local E/I production is excluded by `handoff_local.py`; the active LSTM worker was adopted without interruption. The remote arm runs on pod `mddhcqx9eed6u7` (RTX3090), from an exact, unfitted local E/I initializer and the same fixed40,000-episode protocol. Its paid three-hour cap ends at **06:34:40 UTC on September13,2026**, earlier than the original local deadline. Torch/NumPy/SciPy/Pillow versions match; Linux/GPU/Python patch differences are recorded, and bitwise training equivalence is not claimed. The researcher retrieves and verifies artifacts before terminating this ephemeral pod to end compute and storage billing.

The fixed run is `runs/recurrent_20260912_202158`. Its new 14,400-second allowance began with the first GPU profile at 20:22:00 PDT on September 12, 2026; its absolute deadline is 00:22:00 PDT on September 13. Earlier unused allowances are untouched. Profiling fixed **5,000 updates × 8 episodes = 40,000 fresh episodes per arm**, with validation after 1,250, 2,500, 3,750 and 5,000 updates. This exposure is a feasible acquisition target, not a sufficient-learning guarantee. Healthy training continues through the fixed endpoint regardless of early scores.

## What is shared

Both models warm-start every compatible learned and fixed sensory tensor from `WorkingMemory/runs/wm_20260912_181219/opponent/checkpoint_006860.pt`. This is an expanded, versioned warm start with a new optimizer. All 408,728 transferred learned parameters remain trainable at 3e-5; only motion/orientation task heads receive task losses, and unused heads are preserved. New parameters use 3e-4. Each episode starts with cleared explicit temporal state and contributes one final cross-entropy loss.

Every observed frame produces the current projected fields, updated fast/slow opponent traces, motion/change emissions, and the existing multiscale fused field `[B,64,13,13]`. A shared-initialized convolution reduces this to eight channels. Fixed spatial flattening followed by Linear(1352,128), LayerNorm and SiLU supplies the recurrent input. This preserves spatial ordering through the bottleneck without supplying privileged cue identities, labels, phases or clocks. It does not guarantee successful cue representation.

The existing 128-dimensional sensory feature receives a small nonzero Linear(256,128) residual from the new memory. Its initial weight standard deviation is 0.01/sqrt(256), bias zero. The input and residual projections start identically across arms. Only the opponent traces plus the two new state vectors persist; previous predictions are never stored.

## Memory variants and optimizer

- **LSTM:** 256 cell values and 256 outputs. Independent input, forget, output and proposal preactivation LayerNorms precede separate gate biases. The cell is not normalized. Initial forget biases correspond to nominal time constants 4–128 observed frames; input bias is −2. Recurrent matrices are separately orthogonal with gain 0.5. Total learned parameters: **1,011,872**, including **603,144** new parameters.
- **E/I adaptive:** 256 nonnegative rates and 256 adaptation values. The 205 excitatory/51 inhibitory presynaptic signs constrain recurrent matrix columns throughout training. Positive magnitudes use inverse-softplus initialization, with each row initially summing to 0.6 excitatory and 0.6 inhibitory magnitude. Rate time constants start at 2–8 frames and remain bounded in 1–32; adaptation starts at 16–64 and remains in 4–128. Adaptation strength starts at 0.05 and remains in 0–0.5. Updates use old rates and old adaptation synchronously, without normalization of rates, adaptation or recurrent current. Total learned parameters: **714,912**, including **306,184** new parameters.

Both add 512 fp32 state values per example (2 KiB) beyond the opponent traces. Parameter count and initialization differ; this is not a parameter-matched comparison or literal reproduction of a biological circuit. The E/I initialization is more contractive than the LSTM retention-biased initialization. Its trajectories matter when interpreting acquisition.

Fresh Adam uses shared epsilon **1e-10**, chosen before profiling because the focused CPU E/I raw-recurrence gradient RMS was around 1.8e-9. The profiles log actual effective recurrent-weight changes, not merely a nonzero raw-parameter norm. Biases, normalization, timescale/adaptation and raw sign-constrained recurrence parameters receive no generic weight decay. Other matrix/convolution weights receive 1e-4 decay. Global gradient clipping is 1.0; pre-clip norms and clipping indicators are logged. Small or zero gradient fractions are descriptive, not automatic failure triggers.

Training is fp32 without AMP/TF32, one GPU worker per device, two PyTorch CPU threads and one interop thread. The initial plan was serial local execution; the explicit migration permits local LSTM and remote E/I to run concurrently on separate devices. Exact non-reentrant sensory-step checkpointing recomputes stateless operations while preserving full temporal gradients. No state is detached and homogeneous batches require no padding.

## Exact focused exposure

The existing generator laws are unchanged. A fixed interleaved 40-update cycle contains:

| Cell | Frames | Updates/cycle | Terminal episodes/arm |
|---|---:|---:|---:|
| Original motion sensory anchor | 2 | 2 | 2,000 |
| Original orientation sensory anchor | 2 | 2 | 2,000 |
| Motion duration, L2, no added delay | 5 | 9 | 9,000 |
| Motion duration, L8, no added delay | 11 | 9 | 9,000 |
| Orientation one-item match/change, D0, post-delay0 | 5 | 9 | 9,000 |
| Orientation one-item match/change, D4, post-delay2 | 11 | 9 | 9,000 |

Motion asks which cardinal direction occupied the most transitions, including revisits. L2 is the existing minimal constant-direction bridge; L8 uses the existing unique-duration-winner generator. The labels are not net displacement or the longest single run. Orientation recall uses the existing balanced match/change construction and rendered identity/query cues. Delays count observed blank frames; D4/post-delay2 is distinct from the D0/post-delay0 minimal cell. Every blank, cue and report advances both trace and memory state.

Both arms use training seed 593001 and the same task-local stream prefixes and ordering. Model RNG seed 59301 is separate; common interface seed 59311 and core seed 59312 are recorded. Validation seed is 1193001 and locked test seed 1293001. All batches contain fresh generated episodes; no fixed bank or acquisition canary is used.

Validation uses 128 episodes per cell and selects the arithmetic mean of the six cell OVR-AUCs, with exact ties going to the earlier checkpoint. Both models still train to the full endpoint. Final evaluation uses 512 episodes per cell, common across models. A paired intervention resets only the new recurrent state before every frame; opponent traces retain their history and the new branch still processes the current frame. This measures the contribution of that trained recurrent history, not the benefit over a separately trained memory-free control.

## Evidence and reproduction

`results.json` is the live aggregate. Each run preserves its source snapshot, exact configuration, parent identity, supervisor/worker logs, per-update CSV, immutable checkpoint index and full state: weights, fresh Adam state, stream state, all relevant RNG states and exposure counters. `focused_check.json` records the one CPU implementation check; `decision_review.md` records the independent read-only review. Profile fits and their costs are preserved separately and never initialize production.

The explicit worker interface is `C:/Python310/python.exe -B WorkingMemory/RecurrentComparison/train.py JOB.json`. Jobs in the run directory contain the original absolute deadline; they are reproducibility records, not authorization to renew it. Compatible continuation requires the same configuration, source hashes and indexed checkpoint bytes. Do not replay old launch commands as a new run.

State RMS/max, gate saturation, E/I activity, early/late representation gradients and recurrent-gradient/effective-update statistics are recorded every 64 updates on the actual fresh training batch. This introduces no optimizer step beyond the planned training update. State diagnostics are detached summaries, not additional memory routes. Source rationale and biological limits remain in [the design note](../Research/recurrent_memory_without_attention.md).
