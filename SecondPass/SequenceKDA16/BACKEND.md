# Sixteen-head FP32 backend

The model imports `SecondPass.SequenceKDA.kda_backend` unchanged. Its torch-only chunk-local triangular solves implement the [official KDA recurrence](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py) with query scale one, FP32 state and arithmetic, neutral padding and full state gradients across chunks. Noncausal decay differences are masked before exponentiation. There is no FLA/Triton dependency, autocast, mixed precision or truncated movie.

Head count changes tensor shapes, not operator mathematics. The same backend handles `[B,L,16,64]` q/k/v/decay tensors and `[B,L,16]` write strengths. Its final state is `[B,16,64,64]`. State resets to zero for every trial. Only the terminal CLS query is needed for a single layer; unused earlier reads are omitted without affecting final state or gradients. CUDA TF32 must remain disabled.

The original operator has native-length recurrence/gradient evidence. This experiment adds a focused sixteen-head, 64-by-64 short-sequence comparison of outputs, final state and q/k/v/decay/write/initial-state gradients. The short model check verifies all 27 tensors advance under Adam and all sixteen heads receive nonzero gradients. No exhaustive 4,501-token CPU campaign is repeated; the eventual native CUDA profile measures actual larger-head memory and speed before pinning production exposure.

Run `python -m SecondPass.SequenceKDA16.test_model --receipt SecondPass/SequenceKDA16/verification/model_cpu.json` for the focused CPU check. All learned weights are initialized fresh, including every head.
