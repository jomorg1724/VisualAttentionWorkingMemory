# Three-layer FP32 backend

`kda_backend.py` reuses the verified torch-only FP32 gated-delta chunk computation from `SecondPass/SequenceKDA`. Its equations follow the [official KDA recurrent reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py). Chunk-local triangular solves propagate state and full gradients across the whole sequence; neutral padding never truncates or resets a trial. Noncausal decay differences are masked before exponentiation to avoid inverse-decay overflow. Autocast and non-FP32 tensors are rejected, and CUDA matmul TF32 must be disabled.

The first two blocks request all token reads because those contextualized tokens supply keys, values, gates and queries to the next block. The third block requests only final CLS, since earlier third-layer queries cannot affect its recurrent state or final output. Each block begins with zero state; blocks share neither parameters nor state. No states are detached.

The original operator was checked against direct recurrence at native 4,501 tokens, two heads and 64-by-64 state width, including output/state and input gradients. This version adds focused short-stack output and all-parameter gradient parity against three direct recurrent blocks, plus an Adam step advancing every one of 55 trainable tensors. It does not repeat exhaustive long-sequence CPU audits. Native-length CUDA profiling measures the actual three-layer model before production exposure is pinned.

Run `python -m SecondPass.SequenceKDA3.test_model --receipt SecondPass/SequenceKDA3/verification/model_cpu.json` for the focused CPU checks. The backend has no FLA/Triton dependency; PyTorch2.8 is sufficient.
