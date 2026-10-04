"""FP32 gated delta recurrence and differentiable chunkwise equivalent.

Equations follow the official KDA recurrence, with query scale one:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
No FLA/Triton dependency or reduced precision intermediate is used.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _validate(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
              initial_state: Tensor | None) -> None:
    if q.ndim != 4 or q.shape != k.shape or g.shape != k.shape:
        raise ValueError("q, k and log decay must share [batch,time,head,key] shape")
    if v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("v and beta must share batch/time/head dimensions")
    tensors = (q, k, v, g, beta) + (() if initial_state is None else (initial_state,))
    if any(x.dtype != torch.float32 or x.device != q.device for x in tensors):
        raise ValueError("KDA requires FP32 tensors on one device")
    if torch.is_autocast_enabled(q.device.type):
        raise ValueError("KDA requires autocast disabled")
    if initial_state is not None and initial_state.shape != (q.shape[0], q.shape[2], q.shape[3], v.shape[3]):
        raise ValueError("invalid initial-state shape")
    if q.shape[1] == 0:
        raise ValueError("empty token sequence")


def recurrent_kda(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
                  initial_state: Tensor | None = None,
                  final_only: bool = False) -> tuple[Tensor, Tensor]:
    """Plain ordered reference; zero state by default, no detach anywhere."""
    _validate(q, k, v, g, beta, initial_state)
    b, length, heads, keys = q.shape
    state = q.new_zeros(b, heads, keys, v.shape[-1]) if initial_state is None else initial_state
    reads = []
    for i in range(length):
        decayed = state * g[:, i].exp().unsqueeze(-1)
        innovation = v[:, i] - (k[:, i].unsqueeze(-1) * decayed).sum(-2)
        state = decayed + (beta[:, i].unsqueeze(-1) * k[:, i]).unsqueeze(-1) * innovation.unsqueeze(-2)
        if not final_only or i == length - 1:
            reads.append((q[:, i].unsqueeze(-1) * state).sum(-2))
    return (reads[-1] if final_only else torch.stack(reads, dim=1)), state


def chunk_kda(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
              initial_state: Tensor | None = None, chunk_size: int = 32,
              final_only: bool = False) -> tuple[Tensor, Tensor]:
    """Whole-sequence FP32 chunk solve, with full state gradients across chunks.

    For chunk-local cumulative log decay G, innovations obey
      (I + tril(beta_i * <k_i, exp(G_i-G_j) k_j>, -1)) delta
        = beta * (v - exp(G) k M_in).
    Solve once for input-independent U and W; delta = U - W M_in.
    Pairwise differences are masked BEFORE exponentiation, so noncausal
    inverse-decay factors cannot overflow. No decay clamp changes recurrence.
    Padding has zero keys/writes/decays and is therefore exactly neutral.
    """
    _validate(q, k, v, g, beta, initial_state)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("disable torch.backends.cuda.matmul.allow_tf32 for strict FP32")
    b, length, heads, keys = q.shape
    values = v.shape[-1]
    chunks = (length + chunk_size - 1) // chunk_size
    padded = chunks * chunk_size

    def arrange(x: Tensor) -> Tensor:
        if padded != length:
            x = torch.cat((x, x.new_zeros((b, padded - length, *x.shape[2:]))), dim=1)
        return x.transpose(1, 2).reshape(b, heads, chunks, chunk_size, *x.shape[3:])

    kc, vc, gc, bc = [arrange(x) for x in (k, v, g, beta)]
    cumulative = gc.cumsum(-2)
    lower = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).tril(-1)
    # [B,H,N,query_token,key_token,key_channel]; only causal decay ratios.
    difference = cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)
    decay_ratio = difference.masked_fill(~lower.unsqueeze(-1), 0).exp()
    pair_keys = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * decay_ratio).sum(-1)
    identity = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
    system = identity + (bc.unsqueeze(-1) * pair_keys).masked_fill(~lower, 0)
    input_key = cumulative.exp() * kc
    rhs = bc.unsqueeze(-1) * torch.cat((vc, input_key), dim=-1)
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    u, w = solved.split((values, keys), dim=-1)
    end_keys = ((cumulative[..., -1:, :] - cumulative).exp() * kc).transpose(-1, -2)
    end_decay = cumulative[..., -1, :].exp().unsqueeze(-1)
    state = q.new_zeros(b, heads, keys, values) if initial_state is None else initial_state
    incoming, innovations = [], []
    for index in range(chunks):
        delta = u[:, :, index] - w[:, :, index] @ state
        if not final_only:
            incoming.append(state)
            innovations.append(delta)
        state = end_decay[:, :, index] * state + end_keys[:, :, index] @ delta
    if final_only:
        return (q[:, -1].unsqueeze(-1) * state).sum(-2), state
    qc = arrange(q)
    incoming = torch.stack(incoming, dim=2)
    innovations = torch.stack(innovations, dim=2)
    causal = lower | identity.bool()
    overlap = (qc.unsqueeze(-2) * kc.unsqueeze(-3) * decay_ratio).sum(-1).masked_fill(~causal, 0)
    output = (qc * cumulative.exp()) @ incoming + overlap @ innovations
    output = output.reshape(b, heads, padded, values).transpose(1, 2)[:, :length]
    return output, state
