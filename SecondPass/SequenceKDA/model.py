"""Exactly one global patch-sequence KDA, reporting only appended CLS."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from SecondPass.SequenceKDA.kda_backend import chunk_kda, recurrent_kda


def patchify(frames: Tensor) -> Tensor:
    """[B,T,3,100,100] -> [B,T,10,10,300], raster row/column order."""
    if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 100, 100):
        raise ValueError("expected [batch,time,3,100,100] native RGB movies")
    return frames.reshape(*frames.shape[:2], 3, 10, 10, 10, 10).permute(0, 1, 3, 5, 2, 4, 6).reshape(*frames.shape[:2], 10, 10, 300)


def temporal_features(indices: Tensor, width: int = 128) -> Tensor:
    frequencies = torch.exp(torch.arange(0, width, 2, device=indices.device, dtype=torch.float32) * (-math.log(10000) / width))
    angles = indices.float().unsqueeze(-1) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class KDALayer(nn.Module):
    def __init__(self, backend: str = "chunk", chunk_size: int = 32):
        super().__init__()
        if backend not in ("chunk", "reference"):
            raise ValueError("backend must be chunk or reference")
        self.backend, self.chunk_size = backend, chunk_size
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)
        self.decay_proj = nn.Linear(128, 128)
        self.write_proj = nn.Linear(128, 2)
        self.out_proj = nn.Linear(128, 128)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.write_proj.weight)
        with torch.no_grad():
            for head, half_life in enumerate((32, 128)):
                self.decay_proj.bias[head * 64:(head + 1) * 64].fill_(math.log(math.expm1(math.log(2) / (100 * half_life))))
            self.write_proj.bias.fill_(math.log(0.01 / 0.99))

    def forward(self, normalized: Tensor, final_only: bool = True) -> Tensor:
        b, length, _ = normalized.shape
        shape = (b, length, 2, 64)
        q = F.normalize(self.q_proj(normalized).reshape(shape), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(normalized).reshape(shape), dim=-1, eps=1e-6)
        v = self.v_proj(normalized).reshape(shape)
        g = -F.softplus(self.decay_proj(normalized).reshape(shape))
        beta = self.write_proj(normalized).sigmoid()
        if self.backend == "chunk":
            read, _ = chunk_kda(q, k, v, g, beta, chunk_size=self.chunk_size, final_only=final_only)
        else:
            read, _ = recurrent_kda(q, k, v, g, beta, final_only=final_only)
        return self.out_proj(read.flatten(-2))


class SequenceKDA(nn.Module):
    def __init__(self, backend: str = "chunk", chunk_size: int = 32):
        super().__init__()
        self.patch_projection = nn.Linear(300, 128)
        self.patch_norm = nn.LayerNorm(128)
        self.row_embedding = nn.Parameter(torch.empty(10, 128))
        self.column_embedding = nn.Parameter(torch.empty(10, 128))
        self.cls_token = nn.Parameter(torch.empty(1, 1, 128))
        for parameter in (self.row_embedding, self.column_embedding, self.cls_token):
            nn.init.normal_(parameter, std=0.02)
        self.kda_norm = nn.LayerNorm(128)
        self.kda = KDALayer(backend=backend, chunk_size=chunk_size)
        self.final_norm = nn.LayerNorm(128)
        self.classifier = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 2))

    def tokens(self, frames: Tensor) -> Tensor:
        if frames.dtype != torch.float32:
            raise ValueError("native image input must be FP32")
        patches = self.patch_norm(F.gelu(self.patch_projection(patchify(frames - 0.5))))
        b, times = frames.shape[:2]
        positions = self.row_embedding[:, None] + self.column_embedding[None, :]
        time = temporal_features(torch.arange(times + 1, device=frames.device))
        patches = patches + positions[None, None] + time[None, :times, None, None]
        cls = (self.cls_token + time[None, times:times + 1]).expand(b, -1, -1)
        return torch.cat((patches.reshape(b, times * 100, 128), cls), dim=1)

    def forward(self, frames: Tensor, task: str = "krauzlis") -> Tensor:
        if task not in ("krauzlis", "krauzlis_cued_motion"):
            raise ValueError("SequenceKDA is the fixed Krauzlis-only experiment")
        tokens = self.tokens(frames)
        final = tokens[:, -1] + self.kda(self.kda_norm(tokens), final_only=True)
        return self.classifier(self.final_norm(final))
