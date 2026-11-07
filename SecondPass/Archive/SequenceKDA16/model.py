"""One global 16-head KDA, width-128 patch tokens and terminal CLS readout."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from SecondPass.SequenceKDA.kda_backend import chunk_kda, recurrent_kda
from SecondPass.SequenceKDA.model import patchify, temporal_features


class KDALayer(nn.Module):
    heads = 16
    head_dim = 64

    def __init__(self, backend: str = "chunk", chunk_size: int = 32):
        super().__init__()
        if backend not in ("chunk", "reference"):
            raise ValueError("backend must be chunk or reference")
        self.backend, self.chunk_size = backend, chunk_size
        self.q_proj = nn.Linear(128, 1024)
        self.k_proj = nn.Linear(128, 1024)
        self.v_proj = nn.Linear(128, 1024)
        self.decay_proj = nn.Linear(128, 1024)
        self.write_proj = nn.Linear(128, 16)
        self.out_proj = nn.Linear(1024, 128)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.write_proj.weight)
        with torch.no_grad():
            for head in range(16):
                half_life = 32 if head < 8 else 128
                self.decay_proj.bias[head * 64:(head + 1) * 64].fill_(
                    math.log(math.expm1(math.log(2) / (100 * half_life))))
            self.write_proj.bias.fill_(math.log(0.01 / 0.99))

    def forward(self, normalized: Tensor, final_only: bool = True) -> Tensor:
        batch, length, _ = normalized.shape
        shape = (batch, length, 16, 64)
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
        batch, times = frames.shape[:2]
        positions = self.row_embedding[:, None] + self.column_embedding[None, :]
        time = temporal_features(torch.arange(times + 1, device=frames.device))
        patches = patches + positions[None, None] + time[None, :times, None, None]
        cls = (self.cls_token + time[None, times:times + 1]).expand(batch, -1, -1)
        return torch.cat((patches.reshape(batch, times * 100, 128), cls), dim=1)

    def forward(self, frames: Tensor, task: str = "krauzlis_cued_motion") -> Tensor:
        if task not in ("krauzlis", "krauzlis_cued_motion"):
            raise ValueError("SequenceKDA16 is the native Krauzlis-only experiment")
        tokens = self.tokens(frames)
        final = tokens[:, -1] + self.kda(self.kda_norm(tokens), final_only=True)
        return self.classifier(self.final_norm(final))
