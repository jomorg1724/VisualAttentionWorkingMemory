"""Carry prior memory directly while retaining the existing RViT proposal.

These constructors always initialize a fresh whole model. No checkpoint migration,
task changes, loss changes or live-training changes are performed here.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from SecondPass.TwoFrameRViT.model import TwoFrameRViT
from SecondPass.StructuredMotionRViT.model import StructuredMotionRViT


class MemoryCarryBlock(nn.Module):
    """One content-dependent carry gate per spatial token and memory channel."""

    def __init__(self, proposal: nn.Module, initial_carry: float = 0.98):
        super().__init__()
        if not 0.0 < initial_carry < 1.0:
            raise ValueError("initial_carry must be strictly between zero and one")
        self.proposal = proposal
        self.gate = nn.Linear(512, 256)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(initial_carry / (1.0 - initial_carry)))

    def forward(self, current: Tensor, memory: Tensor) -> Tensor:
        # Normalize the gate's observations, never the direct memory path.
        # Parameter-free normalization avoids extra scale/bias learners here.
        observations = torch.cat((F.layer_norm(current, (256,)),
                                  F.layer_norm(memory, (256,))), dim=-1)
        carry = self.gate(observations).sigmoid()
        candidate = self.proposal(current, memory)
        return carry * memory + (1.0 - carry) * candidate


class GatedTwoFrameRViT(TwoFrameRViT):
    def __init__(self, checkpoint_encoder: bool = True, initial_carry: float = 0.98):
        super().__init__(checkpoint_encoder=checkpoint_encoder)
        self.recurrent_block = MemoryCarryBlock(self.recurrent_block, initial_carry)


class GatedStructuredMotionRViT(StructuredMotionRViT):
    def __init__(self, checkpoint_encoder: bool = True, initial_carry: float = 0.98):
        super().__init__(checkpoint_encoder=checkpoint_encoder)
        self.recurrent_block = MemoryCarryBlock(self.recurrent_block, initial_carry)
