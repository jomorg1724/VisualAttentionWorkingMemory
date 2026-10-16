"""Six raw frames from a continuous dot movie; only terminal binary feedback.

Same faster-dot geometry as run02_speed4, fresh task-local random streams.
Metadata is analysis-only and never enters the forward model.
"""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions

NAMESPACES = dict(train=193101, val=193102, test=193103)
SPEEDS, ANGLES = (1.5, 4., 8.), (26, 28)
VERSION = 'outcome_rvit_continuous_six_raw_frames_v1'


def generate(index, split='train', evaluation=False):
    context = index // 2 if evaluation else index
    cell, label = (index // 2) % 6, index % 2
    speed, angle = SPEEDS[cell % 3], ANGLES[cell // 3]
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], int(evaluation), context]))
    theta, sign = float(rng.uniform(0, 2 * math.pi)), int(rng.choice([-1, 1]))
    initial = rng.uniform(0, 100, (int(rng.integers(16, 49)), 2))
    old = speed * np.array([math.cos(theta), math.sin(theta)])
    new_theta = theta + label * sign * math.radians(angle)
    new = speed * np.array([math.cos(new_theta), math.sin(new_theta)])
    positions = np.concatenate((initial[None] + np.arange(3)[:, None, None] * old,
        initial[None] + 3 * old + np.arange(3)[:, None, None] * new))
    movie = torch.from_numpy(np.stack([render_positions(p) for p in positions]))
    metadata = dict(index=index, context=context, cell=cell, speed=speed, angle=angle,
                    namespace=NAMESPACES[split], evaluation=evaluation,
                    signed_change=label * sign * angle, analysis_only=True)
    return movie, label, metadata
