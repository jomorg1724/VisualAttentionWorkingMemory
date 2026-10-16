"""Six continuous frames; pixel-only training targets, analysis metadata separate.

This simplified persistent-dot stimulus is not native Krauzlis. The frame at
index 3 follows the original velocity; only transitions 3->4->5 may change.
"""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions, SPEEDS

NAMESPACES = {'train': 187101, 'val': 187102, 'test': 187103}
ANGLES = (26, 28)
VERSION = 'continuous_six_frame_pixel_contrastive_v1'


def _generate(index, split, evaluation):
    if split not in NAMESPACES or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError('Require a nonnegative integer index and train/val/test split')
    index = int(index)
    # Evaluation counterparts share nuisance draws. Training gives each clip
    # pair a fresh movie, with two unchanged trials for every changed trial.
    context = index // 2 if evaluation else index
    cell = (index // 2) % 6 if evaluation else (index // 3) % 6
    label = index % 2 if evaluation else int(index % 3 == 2)
    speed, angle = SPEEDS[cell % 3], ANGLES[cell // 3]
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], context]))
    theta = float(rng.uniform(0, 2 * math.pi))
    sign = int(rng.choice([-1, 1]))
    initial = rng.uniform(0, 100, (int(rng.integers(16, 49)), 2))
    before_velocity = speed * np.array([math.cos(theta), math.sin(theta)])
    after_theta = theta + label * sign * math.radians(angle)
    after_velocity = speed * np.array([math.cos(after_theta), math.sin(after_theta)])
    before_positions = initial[None] + np.arange(3)[:, None, None] * before_velocity
    after_positions = initial[None] + 3 * before_velocity + np.arange(3)[:, None, None] * after_velocity
    frames = torch.from_numpy(np.stack([render_positions(p) for p in np.concatenate(
        [before_positions, after_positions], axis=0)]))
    metadata = dict(version=VERSION, split=split, namespace=NAMESPACES[split],
                    index=index, context=context, evaluation=evaluation, cell=cell,
                    speed=speed, angle=angle, signed_change=label * sign * angle,
                    theta=theta, dot_count=len(initial),
                    velocity_before_xy=before_velocity.tolist(),
                    velocity_after_xy=after_velocity.tolist(),
                    positions_unwrapped_xy=np.concatenate([before_positions, after_positions]).tolist(),
                    analysis_only=True)
    return frames[:3], frames[3:], label, metadata


def generate(index, split='train'):
    """Fresh training movie, occasional change; labels are analysis-only."""
    return _generate(index, split, False)


def generate_evaluation(index, split='val'):
    """Balanced adjacent unchanged/changed movies, paired nuisance, six cells."""
    return _generate(index, split, True)
