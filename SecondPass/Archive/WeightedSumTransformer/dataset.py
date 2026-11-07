"""Fresh balanced continuous six-frame dots; simplified, not native Krauzlis.

The second clip's first frame continues the old velocity. Changes affect only
its last two transitions. Training uses independent nuisances per movie;
held-out evaluation pairs unchanged/changed movies with shared nuisances.
"""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions, SPEEDS

NAMESPACES = {'train': 191101, 'val': 191102, 'test': 191103}
ANGLES = (26, 28)
VERSION = 'continuous_six_frame_weighted_sum_transformer_v1'
SPEED_SCALE = 1.


def configure_speed_scale(value):
    """Set one explicit speed intervention for this worker's entire run."""
    global SPEED_SCALE
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Speed scale must be finite and positive')
    SPEED_SCALE = value


def _generate(index, split, evaluation):
    if split not in NAMESPACES or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError('Require nonnegative integer index and train/val/test split')
    index = int(index)
    context = index // 2 if evaluation else index
    cell = (index // 2) % 6
    label = index % 2
    base_speed, angle = SPEEDS[cell % 3], ANGLES[cell // 3]
    speed = base_speed * SPEED_SCALE
    domain = int(evaluation)
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], domain, context]))
    theta = float(rng.uniform(0, 2 * math.pi))
    sign = int(rng.choice([-1, 1]))
    initial = rng.uniform(0, 100, (int(rng.integers(16, 49)), 2))
    before_velocity = speed * np.array([math.cos(theta), math.sin(theta)])
    after_theta = theta + label * sign * math.radians(angle)
    after_velocity = speed * np.array([math.cos(after_theta), math.sin(after_theta)])
    before_positions = initial[None] + np.arange(3)[:, None, None] * before_velocity
    after_positions = initial[None] + 3 * before_velocity + np.arange(3)[:, None, None] * after_velocity
    positions = np.concatenate([before_positions, after_positions])
    frames = torch.from_numpy(np.stack([render_positions(p) for p in positions]))
    metadata = dict(version=VERSION, split=split, namespace=NAMESPACES[split], domain=domain,
                    index=index, context=context, evaluation=evaluation, cell=cell,
                    speed=speed, base_speed=base_speed, speed_scale=SPEED_SCALE,
                    angle=angle, signed_change=label * sign * angle,
                    theta=theta, dot_count=len(initial),
                    velocity_before_xy=before_velocity.tolist(),
                    velocity_after_xy=after_velocity.tolist(),
                    positions_unwrapped_xy=positions.tolist(), analysis_only=True)
    return frames[:3], frames[3:], label, metadata


def generate(index, split='train'):
    return _generate(index, split, False)


def generate_evaluation(index, split='val'):
    return _generate(index, split, True)
