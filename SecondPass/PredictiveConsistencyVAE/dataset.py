"""Fresh simplified continuous-dot movies; never supply analysis labels to loss."""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions, SPEEDS

NAMESPACES = {'train': 189101, 'val': 189102, 'test': 189103}
ANGLES = (26, 28)
VERSION = 'continuous_twelve_frame_predictive_consistency_v2'


def _generate(index, split, evaluation, clip_frames):
    if split not in NAMESPACES or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError('Require a nonnegative integer index and train/val/test split')
    index = int(index)
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
    before_positions = initial[None] + np.arange(clip_frames)[:, None, None] * before_velocity
    after_positions = initial[None] + clip_frames * before_velocity + np.arange(clip_frames)[:, None, None] * after_velocity
    positions = np.concatenate([before_positions, after_positions])
    frames = torch.from_numpy(np.stack([render_positions(p) for p in positions]))
    metadata = dict(version=VERSION, split=split, namespace=NAMESPACES[split],
                    index=index, context=context, evaluation=evaluation, cell=cell,
                    speed=speed, angle=angle, signed_change=label * sign * angle,
                    theta=theta, dot_count=len(initial),
                    velocity_before_xy=before_velocity.tolist(),
                    velocity_after_xy=after_velocity.tolist(),
                    positions_unwrapped_xy=positions.tolist(), clip_frames=clip_frames,
                    analysis_only=True)
    return frames[:clip_frames], frames[clip_frames:], label, metadata


def generate(index, split='train'):
    """Two stable six-frame clips; each has three context and three future frames."""
    return _generate(index, split, False, 6)


def generate_evaluation(index, split='val'):
    """Original three-plus-three change assay, not a twelve-frame task rewrite."""
    return _generate(index, split, True, 3)


def generate_objective_evaluation(index, split='val'):
    """Fresh full-horizon unsupervised validation, distinct from change evaluation."""
    return _generate(int(index) + 60000000, split, True, 6)
