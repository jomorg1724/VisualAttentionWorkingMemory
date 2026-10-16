"""Fresh continuous six-frame movies for weighted-image VAE and classifier.

Both three-frame clips share a persistent dot layout. The transition to frame3
continues the original velocity; only 3->4->5 changes on a changed trial. This
is the simplified synthetic assay, not a native Krauzlis replication.
"""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions, SPEEDS

NAMESPACES = {'train': 190101, 'val': 190102, 'test': 190103}
ANGLES = (26, 28)
VERSION = 'continuous_six_frame_weighted_sum_123_v1'


def _generate(index, split, mode):
    if split not in NAMESPACES or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError('Require nonnegative integer index and train/val/test split')
    index = int(index)
    paired = mode == 'evaluation'
    balanced = mode != 'vae'
    context = index // 2 if paired else index
    cell = (index // 2) % 6 if balanced else (index // 3) % 6
    label = index % 2 if balanced else int(index % 3 == 2)
    speed, angle = SPEEDS[cell % 3], ANGLES[cell // 3]
    # Mode-specific domains prevent classifier movies replaying VAE movies,
    # while evaluation counterparts alone share nuisance draws intentionally.
    domain = {'vae': 0, 'classifier': 1, 'evaluation': 2}[mode]
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], domain, context]))
    theta = float(rng.uniform(0, 2 * math.pi))
    sign = int(rng.choice([-1, 1]))
    initial = rng.uniform(0, 100, (int(rng.integers(16, 49)), 2))
    vb = speed * np.array([math.cos(theta), math.sin(theta)])
    after_theta = theta + label * sign * math.radians(angle)
    va = speed * np.array([math.cos(after_theta), math.sin(after_theta)])
    before = initial[None] + np.arange(3)[:, None, None] * vb
    after = initial[None] + 3 * vb + np.arange(3)[:, None, None] * va
    positions = np.concatenate([before, after])
    frames = torch.from_numpy(np.stack([render_positions(p) for p in positions]))
    metadata = dict(version=VERSION, split=split, namespace=NAMESPACES[split], domain=domain,
                    index=index, context=context, mode=mode, evaluation=paired, cell=cell,
                    speed=speed, angle=angle, signed_change=label * sign * angle,
                    theta=theta, dot_count=len(initial), velocity_before_xy=vb.tolist(),
                    velocity_after_xy=va.tolist(), positions_unwrapped_xy=positions.tolist(),
                    analysis_only=True)
    return frames[:3], frames[3:], label, metadata


def generate(index, split='train'):
    """Image-only VAE stage: occasional changes (one third), labels ignored."""
    return _generate(index, split, 'vae')


def generate_classifier(index, split='train'):
    """Balanced fresh supervised pairs, independently drawn nuisance per trial."""
    return _generate(index, split, 'classifier')


def generate_evaluation(index, split='val'):
    """Balanced paired unchanged/changed nuisance draws for held-out evaluation."""
    return _generate(index, split, 'evaluation')
