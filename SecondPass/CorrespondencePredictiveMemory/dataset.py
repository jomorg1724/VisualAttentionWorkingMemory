"""Two independent appearances of the same six-frame transformation sequence.

Only pixels enter the model. Shared dynamics identify augmentation positives;
no angle, displacement or simulator state is an optimization target.
"""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions

NAMESPACES = dict(train=194101, val=194102, test=194103)
SPEEDS, ANGLES = (1.5, 4., 8.), (26, 28)
VERSION = 'paired_appearance_six_frame_correspondence_v1'


def generate(index, split='train', evaluation=False, view=0, domain='standard'):
    if domain not in ('standard', 'transfer') or (domain == 'transfer' and split != 'test'):
        raise ValueError('Transfer is a held-out test only')
    context = index // 2 if evaluation else index
    cell, label = (index // 2) % 6, index % 2
    speeds, angles = (SPEEDS, ANGLES) if domain == 'standard' else ((.75, 3., 6.), (18, 42))
    speed, angle = speeds[cell % 3], angles[cell // 3]
    domain_id = int(domain == 'transfer')
    dynamics = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], int(evaluation), context, 0, domain_id]))
    theta, sign = float(dynamics.uniform(0, 2*math.pi)), int(dynamics.choice([-1, 1]))
    appearance = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], int(evaluation), context, 1, view, domain_id]))
    initial = appearance.uniform(0, 100, (int(appearance.integers(16, 49)), 2))
    old = speed * np.array([math.cos(theta), math.sin(theta)])
    new_theta = theta + label * sign * math.radians(angle)
    new = speed * np.array([math.cos(new_theta), math.sin(new_theta)])
    positions = np.concatenate((initial[None] + np.arange(3)[:, None, None]*old,
        initial[None] + 3*old + np.arange(3)[:, None, None]*new))
    # Grayscale storage avoids three identical RAM copies. Model accepts RGB too.
    movie = torch.from_numpy(np.stack([render_positions(p)[:1] for p in positions]))
    metadata = dict(index=index, context=context, cell=cell, speed=speed, angle=angle,
        namespace=NAMESPACES[split], evaluation=evaluation, view=view, domain=domain, analysis_only=True)
    return movie, label, metadata


def training_pair(index):
    first, label, meta = generate(index, view=0)
    second, other_label, _ = generate(index, view=1)
    assert label == other_label
    return torch.stack((first, second)), label, meta
