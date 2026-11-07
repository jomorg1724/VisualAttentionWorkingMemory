"""Independent dot clips with angular similarity supplied only to the loss."""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions, SPEEDS

NAMESPACES = {'train':186101, 'val':186102, 'test':186103}


def render_clip(rng, theta):
    speed = float(rng.choice(SPEEDS))
    positions = rng.uniform(0, 100, (int(rng.integers(16, 49)), 2))
    velocity = speed * np.array([math.cos(theta), math.sin(theta)])
    movie = positions[None] + np.arange(3)[:, None, None] * velocity
    return torch.from_numpy(np.stack([render_positions(p) for p in movie])), speed


def generate(index, split='train'):
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], index]))
    first_theta = float(rng.uniform(0, 2 * math.pi))
    group = index % 3
    delta = 0. if group == 0 else (float(rng.choice([26.,28.])) if group == 1 else float(rng.uniform(45,180)))
    signed = delta * int(rng.choice([-1,1]))
    second_theta = (first_theta + math.radians(signed)) % (2 * math.pi)
    first, first_speed = render_clip(rng, first_theta)
    second, second_speed = render_clip(rng, second_theta)
    metadata = dict(index=index, split=split, group=group, theta_first=first_theta,
                    theta_second=second_theta, delta_degrees=delta,
                    first_speed=first_speed, second_speed=second_speed)
    return first, second, math.radians(delta), metadata
