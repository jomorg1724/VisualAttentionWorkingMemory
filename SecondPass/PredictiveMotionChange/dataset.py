"""Six continuous dot frames, with constant directions inside each three-frame clip."""
import math
import numpy as np
import torch
from SecondPass.VariationalMotionPredictor.dataset import render_positions, SPEEDS

NAMESPACES = {'train': 185101, 'val': 185102, 'test': 185103}
ANGLES = (26, 28)


def generate(index, split='train'):
    # Adjacent change/no-change examples share all nuisance variables and the
    # complete before clip. Neither clip alone has a direction-label association.
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACES[split], index // 2]))
    label = index % 2
    speed = SPEEDS[(index // 2) % 3]
    angle = ANGLES[(index // 6) % 2]
    theta = rng.uniform(0, 2 * math.pi)
    sign = int(rng.choice([-1, 1]))
    initial = rng.uniform(0, 100, (int(rng.integers(16, 49)), 2))
    before_velocity = speed * np.array([math.cos(theta), math.sin(theta)])
    after_theta = theta + label * sign * math.radians(angle)
    after_velocity = speed * np.array([math.cos(after_theta), math.sin(after_theta)])
    before_positions = initial[None] + np.arange(3)[:, None, None] * before_velocity
    # Frame3 starts where unchanged motion would put it; transitions3->4->5
    # use the after direction. No image reinitialization or dot respawning.
    after_positions = initial[None] + 3 * before_velocity + np.arange(3)[:, None, None] * after_velocity
    before = torch.from_numpy(np.stack([render_positions(p) for p in before_positions]))
    after = torch.from_numpy(np.stack([render_positions(p) for p in after_positions]))
    return before, after, label, dict(index=index, pair=index // 2, speed=speed,
                                    angle=angle, signed_change=label * sign * angle)
