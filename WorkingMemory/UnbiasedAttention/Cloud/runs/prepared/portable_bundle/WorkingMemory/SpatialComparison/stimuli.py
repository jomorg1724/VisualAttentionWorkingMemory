"""CPU-only spatial binding sequences; labels/metadata never enter a model.

Binding adds a two-location preserve/swap task. Other families reuse the current
sequence generator without editing it. Each family owns checkpointable RNG state.
"""
from __future__ import annotations

import copy
import re
import numpy as np
import torch
from WorkingMemory.stimuli import SequenceStream, setting, visual_cues

VERSION = 'spatial_binding_sequences_v1'
FAMILY_HEAD = {'orientation_single': 'orientation',
               'orientation_binding': 'orientation_binding',
               'motion_direction': 'motion_direction'}
TASK_CLASSES = {'orientation': 2, 'orientation_binding': 2, 'motion_direction': 4}
FAMILIES = tuple(FAMILY_HEAD)
DELAYS = (0, 4, 12, 24)
SEPARATIONS_DEGREES = (15, 30, 60)
PATCH_RADIUS = 11.0
LOCATION_GRIDS = {
    'standard': {'midpoint_x': (46., 50., 54.), 'y': (32., 50., 68.)},
    'locations': {'midpoint_x': (48., 52.), 'y': (41., 59.)},
}


def counterbalance_cases():
    """Exact four-case assignment design; A/B are unordered orientation items."""
    return [dict(sample_bit=s, label=y, sample=[s, 1-s],
                 probe=[s ^ y, 1-(s ^ y)]) for s in (0, 1) for y in (0, 1)]


def normalize_condition(condition):
    if isinstance(condition, (int, np.integer)):
        cfg = {'delay': int(condition)}
    elif isinstance(condition, str):
        match = re.search(r'D(\d+)', condition)
        if not match:
            raise ValueError('String condition must contain D0/D4/D12/D24')
        cfg = {'delay': int(match.group(1)), 'name': condition}
        if 'near' in condition: cfg['spacing'] = 'near'
        if 'far' in condition: cfg['spacing'] = 'far'
    else:
        cfg = dict(condition)
    cfg.setdefault('delay', 0)
    cfg.setdefault('spacing', 'mixed')
    cfg.setdefault('name', 'D%d_%s' % (cfg['delay'], cfg['spacing']))
    if cfg['delay'] not in DELAYS or cfg['spacing'] not in ('near', 'far', 'mixed'):
        raise ValueError(cfg)
    return cfg


def condition_frames(condition, family):
    cfg = normalize_condition(condition)
    return (11 if family == 'motion_direction' else 5) + cfg['delay']


class SpatialStream:
    def __init__(self, seed, split='train'):
        if split not in ('train', 'val', 'test', 'locations'): raise ValueError(split)
        self.seed, self.split = int(seed), split
        native_split = 'test' if split == 'locations' else split
        self.native = {
            'orientation_single': SequenceStream(self.seed + 100003, native_split),
            'motion_direction': SequenceStream(self.seed + 200003, native_split),
        }
        self.rng = np.random.default_rng(self.seed + 300007)
        self.counters = dict.fromkeys(FAMILIES, 0)
        self.pending_cases = []
        self.yy, self.xx = np.mgrid[:100, :100]

    def state_dict(self):
        return dict(version=VERSION, seed=self.seed, split=self.split,
                    rng=copy.deepcopy(self.rng.bit_generator.state),
                    counters=copy.deepcopy(self.counters),
                    pending_cases=copy.deepcopy(self.pending_cases),
                    native={k: v.state_dict() for k, v in self.native.items()})

    def load_state_dict(self, state):
        if (state['version'], state['seed'], state['split']) != (VERSION, self.seed, self.split):
            raise ValueError('Spatial stream identity mismatch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.counters = copy.deepcopy(state['counters'])
        self.pending_cases = copy.deepcopy(state['pending_cases'])
        for k, v in self.native.items(): v.load_state_dict(state['native'][k])

    def _case(self):
        if not self.pending_cases:
            self.pending_cases = self.rng.permutation(4).tolist()
        return counterbalance_cases()[self.pending_cases.pop()]

    def _render_pair(self, positions, angles, wavelength, amplitude):
        # Each patch phase and each pixel-noise raster are fresh on every frame.
        # Shared episode frequency/amplitude never identify or travel with items.
        field = self.rng.normal(0., .008, (100, 100))
        phases = self.rng.uniform(0, 2*np.pi, 2)
        for (x, y), theta, phase in zip(positions, angles, phases):
            dx, dy = self.xx-x, self.yy-y
            radius2 = dx*dx+dy*dy
            envelope = np.exp(-radius2/(2*4.5**2)) * (radius2 <= PATCH_RADIUS**2)
            carrier = np.cos(2*np.pi*(dx*np.cos(theta)+dy*np.sin(theta))/wavelength+phase)
            field += amplitude*envelope*carrier
        raster = np.repeat(np.clip(.5+field, 0, 1)[None], 3, axis=0).astype(np.float32)
        return raster, phases.tolist()

    def _binding(self, cfg):
        case = self._case()
        spacing = cfg['spacing']
        if spacing == 'mixed': spacing = str(self.rng.choice(['near', 'far']))
        distance = 26. if spacing == 'near' else 46.
        grid_name = 'locations' if self.split == 'locations' else 'standard'
        grid = LOCATION_GRIDS[grid_name]
        midpoint = float(self.rng.choice(grid['midpoint_x']))
        y = float(self.rng.choice(grid['y']))
        positions = [[midpoint-distance/2, y], [midpoint+distance/2, y]]
        separation = int(self.rng.choice(SEPARATIONS_DEGREES))
        theta = float(self.rng.uniform(0, np.pi))
        signed_sep = separation * int(self.rng.choice([-1, 1]))
        angles = np.mod([theta, theta+np.deg2rad(signed_sep)], np.pi)
        sample_angles = angles[case['sample']]
        probe_angles = angles[case['probe']]
        wavelength = float(self.rng.uniform(5., 7.))
        amplitude = float(self.rng.uniform(.26, .34))
        blank = np.full((3, 100, 100), .5, np.float32)
        frames = [visual_cues(blank, 'recall', 'sample')]
        phase_records = []
        for _ in range(2):
            raster, phases = self._render_pair(positions, sample_angles, wavelength, amplitude)
            frames.append(visual_cues(raster, 'recall', 'sample'))
            phase_records.append(phases)
        # No RNG draws in blank rendering: exact evidence matching across delays.
        frames.extend(visual_cues(blank, 'recall', 'ignore') for _ in range(cfg['delay']))
        frames.append(visual_cues(blank, 'recall', 'query'))
        raster, phases = self._render_pair(positions, probe_angles, wavelength, amplitude)
        frames.append(visual_cues(raster, 'recall', 'report'))
        phase_records.append(phases)
        metadata = dict(label=case['label'], sample_bit=case['sample_bit'],
            sample_assignment=case['sample'], probe_assignment=case['probe'],
            orientation_inventory_radians=angles.tolist(),
            sample_angles_radians=sample_angles.tolist(), probe_angles_radians=probe_angles.tolist(),
            separation_degrees=separation, signed_separation_degrees=signed_sep,
            shared_wavelength_pixels=wavelength, shared_amplitude=amplitude,
            positions_xy=positions, spacing=spacing, spacing_pixels=distance, location_grid=grid_name,
            phase_radians_by_frame=phase_records, label_semantics='0=preserve;1=swap',
            protocol='spatial_binding', post_delay=0, target_last_sample_frame=2,
            probe_frame=4+cfg['delay'], evidence_events=2, sample_presentations=2,
            cue_presentations=2, probe_presentations=1, report_presentations=1,
            blank_presentations=cfg['delay'], distractor_presentations=0,
            independent_base_episode=True)
        return np.stack(frames), case['label'], metadata

    def batch(self, n, family, condition):
        if family not in FAMILIES or int(n) <= 0: raise ValueError((n, family))
        cfg = normalize_condition(condition)
        if family == 'orientation_binding':
            rows = [self._binding(cfg) for _ in range(n)]
            x = torch.from_numpy(np.stack([r[0] for r in rows]))
            labels = torch.tensor([r[1] for r in rows], dtype=torch.long)
            metadata = [r[2] for r in rows]
        else:
            if family == 'orientation_single':
                native_cfg = setting(protocol='recall', load=1, delay=cfg['delay'], post_delay=0)
            else:
                native_cfg = setting(protocol='integration', length=8, delay=cfg['delay'])
            x, labels, metadata = self.native[family].batch(n, FAMILY_HEAD[family], native_cfg)
        for meta in metadata:
            ordinal = self.counters[family]
            trial_id = f'Spatial/{self.split}/{self.seed}/{family}/{ordinal}'
            meta.update(version=VERSION, family=family, task=FAMILY_HEAD[family],
                        split=self.split, trial_id=trial_id, paired_base_id=trial_id,
                        condition=cfg['name'], condition_config=copy.deepcopy(cfg),
                        frame_count=int(x.shape[1]), encoder_updates=int(x.shape[1]))
            self.counters[family] += 1
        return x, labels, metadata
