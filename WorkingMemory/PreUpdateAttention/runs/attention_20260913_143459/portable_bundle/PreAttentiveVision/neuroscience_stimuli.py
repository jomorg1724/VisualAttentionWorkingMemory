"""Version 2: two-frame cardinal random-dot motion, inspired by Lovejoy2010.

New task, not a modification of the version-1 change-detection stimuli.
See cardinal_motion.md for reported parameters versus raster adaptations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import torch

TASK_CLASSES = {'motion_direction': 4, 'orientation': 2, 'contrast': 2,
                'spatial_frequency': 2, 'chromatic_increment': 2,
                'contour': 2, 'natural_spectrum': 2}
DIRECTION_NAMES = ('right', 'up', 'left', 'down')
# Image coordinates: x rightward, y downward.
DIRECTION_VECTORS = np.array(((1, 0), (0, -1), (-1, 0), (0, 1)), np.float32)
MOTION_PROTOCOL = {
    'version': 'cardinal_motion_v2', 'shape': [100, 100, 3],
    'aperture_diameter_pixels': 85., 'domain_dots': 256,
    'raster_pixels_per_degree': 10., 'displacements_pixels': [1., 2., 3.],
    'dot_sigma_pixels': .7, 'dot_lifetime_frames': 2,
    'survivor_coherence': 1., 'background': .2, 'dot_increment': .7,
    'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3412590/',
}


class CardinalMotionStream:
    """CPU-only seed-local stream; balanced labels within multiples of four.

    The entire square domain is periodic and stationary. A fixed circular mask
    selects the visible patch after rendering. Half the dot identities survive
    across the two frames, and the others are reborn at independent positions.
    All surviving dots move in the selected cardinal direction. Frame marginals
    contain no direction cue in distribution; direction requires the pair.
    """
    def __init__(self, seed, split='train'):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.seed, self.split = int(seed), split
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        yy, xx = np.mgrid[:100, :100]
        self.mask = (xx - 49.5) ** 2 + (yy - 49.5) ** 2 <= 42.5 ** 2

    def state_dict(self):
        return {'rng': copy.deepcopy(self.rng.bit_generator.state), 'counter': self.counter,
                'seed': self.seed, 'split': self.split, 'protocol': MOTION_PROTOCOL}

    def load_state_dict(self, state):
        if (state['seed'], state['split'], state['protocol']) != (self.seed, self.split, MOTION_PROTOCOL):
            raise ValueError('Motion stream identity/protocol mismatch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.counter = state['counter']

    def _render(self, positions):
        field = np.zeros((100, 100), np.float32)
        lo = np.floor(positions).astype(np.int64)
        frac = positions - lo
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            weight = (frac[:, 0] if dx else 1-frac[:, 0]) * (frac[:, 1] if dy else 1-frac[:, 1])
            np.add.at(field, ((lo[:, 1]+dy) % 100, (lo[:, 0]+dx) % 100), weight)
        field = gaussian_filter(field, sigma=.7, mode='wrap') * (2*np.pi*.7**2)
        image = np.full((100, 100), .2, np.float32)
        image[self.mask] += .7 * field[self.mask]
        return np.repeat(np.clip(image, 0, 1)[None, :, :], 3, axis=0)

    def batch(self, n, task='motion_direction', displacement=None):
        if task != 'motion_direction':
            raise ValueError(task)
        if displacement is not None and float(displacement) not in (1., 2., 3.):
            raise ValueError('Version-2 displacement must be 1, 2 or 3 pixels')
        labels = (np.arange(n, dtype=np.int64) + int(self.rng.integers(4))) % 4
        self.rng.shuffle(labels)
        result = np.empty((n, 2, 3, 100, 100), np.float32)
        metadata = []
        for i, label in enumerate(labels):
            step = float(displacement if displacement is not None else self.rng.choice([1., 2., 3.]))
            before = self.rng.uniform(0, 100, (256, 2)).astype(np.float32)
            ages = np.zeros(256, np.int64)
            ages[self.rng.permutation(256)[:128]] = 1
            survivor = ages == 0
            after = (before + step * DIRECTION_VECTORS[label]) % 100
            after[~survivor] = self.rng.uniform(0, 100, (128, 2))
            result[i, 0], result[i, 1] = self._render(before), self._render(after)
            metadata.append({
                'task': 'motion_direction', 'family': 'motion_direction', 'protocol': MOTION_PROTOCOL['version'],
                'direction': DIRECTION_NAMES[label], 'label': int(label),
                'displacement_pixels': step, 'difficulty': f'displacement_{int(step)}px',
                'coherence': 1., 'coherence_definition': 'fraction of surviving dots following the selected direction',
                'survivor_count_domain': 128, 'reborn_count_domain': 128,
                'base_id': None, 'split': self.split,
                'trial_id': f'{self.split}/{self.seed}/{self.counter}',
            })
            self.counter += 1
        return torch.from_numpy(result), torch.from_numpy(labels), metadata


BATTERY_PROTOCOL = 'seven_sensory_tasks_v3'


class TaskStream(CardinalMotionStream):
    """Shared stream for the ordered seven-task sensory battery.

    Binary interval labels identify frame0 or frame1 containing the larger
    quantity / aligned contour. Orientation labels0counterclockwise1clockwise.
    """
    def __init__(self, seed, split='train'):
        super().__init__(seed, split)
        self._natural = None
        yy, xx = np.mgrid[:100, :100].astype(np.float32)
        self.xx, self.yy = xx-49.5, yy-49.5
        self.envelope = np.exp(-(self.xx**2+self.yy**2)/(2*19.**2))
        gy, gx = np.mgrid[-5:6, -5:6].astype(np.float32)
        self.gx, self.gy = gx, gy
        self.small_envelope = np.exp(-(gx**2+gy**2)/(2*2.**2))

    def _get_natural(self):
        if self._natural is None:
            from PreAttentiveVision.natural_stimuli import NaturalSpectrum
            self._natural = NaturalSpectrum(self.split)
        return self._natural

    def state_dict(self):
        state = super().state_dict()
        state['battery_protocol'] = BATTERY_PROTOCOL
        state['natural_identity'] = None if self._natural is None else self._natural.identity
        return state

    def load_state_dict(self, state):
        if state.get('battery_protocol') != BATTERY_PROTOCOL:
            raise ValueError('Battery protocol mismatch')
        if state.get('natural_identity') is not None and state['natural_identity'] != self._get_natural().identity:
            raise ValueError('Natural source/protocol mismatch')
        super().load_state_dict(state)

    def _grating(self, theta, frequency, phase):
        carrier = np.cos(2*np.pi*frequency*(self.xx*np.cos(theta)+self.yy*np.sin(theta))/100+phase)
        image = self.envelope * carrier
        image -= image.mean()
        return image / np.max(np.abs(image))

    def _achromatic(self, field):
        # Same nuisance distribution for bothframes, labels and difficultybins.
        image = .5 + field + self.rng.uniform(-.004, .004, field.shape)
        if image.min() < 0 or image.max() > 1:
            raise RuntimeError('Stimulus intensity outside gamut')
        return np.repeat(image[None,:,:], 3, axis=0).astype(np.float32)

    def _orientation(self, label, level):
        degrees = (4., 10., 22.)[level]
        theta = float(self.rng.uniform(0, np.pi))
        signed = np.deg2rad(degrees) * (1 if label else -1)
        frequency = float(self.rng.uniform(5, 10))
        contrast = float(self.rng.uniform(.2, .38))
        phases = self.rng.uniform(0, 2*np.pi, 2)
        fields = [contrast*self._grating(theta,frequency,phases[0]),
                  contrast*self._grating(theta+signed,frequency,phases[1])]
        return np.stack([self._achromatic(f) for f in fields]), {
            'signed_orientation_degrees': degrees*(1 if label else -1),
            'base_orientation_degrees': float(np.rad2deg(theta)),
            'frequency_cycles_per_image': frequency, 'contrast':contrast,
            'difficulty': f'delta_{int(degrees)}deg',
        }

    def _contrast(self, label, level):
        increment = (.025, .06, .13)[level]
        pedestal = float(self.rng.choice([.08, .18, .3]))
        frequency = float(self.rng.uniform(5,10))
        theta, phase = self.rng.uniform(0,np.pi), self.rng.uniform(0,2*np.pi)
        pattern = self._grating(theta, frequency, phase)
        contrasts = [pedestal,pedestal]
        contrasts[label] += increment
        return np.stack([self._achromatic(c*pattern) for c in contrasts]), {
            'contrast_increment':increment, 'pedestal':pedestal,
            'frame_contrasts':contrasts, 'frequency_cycles_per_image':frequency,
            'difficulty':f'increment_{increment:g}_pedestal_{pedestal:g}',
        }

    def _spatial_frequency(self, label, level):
        octave = (.08,.18,.35)[level]
        frequency = float(self.rng.uniform(4,9))
        frequencies = [frequency,frequency]
        frequencies[label] *= 2**octave
        theta = self.rng.uniform(0,np.pi)
        contrast = self.rng.uniform(.2,.38)
        phases = self.rng.uniform(0,2*np.pi,2)
        return np.stack([self._achromatic(contrast*self._grating(theta,f,phase))
                         for f,phase in zip(frequencies,phases)]), {
            'frequency_octave_increment':octave, 'frame_frequencies':frequencies,
            'difficulty':f'octave_{octave:g}',
        }

    def _chromatic_increment(self, label, level):
        w = np.array([.2126,.7152,.0722], np.float32)
        axis = np.array([w[1],-w[0],0],np.float32)
        axis /= np.linalg.norm(axis)
        increment = (.018,.045,.1)[level]
        base = self.rng.uniform(.3,.65,3).astype(np.float32)
        colors = [base.copy(),base.copy()]
        colors[label] += increment*axis
        # Soft equal-geometry patch; linearRGB, no gamma or clamping.
        radius = self.rng.uniform(17,24)
        distance = np.sqrt(self.xx**2+self.yy**2)
        mask = np.clip((radius+.5-distance)/1.,0,1)
        result = []
        for color in colors:
            image = .5 + mask[:,:,None]*(color-.5)[None,None,:]
            image += self.rng.uniform(-.001,.001,image.shape)
            result.append(image.transpose(2,0,1).astype(np.float32))
        return np.stack(result), {
            'chromatic_increment':increment, 'axis_linear_rgb':axis.tolist(),
            'frame_colors': [c.tolist() for c in colors],
            'frame_luminances': [float(w@c) for c in colors],
            'difficulty':f'increment_{increment:g}',
        }

    def _contour(self, label, level):
        jitter = (2.,8.,16.)[level]
        bend = self.rng.uniform(-9,9)
        t = np.linspace(-1,1,7)
        path = np.stack([31*t,bend*t*t],axis=1)
        angle = self.rng.uniform(0,2*np.pi)
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        center = 49.5+self.rng.uniform(-3,3,2)
        path = path@rotation.T+center
        positions = [p for p in path]
        for _ in range(2000):
            if len(positions) == 32: break
            p = self.rng.uniform(8,92,2)
            if np.min(np.linalg.norm(np.array(positions)-p,axis=1)) >= 10:
                positions.append(p)
        if len(positions) != 32:
            raise RuntimeError('Unable to place contour clutter')
        positions = np.rint(positions).astype(int)
        tangent = np.arctan2(2*bend*t,31)+angle
        orientations = self.rng.uniform(0,np.pi,32)
        orientations[:7] = tangent + np.pi/2 + np.deg2rad(self.rng.normal(0,jitter,7))
        permutation = self.rng.permutation(32)
        scrambled = orientations[permutation]
        phases = self.rng.uniform(0,2*np.pi,32)
        def render(angles):
            field = np.zeros((100,100),np.float32)
            for (x,y),theta,phase in zip(positions,angles,phases):
                carrier = np.cos(2*np.pi*(self.gx*np.cos(theta)+self.gy*np.sin(theta))/5+phase)
                field[y-5:y+6,x-5:x+6] += .34*self.small_envelope*carrier
            return self._achromatic(field)
        structured,control = render(orientations),render(scrambled)
        result = [control,control]
        result[label] = structured
        return np.stack(result), {
            'alignment_jitter_degrees':jitter, 'path_bend_pixels':float(bend),
            'path_elements':7, 'total_elements':32, 'orientation_multiset_matched':True,
            'difficulty':f'jitter_{int(jitter)}deg',
        }

    def batch(self, n, task='motion_direction', displacement=None):
        if task == 'motion_direction':
            return super().batch(n,task,displacement)
        if task not in TASK_CLASSES:
            raise ValueError(task)
        labels = (np.arange(n,dtype=np.int64)+int(self.rng.integers(2)))%2
        self.rng.shuffle(labels)
        frames = np.empty((n,2,3,100,100),np.float32)
        metadata = []
        for i,label in enumerate(labels):
            if task == 'natural_spectrum':
                frames[i],meta = self._get_natural().sample(self.rng,int(label))
            else:
                level = int(self.rng.integers(3))
                frames[i],meta = getattr(self,'_'+task)(int(label),level)
            metadata.append(dict(meta,task=task,family=task,label=int(label),
                                 protocol=BATTERY_PROTOCOL,split=self.split,
                                 base_id=meta.get('base_id'),
                                 trial_id=f'{self.split}/{self.seed}/{self.counter}'))
            self.counter += 1
        return torch.from_numpy(frames),torch.from_numpy(labels),metadata


def preview(output=None):
    output = Path(output or Path(__file__).parent / 'cardinal_motion_examples.png')
    stream = CardinalMotionStream(2026091202, 'val')
    frames, labels, metadata = stream.batch(4, displacement=2)
    records = []
    canvas = Image.new('RGB', (445, 235), 'white')
    draw = ImageDraw.Draw(canvas)
    gif = [Image.new('RGB', (445, 125), 'white') for _ in range(2)]
    for label in range(4):
        idx = int(torch.where(labels == label)[0][0])
        draw.text((label*110+5, 4), DIRECTION_NAMES[label], fill='black')
        records.append(metadata[idx])
        for t in range(2):
            im = Image.fromarray((frames[idx,t].numpy().transpose(1,2,0)*255).astype(np.uint8))
            canvas.paste(im, (label*110+5, 20+t*105))
            gif[t].paste(im, (label*110+5, 20))
            ImageDraw.Draw(gif[t]).text((label*110+5, 4), DIRECTION_NAMES[label], fill='black')
    canvas.save(output)
    # This is one two-frame transition plus blank; no reversing loop that would
    # misleadingly show opposite motion on every other transition.
    blank = Image.new('RGB', (445, 125), 'white')
    gif[0].save(output.with_suffix('.gif'), save_all=True, append_images=[gif[1], blank], duration=[300,300,600], loop=0)
    output.with_suffix('.json').write_text(json.dumps({'protocol': MOTION_PROTOCOL, 'examples': records, 'preview_timing': '300ms each image plus600ms blank; display-only slowed preview, not a scientific refresh setting'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    preview()
