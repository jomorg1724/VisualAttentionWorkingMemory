"""Fresh two-frame change stimuli; no task metadata enters the network.

Natural images are official CIFAR-10 32x32 photographs resized to 100x100.
See task.md for the task's conveniences and limitations.
"""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
import torch

FAMILIES = ('gabors', 'dots', 'shapes', 'natural')
DIFFICULTIES = ('easy', 'medium', 'hard')
DATA_ROOT = Path(__file__).resolve().parent / 'data'
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'


def prepare_cifar(data_root=None):
    """Verify official archive and extract only six regular binary files safely."""
    root = Path(data_root or DATA_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'cifar-10-batches-bin'
    names = [f'data_batch_{i}.bin' for i in range(1, 6)] + ['test_batch.bin']
    if all((target / name).is_file() and (target / name).stat().st_size == 30730000 for name in names):
        return target
    archive = root / 'cifar-10-binary.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve(CIFAR_URL, archive)
    md5 = hashlib.md5()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            md5.update(block)
    if md5.hexdigest() != CIFAR_MD5:
        raise RuntimeError('Official CIFAR archive checksum mismatch; do not use it.')
    target.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for name in names:
            member = tar.getmember('cifar-10-batches-bin/' + name)
            if not member.isfile() or member.size != 30730000:
                raise RuntimeError('Unexpected CIFAR archive member')
            destination = (target / name).resolve()
            if destination.parent != target.resolve():
                raise RuntimeError('Unsafe archive path')
            with tar.extractfile(member) as source, destination.open('wb') as out:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
    (root / 'cifar_provenance.json').write_text(json.dumps({
        'url': CIFAR_URL, 'archive_md5': CIFAR_MD5,
        'official_description': 'https://www.cs.toronto.edu/~kriz/cifar.html',
        'native_size': [32, 32, 3], 'presented_size': [100, 100, 3],
        'resize': 'PIL bilinear', 'source_license': 'No explicit license grant stated on official dataset page; local research use, no redistribution.',
        'split': {'train': 'official train indices [0,45000)', 'val': 'official train indices [45000,50000)', 'test': 'official test indices [0,10000)'}
    }, indent=2), encoding='utf-8')
    return target


class PairStream:
    """Deterministic seed-local procedural stream, returning CPU Torch tensors.

    Labels are balanced within each even-sized batch and randomized in order.
    Families and difficulty are uniform unless explicitly fixed. All operations
    are defined before adding identical-distribution frame-specific nuisance.
    """
    def __init__(self, seed, split='train', data_root=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.rng = np.random.default_rng(seed)
        self.seed, self.split = int(seed), split
        self.root = Path(data_root or DATA_ROOT)
        self._cifar = {}
        yy, xx = np.mgrid[-10:11, -10:11].astype(np.float32)
        self.gx, self.gy = xx, yy
        self.envelope = np.exp(-(xx * xx + yy * yy) / (2 * 5.0 ** 2))
        self.counter = 0

    def state_dict(self):
        return {'rng': copy.deepcopy(self.rng.bit_generator.state),
                'counter': self.counter, 'seed': self.seed, 'split': self.split}

    def load_state_dict(self, state):
        if state['seed'] != self.seed or state['split'] != self.split:
            raise ValueError('Stream seed/split mismatch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.counter = state['counter']

    def _noise(self, image):
        # Nuisance law does not depend on changed label, family or difficulty.
        gain = self.rng.uniform(0.97, 1.03)
        offset = self.rng.uniform(-0.01, 0.01)
        noise = self.rng.normal(0, 0.025, image.shape).astype(np.float32)
        return np.clip(image * gain + offset + noise, 0, 1)

    def _gabors(self, changed, difficulty):
        angles = self.rng.uniform(0, np.pi, 9)
        contrasts = self.rng.uniform(0.22, 0.4, 9)
        phases = self.rng.uniform(0, 2 * np.pi, 9)
        index = int(self.rng.integers(9))
        degrees = float(self.rng.uniform(*{'easy': (35, 65), 'medium': (15, 35), 'hard': (5, 15)}[difficulty]))
        other = angles.copy()
        if changed:
            other[index] += np.deg2rad(degrees) * self.rng.choice([-1, 1])
        def render(theta):
            canvas = np.full((100, 100), 0.5, np.float32)
            for i, a in enumerate(theta):
                cy, cx = 20 + 30 * (i // 3), 20 + 30 * (i % 3)
                carrier = np.cos((self.gx * np.cos(a) + self.gy * np.sin(a)) * (2 * np.pi / 7) + phases[i])
                canvas[cy-10:cy+11, cx-10:cx+11] += contrasts[i] * self.envelope * carrier
            return np.repeat(canvas[..., None], 3, -1)
        return render(angles), render(other), {'operation': 'one_gabor_orientation', 'magnitude': degrees if changed else 0., 'magnitude_unit': 'degrees', 'base_id': None}

    def _dots(self, changed, difficulty):
        # Jittered 4x4 positions keep dots separable and avoid border clipping.
        centers = np.array([(x, y) for y in (18, 39, 60, 81) for x in (18, 39, 60, 81)], np.float32)
        centers += self.rng.uniform(-3, 3, centers.shape)
        colors = self.rng.uniform(0.4, 0.95, (16, 3))
        index = int(self.rng.integers(16))
        distance = float(self.rng.uniform(*{'easy': (6, 9), 'medium': (3, 6), 'hard': (1.5, 3)}[difficulty]))
        angle = self.rng.uniform(0, 2 * np.pi)
        other = centers.copy()
        if changed:
            other[index] += distance * np.array([np.cos(angle), np.sin(angle)])
        yy, xx = np.mgrid[:100, :100]
        def render(pos):
            im = np.full((100, 100, 3), 0.08, np.float32)
            for (x, y), color in zip(pos, colors):
                dot = np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 1.8**2)).astype(np.float32)
                im += dot[..., None] * (color - 0.08)
            return np.clip(im, 0, 1)
        return render(centers), render(other), {'operation': 'one_dot_displacement', 'magnitude': distance if changed else 0., 'magnitude_unit': 'pixels', 'base_id': None}

    def _shapes(self, changed, difficulty):
        positions = [(x, y) for y in (22, 50, 78) for x in (22, 50, 78)]
        colors = self.rng.uniform(0.2, 0.9, (9, 3))
        kinds = self.rng.integers(0, 3, 9)
        index = int(self.rng.integers(9))
        op = str(self.rng.choice(['color', 'shape']))
        size = {'easy': 9, 'medium': 7, 'hard': 5}[difficulty]
        delta = float(self.rng.uniform(*{'easy': (.35, .5), 'medium': (.2, .35), 'hard': (.09, .2)}[difficulty]))
        colors2, kinds2 = colors.copy(), kinds.copy()
        if changed and op == 'shape':
            kinds2[index] = (kinds[index] + int(self.rng.integers(1, 3))) % 3
        elif changed:
            channel = int(self.rng.integers(3))
            sign = -1 if colors[index, channel] > .55 else 1
            colors2[index, channel] = np.clip(colors[index, channel] + sign * delta, .05, .95)
            delta = float(abs(colors2[index, channel] - colors[index, channel]))
        def render(cs, ks):
            im = Image.new('RGB', (100, 100), (32, 32, 32))
            draw = ImageDraw.Draw(im)
            for (x, y), color, kind in zip(positions, cs, ks):
                fill = tuple((color * 255).astype(int))
                box = (x-size, y-size, x+size, y+size)
                if kind == 0: draw.ellipse(box, fill=fill)
                elif kind == 1: draw.rectangle(box, fill=fill)
                else: draw.polygon([(x,y-size),(x+size,y+size),(x-size,y+size)], fill=fill)
            return np.asarray(im, dtype=np.float32) / 255.
        return render(colors, kinds), render(colors2, kinds2), {'operation': 'one_object_' + op, 'magnitude': (delta if op == 'color' else float(size)) if changed else 0., 'magnitude_unit': 'channel_intensity' if op == 'color' else 'shape_radius_pixels', 'base_id': None}

    def _natural(self, changed, difficulty):
        if self.split == 'test':
            idx = int(self.rng.integers(10000)); name = 'test_batch.bin'; local = idx; base = f'cifar10/test/{idx}'
        else:
            low, high = (0, 45000) if self.split == 'train' else (45000, 50000)
            idx = int(self.rng.integers(low, high)); name = f'data_batch_{idx//10000+1}.bin'; local = idx % 10000; base = f'cifar10/train/{idx}'
        if name not in self._cifar:
            path = self.root / 'cifar-10-batches-bin' / name
            if not path.exists():
                raise FileNotFoundError(f'{path}: run stimuli.py --prepare once before training')
            self._cifar[name] = np.memmap(path, mode='r', dtype=np.uint8, shape=(10000, 3073))
        record = self._cifar[name][local]
        rgb = record[1:].reshape(3, 32, 32).transpose(1, 2, 0)
        a = np.asarray(Image.fromarray(rgb).resize((100, 100), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        b = a.copy()
        op = str(self.rng.choice(['local_color', 'local_occlusion', 'translation']))
        size = int(self.rng.integers(*{'easy': (20, 31), 'medium': (12, 21), 'hard': (6, 13)}[difficulty]))
        y, x = self.rng.integers(5, 95-size, 2)
        magnitude = float(size)
        if op == 'translation':
            shift = int(self.rng.choice({'easy': [5,6,7], 'medium': [3,4], 'hard': [1,2]}[difficulty]))
            dx, dy = self.rng.choice([-1, 1], 2) * shift
            if changed:
                # Reflection avoids a changed-only black border cue.
                pad = np.pad(a, ((8,8),(8,8),(0,0)), mode='reflect')
                b = pad[8+dy:108+dy, 8+dx:108+dx].copy()
            magnitude = float(np.hypot(dx,dy))
        elif op == 'local_color':
            channel = int(self.rng.integers(3))
            patch = b[y:y+size,x:x+size,channel]
            sign = -1 if patch.mean() > .5 else 1
            if changed: patch[:] = np.clip(patch + sign * .3, 0, 1)
        elif changed:
            patch = b[y:y+size,x:x+size]
            # Contrast to local mean prevents an invisible zero-amplitude edit.
            color = np.where(patch.mean((0,1)) > .5, .1, .9)
            patch[:] = color
        return a, b, {'operation': op, 'magnitude': magnitude if changed else 0., 'magnitude_unit': 'pixels', 'base_id': base}

    def batch(self, n, family=None, difficulty=None):
        if family is not None and family not in FAMILIES: raise ValueError(family)
        if difficulty is not None and difficulty not in DIFFICULTIES: raise ValueError(difficulty)
        labels = np.arange(n, dtype=np.int64) % 2
        if n % 2 and self.rng.random() < .5: labels = 1 - labels
        self.rng.shuffle(labels)
        result = np.empty((n, 2, 3, 100, 100), np.float32)
        metadata = []
        for i, label in enumerate(labels):
            fam = family or str(self.rng.choice(FAMILIES))
            diff = difficulty or str(self.rng.choice(DIFFICULTIES))
            # Draw both variants regardless of label. Every observed frame has
            # the same A/B marginal in each label class, including edit artifacts.
            a, b, meta = getattr(self, '_' + fam)(True, diff)
            first = int(self.rng.integers(2))
            second = 1 - first if label else first
            variants = (a, b)
            result[i, 0] = self._noise(variants[first]).transpose(2, 0, 1)
            result[i, 1] = self._noise(variants[second]).transpose(2, 0, 1)
            metadata.append(dict(meta, family=fam, difficulty=diff, split=self.split,
                                 trial_id=f'{self.split}/{self.seed}/{self.counter}', changed=int(label),
                                 variant_pair=[first, second],
                                 observed_magnitude=meta['magnitude'] if label else 0.))
            self.counter += 1
        return torch.from_numpy(result), torch.from_numpy(labels), metadata


def write_examples(output=None):
    output = Path(output or Path(__file__).resolve().parent / 'stimulus_examples.png')
    stream = PairStream(2026091201, 'val')
    canvas = Image.new('RGB', (630, 500), 'white')
    draw = ImageDraw.Draw(canvas)
    records = []
    for row, family in enumerate(FAMILIES):
        images, labels, metadata = stream.batch(2, family=family, difficulty='medium')
        draw.text((8, row*125+5), family + ' (medium)', fill='black')
        for col in range(2):
            for frame in range(2):
                im = Image.fromarray((images[col,frame].numpy().transpose(1,2,0)*255).astype(np.uint8))
                canvas.paste(im, (150+col*240+frame*105, row*125+20))
            draw.text((150+col*240, row*125+5), 'change' if labels[col] else 'no change', fill='black')
        records.extend(metadata)
    canvas.save(output)
    output.with_suffix('.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--examples', action='store_true')
    args = parser.parse_args()
    if args.prepare: print(prepare_cifar())
    if args.examples: write_examples()
