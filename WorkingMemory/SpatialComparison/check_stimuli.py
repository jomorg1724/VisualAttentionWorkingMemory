"""One focused CPU check of generator contracts and visual binding stimuli."""
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from WorkingMemory.SpatialComparison.stimuli import (
    SpatialStream, FAMILIES, DELAYS, counterbalance_cases, LOCATION_GRIDS, PATCH_RADIUS)


def run():
    start = time.time()
    checks = {}
    for family in FAMILIES:
        stream = SpatialStream(847211, 'test')
        state = stream.state_dict()
        reference = None
        for delay in DELAYS:
            stream.load_state_dict(state)
            x, y, meta = stream.batch(4, family, {'delay': delay})
            assert x.shape == (4, (11 if family == 'motion_direction' else 5)+delay, 3, 100, 100)
            assert x.dtype == torch.float32 and y.dtype == torch.long
            assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
            stable = torch.cat([x[:, :10], x[:, -1:]], 1) if family == 'motion_direction' else torch.cat([x[:, :3], x[:, -2:]], 1)
            if reference is None: reference = (stable, y, [m['trial_id'] for m in meta])
            else:
                assert torch.equal(stable, reference[0]) and torch.equal(y, reference[1])
                assert [m['trial_id'] for m in meta] == reference[2]
        stream.load_state_dict(state)
        a = stream.batch(3, family, 4)
        stream.load_state_dict(state)
        b = stream.batch(3, family, 4)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
        checks[family] = {'shape_range_dtype': True, 'full_rng_replay': True,
                          'exact_evidence_and_probe_across_delays': True}
    cases = counterbalance_cases()
    for frame in ('sample', 'probe'):
        for location in (0, 1):
            assert sorted(c[frame][location] for c in cases if c['label'] == 0) == [0, 1]
            assert sorted(c[frame][location] for c in cases if c['label'] == 1) == [0, 1]
    for c in cases:
        assert sorted(c['sample']) == sorted(c['probe']) == [0, 1]
        assert (c['sample'] != c['probe']) == bool(c['label'])
    s = SpatialStream(88347)
    x, y, meta = s.batch(8, 'orientation_binding', 0)
    for block in (meta[:4], meta[4:]):
        assert sorted((m['sample_bit'], m['label']) for m in block) == [(0,0),(0,1),(1,0),(1,1)]
    for i, m in enumerate(meta):
        assert not torch.equal(x[i, 1, :, 15:85, 15:85], x[i, 2, :, 15:85, 15:85])
        assert len(set(sum(m['phase_radians_by_frame'], []))) == 6
    checks['binding_design'] = {'exact_four_case_balance': True,
        'label_independent_sample_probe_assignment_marginals': True,
        'identical_orientation_inventory': True, 'fresh_phase_and_noise': True,
        'raster_marginals': 'Equal in distribution, not duplicated raster quartets.'}
    position_sets = []
    for name, grid in LOCATION_GRIDS.items():
        points = {(mid+sign*d/2, y) for mid in grid['midpoint_x'] for y in grid['y']
                  for d in (26.,46.) for sign in (-1,1)}
        assert all(10 < x-PATCH_RADIUS and x+PATCH_RADIUS < 90 and
                   10 < y-PATCH_RADIUS and y+PATCH_RADIUS < 90 for x,y in points)
        position_sets.append(points)
    assert position_sets[0].isdisjoint(position_sets[1])
    checks['locations'] = {'train_and_heldout_centers_disjoint': True,
                          'patches_outside_cue_borders': True,
                          'pair_supports_nonoverlapping': 26 > 2*PATCH_RADIUS}
    a, b = SpatialStream(71928), SpatialStream(71928)
    a.batch(4, 'orientation_binding', 0)
    a.batch(4, 'motion_direction', 0)
    xa, ya, ma = a.batch(4, 'orientation_single', 0)
    xb, yb, mb = b.batch(4, 'orientation_single', 0)
    assert torch.equal(xa, xb) and torch.equal(ya, yb) and ma == mb
    checks['task_local_rng_independence'] = True

    root = Path(__file__).resolve().parent
    fig, axes = plt.subplots(4, 3, figsize=(9, 11), facecolor='#f7f7f7')
    stream = SpatialStream(678231, 'test')
    for row, (spacing, label) in enumerate([('near',0),('near',1),('far',0),('far',1)]):
        images, labels, metadata = stream.batch(4, 'orientation_binding', {'delay':0, 'spacing':spacing})
        index = labels.tolist().index(label)
        m = metadata[index]
        for col, frame in enumerate((1,2,4)):
            axes[row,col].imshow(images[index,frame].permute(1,2,0).numpy(), vmin=0, vmax=1)
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row == 0: axes[row,col].set_title(('Sample frame 1','Sample frame 2','Probe / report')[col])
        axes[row,0].set_ylabel(f"{spacing.title()} · {'preserve' if label == 0 else 'swap'}\n{m['separation_degrees']}° separation")
    fig.suptitle('Spatial orientation binding: independent phases, preserved item inventory', fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(root/'stimulus_contact_sheet.png', dpi=140)
    plt.close(fig)
    result = dict(status='passed', device='CPU only; no model inference or training',
                  elapsed_seconds=time.time()-start, checks=checks,
                  source_sha256=hashlib.sha256((root/'stimuli.py').read_bytes()).hexdigest(),
                  contact_sheet=str(root/'stimulus_contact_sheet.png'))
    (root/'stimulus_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': run()
