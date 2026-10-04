"""Bounded CPU render/replay/forward checks. Never optimizes or saves weights."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from PreAttentiveVision.natural_stimuli import DATA_ROOT
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline
from WorkingMemory.SpatialTaskBattery.stimuli import frame_count
from .suite import CATALOG, SuiteStream, task_classes


def verify(output, model_smoke=False):
    torch.set_num_threads(2)
    stream = SuiteStream('val')
    report: dict[str, Any] = dict(version=CATALOG['version'], timestamp=datetime.now(timezone.utc).isoformat(),
                  device='cpu', training_updates=0, expected_tasks=len(CATALOG['tasks']),
                  expected_cells=sum(len(t['conditions']) for t in CATALOG['tasks']),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   numpy=np.__version__), cells=[], model=None)
    manifest_path = Path(DATA_ROOT) / 'manifest.json'
    dataset_ready = manifest_path.is_file()
    report['dataset'] = dict(root=str(DATA_ROOT), status='present' if dataset_ready else 'missing')
    if dataset_ready:
        records = json.loads(manifest_path.read_text())['images']
        counts = Counter(r['split'] for r in records)
        assert dict(counts) == dict(train=200, val=100, test=200)
        assert len({r['base_id'] for r in records}) == len(records)
        assert len({r['sha256'] for r in records}) == len(records)
        assert all((Path(DATA_ROOT) / r['file']).is_file() for r in records)
        report['dataset'].update(split_counts=dict(counts), disjoint_source_ids=True,
                                 disjoint_source_hashes=True, all_files_present=True)
    model = None
    before = {}
    if model_smoke:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731099)
            model = AccumulatorBaseline(task_classes(), stack=3, center=True, accumulator='kda').cpu().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        report['model'] = dict(initialization='fresh_random_no_checkpoint',
                               parameters=sum(p.numel() for p in model.parameters()),
                               all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                               heads=sorted(model.heads), forward_tasks=[])
    for task in CATALOG['tasks']:
        longest = None
        for cell in task['conditions']:
            row = dict(task=task['id'], cell=cell['id'])
            if task['requires_bsds500'] and not dataset_ready:
                row.update(status='blocked_missing_dataset', reason='Prepare official BSDS500; no replacement images used')
                report['cells'].append(row)
                continue
            saved = stream.state_dict()
            x, y, metadata = stream.batch(4, task['id'], cell['id'])
            stream.load_state_dict(saved)
            xx, yy, mm = stream.batch(4, task['id'], cell['id'])
            assert torch.equal(x, xx) and torch.equal(y, yy) and metadata == mm
            native: Any = stream._make_stream(task['id'], cell['id'])
            if task['group'] == 'sensory':
                nx, ny, nm = native.batch(4, task['id'])
                expected_frames = 2
            else:
                nx, ny, nm = native.batch(4, task['id'], cell['kwargs'])
                expected_frames = 4 if task['id'] == 'orientation_ring' else frame_count(task['id'], cell['kwargs'])
            assert torch.equal(x, nx) and torch.equal(y, ny)
            assert all(all(m[k] == v for k, v in old.items()) for m, old in zip(metadata, nm))
            assert x.shape == (4, expected_frames, 3, 100, 100)
            assert x.dtype == torch.float32 and y.dtype == torch.int64
            assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
            assert y.min() >= 0 and y.max() < task['classes']
            ids = [m['suite_trial_id'] for m in metadata]
            assert len(set(ids)) == len(ids)
            if task['id'] == 'image_recognition':
                for sample, label, meta in zip(x, y, metadata):
                    assert (meta['probe_raster_sha256'] in meta['study_raster_sha256']) == bool(label)
                    probes = meta['probe_frames']
                    assert all(torch.equal(sample[probes[0]], sample[p]) for p in probes)
                    if label:
                        study = meta['study_frames'][meta['seen_study_index']]
                        assert torch.equal(sample[study], sample[probes[0]])
                    if cell['kwargs']['load'] == 0:
                        assert label == 0 and not cell['selection_eligible']
            if task['id'] == 'natural_spectrum':
                assert all(np.argmin(m['betas_by_frame']) == int(label) for m, label in zip(metadata, y))
            row.update(status='passed', frames=expected_frames, episodes=4,
                       class_counts=torch.bincount(y, minlength=task['classes']).tolist(),
                       exact_replay=True, native_equivalence=True,
                       selection_eligible=cell['selection_eligible'])
            report['cells'].append(row)
            if longest is None or x.shape[1] > longest.shape[1]:
                longest = x[:1].clone()
        if model is not None and longest is not None:
            with torch.inference_mode():
                logits = model(longest, task['id'])
            assert logits.shape == (1, task['classes']) and torch.isfinite(logits).all()
            report['model']['forward_tasks'].append(dict(task=task['id'], frames=longest.shape[1],
                                                        output_shape=list(logits.shape), finite=True))
    if model is not None:
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        report['model']['parameters_unchanged'] = True
    report['passed_cells'] = sum(c['status'] == 'passed' for c in report['cells'])
    report['blocked_cells'] = len(report['cells']) - report['passed_cells']
    report['status'] = 'passed' if report['blocked_cells'] == 0 else 'blocked_missing_dataset'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'checks')
    parser.add_argument('--model-smoke', action='store_true', help='CPU forward only, fresh KDA, no optimizer')
    args = parser.parse_args()
    result = verify(args.output, args.model_smoke)
    print(json.dumps({k: result[k] for k in ('status', 'expected_tasks', 'expected_cells', 'passed_cells', 'blocked_cells', 'training_updates')}, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 2)
