"""Thin, task/condition-local adapters for existing renderers. No training code."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from PreAttentiveVision.neuroscience_stimuli import TaskStream
from PreAttentiveVision.natural_stimuli import DATA_ROOT, sha256
from WorkingMemory.PlainBaseline.variants import VariantStream
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

CATALOG = json.loads(Path(__file__).with_name('catalog.json').read_text())
TASKS = {task['id']: task for task in CATALOG['tasks']}


def task_classes():
    """Distinct output heads; interval labels are not change/no-change labels."""
    return {task: spec['classes'] for task, spec in TASKS.items()}


class SuiteStream:
    """One native stream per task/condition, with no padding or metadata inputs.

    Split seeds are new, not historical test seeds. Cell streams are independent
    of sampling in any other cell. Callers choose the cell explicitly; this is
    not a curriculum or optimizer schedule.
    """
    def __init__(self, split='train'):
        if split not in CATALOG['split_seeds']:
            raise ValueError(f'Unknown split: {split}')
        self.split = split
        self._streams = {}

    def state_dict(self):
        uses_photos = any(TASKS[task]['requires_bsds500'] for task, _ in self._streams)
        return copy.deepcopy(dict(
            catalog=CATALOG, split=self.split,
            dataset_manifest_sha256=sha256(Path(DATA_ROOT) / 'manifest.json') if uses_photos else None,
            streams=[dict(task=task, cell=cell, state=stream.state_dict())
                     for (task, cell), stream in sorted(self._streams.items())]))

    def load_state_dict(self, state):
        if state.get('catalog') != CATALOG or state.get('split') != self.split:
            raise ValueError('Suite catalog/split mismatch')
        uses_photos = any(TASKS[row['task']]['requires_bsds500'] for row in state['streams'])
        identity = state.get('dataset_manifest_sha256')
        if uses_photos and (not identity or identity != sha256(Path(DATA_ROOT) / 'manifest.json')):
            raise ValueError('Photo dataset manifest mismatch')
        restored = {}
        for row in state['streams']:
            key = (row['task'], row['cell'])
            if key in restored:
                raise ValueError('Duplicate cell state')
            native = self._make_stream(*key)
            native.load_state_dict(copy.deepcopy(row['state']))
            restored[key] = native
        # Commit only after every cell validates, discarding post-snapshot cells.
        self._streams = restored

    def _cell(self, task, cell):
        if task not in TASKS:
            raise ValueError(f'Unknown task: {task}')
        for index, condition in enumerate(TASKS[task]['conditions']):
            if condition['id'] == cell:
                return index, condition
        raise ValueError(f'Unknown cell for {task}: {cell}')

    def stream_seed(self, task, cell):
        index, _ = self._cell(task, cell)
        # Non-overlapping split/task/cell namespaces; never Python hash().
        split = CATALOG['split_seeds'][self.split]
        return split * 100000 + TASKS[task]['stream_id'] * 1000 + index

    def _make_stream(self, task, cell):
        seed = self.stream_seed(task, cell)
        if TASKS[task]['group'] == 'sensory':
            return TaskStream(seed, self.split)
        if task == 'orientation_ring':
            return VariantStream(seed, self.split)
        return SpatialBatteryStream(seed, self.split)

    def batch(self, n, task, cell):
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError('Batch size must be a positive integer')
        _, condition = self._cell(task, cell)
        key = (task, cell)
        if key not in self._streams:
            self._streams[key] = self._make_stream(task, cell)
        native = self._streams[key]
        if TASKS[task]['group'] == 'sensory':
            images, labels, metadata = native.batch(n, task)
        else:
            images, labels, metadata = native.batch(n, task, condition['kwargs'])
        # Preserve every native metadata field. These records are analysis-only.
        metadata = [dict(row, suite_version=CATALOG['version'], suite_task=task,
                         suite_cell=cell, suite_trial_id=f'{task}/{cell}/{row["trial_id"]}')
                    for row in metadata]
        return images, labels, metadata
