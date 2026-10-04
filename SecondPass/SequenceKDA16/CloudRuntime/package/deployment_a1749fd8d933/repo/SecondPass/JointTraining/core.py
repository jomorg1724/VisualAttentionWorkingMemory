"""Auditable primitives for the authorized, fresh-weight local joint run."""
from __future__ import annotations
import copy
import random
from SecondPass.TaskSuite.suite import CATALOG, TASKS


class BalancedScheduler:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.tasks = []
        self.cells = {t: [] for t in TASKS}
        self.updates = 0

    def next(self):
        if not self.tasks:
            self.tasks = list(TASKS)
            self.rng.shuffle(self.tasks)
        task = self.tasks.pop()
        if not self.cells[task]:
            self.cells[task] = [c['id'] for c in TASKS[task]['conditions']]
            self.rng.shuffle(self.cells[task])
        self.updates += 1
        return task, self.cells[task].pop()

    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.getstate(), tasks=self.tasks,
                                  cells=self.cells, updates=self.updates))

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])
        self.tasks = copy.deepcopy(state['tasks'])
        self.cells = copy.deepcopy(state['cells'])
        self.updates = state['updates']


def score_cell(task, cell, labels, probabilities, metadata):
    import json
    import numpy as np
    from WorkingMemory.BatteryAudit.observers import auc_binary
    y, p = np.asarray(labels), np.asarray(probabilities)
    k = TASKS[task]['classes']
    pred = p.argmax(1)
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    recalls = [float(confusion[i,i] / confusion[i].sum()) if confusion[i].sum() else None for i in range(k)]
    aucs = [auc_binary((y == i).astype(int), p[:,i]) for i in range(k)]
    aucs = [float(a) if np.isfinite(a) else None for a in aucs]
    empty = task == 'image_recognition' and cell.startswith('N0_')
    out = dict(task=task, cell=cell, n=len(y), confusion=confusion.tolist(),
               class_recall=None if empty else recalls,
               accuracy=None if empty else float((y == pred).mean()),
               balanced_accuracy=None if empty or None in recalls else float(np.mean(recalls)),
               auc=None if empty or None in aucs else float(np.mean(aucs)),
               auc_ovr=None if empty else aucs)
    if k == 2:
        out.update(specificity=recalls[0], false_positive_rate=None if recalls[0] is None else 1-recalls[0])
    if task == 'krauzlis_cued_motion':
        out['events'] = {}
        for event in ('target','foil','catch'):
            mask = np.array([m.get('event_type') == event for m in metadata])
            out['events'][event] = dict(n=int(mask.sum()), positive_rate=float((pred[mask] == 1).mean()) if mask.any() else None,
                target_side_counts={str(side):sum(bool(ok) and m.get('target_location') == side for ok,m in zip(mask,metadata)) for side in (0,1)})
        out['target_hit_rate'] = out['events']['target']['positive_rate']
        out['foil_false_alarm_rate'] = out['events']['foil']['positive_rate']
        out['catch_false_positive_rate'] = out['events']['catch']['positive_rate']
    out['strata'] = {}
    for field in TASKS[task]['report_by']:
        groups = {}
        for i,m in enumerate(metadata):
            value = m.get(field, m.get('study_load') if field == 'load' else m.get('condition', {}).get(field))
            key = json.dumps(value, sort_keys=True)
            groups.setdefault(key, []).append(i)
        out['strata'][field] = {key: dict(n=len(ix), accuracy=None if empty else float((pred[ix] == y[ix]).mean()),
            positive_rate=float((pred[ix] == 1).mean()) if k == 2 else None) for key,ix in groups.items()}
    return out


def cpu_tree(obj):
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def atomic_json(path, data):
    import json, os
    from pathlib import Path
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def append_jsonl(path, data):
    import json, os
    with open(path, 'a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def tree_equal(a, b):
    import torch, numpy as np
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a == b


def save_checkpoint(path, model, optimizer, scheduler, stream, state, device):
    import os, torch, numpy as np, hashlib
    from pathlib import Path
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rng = dict(cpu=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
               mps=torch.mps.get_rng_state() if str(device) == 'mps' else None)
    payload = cpu_tree(dict(schema=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), stream=stream.state_dict(), rng=rng, state=state))
    temp = path.with_suffix('.pt.tmp')
    with temp.open('wb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    loaded = torch.load(path, map_location='cpu')
    if not tree_equal(payload, loaded):
        raise RuntimeError('Checkpoint readback mismatch: ' + str(path))
    return dict(path=str(path.resolve()), verified=True, step=state['step'], bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def restore_checkpoint(path, model, optimizer, scheduler, stream, device):
    import torch, numpy as np
    saved = torch.load(path, map_location='cpu')
    if saved['schema'] != 1: raise ValueError('Checkpoint schema mismatch')
    stream.load_state_dict(saved['stream'])
    scheduler.load_state_dict(saved['scheduler'])
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng']['cpu'])
    np.random.set_state(saved['rng']['numpy']); random.setstate(saved['rng']['python'])
    if str(device) == 'mps': torch.mps.set_rng_state(saved['rng']['mps'])
    return saved['state']


def summarize(rows):
    def mean(values):
        return sum(values)/len(values) if values and all(v is not None for v in values) else None
    tasks = {}
    for task, spec in TASKS.items():
        eligible = {c['id'] for c in spec['conditions'] if c['selection_eligible']}
        selected = [r for r in rows if r['task'] == task and r['cell'] in eligible]
        complete = len(selected) == len(eligible)
        ba = mean([r['balanced_accuracy'] for r in selected]) if complete else None
        auc = mean([r['auc'] for r in selected]) if complete else None
        chance = 1/spec['classes']
        tasks[task] = dict(balanced_accuracy=ba, auc=auc, chance_normalized_ba=None if ba is None else (ba-chance)/(1-chance))
    normalized = [r['chance_normalized_ba'] for r in tasks.values()]
    auc = mean([r['auc'] for r in tasks.values()])
    key = [min(normalized), auc] if None not in normalized and auc is not None else None
    groups = {}
    for group in ('sensory','orientation_auxiliary','spatial'):
        names = [t for t in TASKS if TASKS[t]['group'] == group]
        groups[group] = {metric: mean([tasks[t][metric] for t in names]) for metric in ('balanced_accuracy','chance_normalized_ba','auc')}
    return dict(tasks=tasks, groups=groups, selection_key=key, equal_task_mean_auc=auc)


def may_update(*, now, deadline, reserve, estimate, step, max_steps, stopped=False):
    return not stopped and step < max_steps and now + reserve + estimate < deadline
