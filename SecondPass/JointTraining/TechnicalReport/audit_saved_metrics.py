"""Read-only saved-artifact audit; CPU construction only, no model forward/evaluation.
Run from repo root with OMP/OPENBLAS/MKL/VECLIB/NUMEXPR threads all set to 1.
Refuses to overwrite the timestamped snapshot; verify existing with --verify.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'SecondPass/JointTraining/runs'
V1 = RUNS / 'fresh_kda_joint_01'
V2 = RUNS / 'fresh_kda_joint_01_continuation_v2'
V3 = RUNS / 'fresh_kda_joint_01_continuation_v3_8h'
sys.path.insert(0, str(ROOT))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=sha(raw), bytes=len(raw), data=json.loads(raw))

def verify(s):
    expected = {(t['id'], c['id']) for t in s['catalog']['data']['tasks'] for c in t['conditions']}
    assert len(expected) == 35
    assert len(s['catalog']['data']['tasks']) == 13
    for e in s['evaluations']:
        d = e['result']; rows = d['cells']
        assert d['complete'] and d['complete_cells'] == d['expected_cells'] == len(rows) == 35
        assert {(r['task'], r['cell']) for r in rows} == expected
        assert len({(r['task'], r['cell']) for r in rows}) == 35
        assert set(d['summary']['tasks']) == {t for t, _ in expected}
        assert sum(r['n'] for r in rows) == e['accounting']['episodes']
        for r in rows:
            confusion = r['confusion']
            assert sum(map(sum, confusion)) == r['n']
            assert r['n'] == (d['krauzlis_n'] if r['task'] == 'krauzlis_cued_motion' else d['n'])
            if r['cell'].startswith('N0_'):
                assert r['balanced_accuracy'] is None and r['auc'] is None
            else:
                recalls = [confusion[i][i] / sum(row) for i, row in enumerate(confusion)]
                assert r['class_recall'] == recalls
                assert abs(r['balanced_accuracy'] - sum(recalls) / len(recalls)) < 1e-12
                assert abs(r['accuracy'] - sum(confusion[i][i] for i in range(len(confusion))) / r['n']) < 1e-12
            if 'events' in r:
                assert sum(x['n'] for x in r['events'].values()) == r['n']
                assert all(sum(x['target_side_counts'].values()) == x['n'] for x in r['events'].values())
            for strata in r['strata'].values():
                assert sum(g['n'] for g in strata.values()) == r['n']
    a = s['architecture']
    assert sum(a['parameter_groups'].values()) == a['total_parameters']
    assert sum(p['numel'] for p in a['parameter_tensors']) == a['total_parameters']
    assert a['all_parameters_cpu'] and a['model_forward_calls'] == 0
    assert all(x['match'] for x in s['source_verification'])
    for row in s['training_exposure']['snapshots']:
        e = row['per_task_exposure']
        assert sum(t['updates'] for t in e.values()) == row['step']
        assert sum(t['episodes'] for t in e.values()) == row['cumulative_episodes']
        assert sum(t['frames'] for t in e.values()) == row['cumulative_frames']
        assert all(sum(t['cells'].values()) == t['episodes'] for t in e.values())
    return dict(evaluations=len(s['evaluations']), cells_per_evaluation=35,
                total_parameters=a['total_parameters'], all_checks_passed=True)

if '--verify' in sys.argv:
    s = json.loads((OUT / 'metrics_snapshot.json').read_text())
    print(json.dumps(verify(s), indent=2))
    raise SystemExit(0)
if (OUT / 'metrics_snapshot.json').exists():
    raise SystemExit('Refusing to overwrite immutable snapshot')

started = dt.datetime.now(dt.timezone.utc).isoformat()
cat = read(ROOT / 'SecondPass/TaskSuite/catalog.json')
config = read(V3 / 'config.json')
live = read(V3 / 'live_status.json')
metadata = {}
for version, directory in [('v1', V1), ('v2', V2), ('v3', V3)]:
    for name in ('config.json', 'budget.json', 'migration.json', 'resume_integrity.json', 'report.json', 'latest_checkpoint.json'):
        p = directory / name
        if p.exists(): metadata[f'{version}/{name}'] = read(p)
source_verification = []
for path, expected in {**config['data']['source_hashes'], **config['data']['continuation_source_hashes']}.items():
    actual = sha(Path(path).read_bytes())
    source_verification.append(dict(path=path, expected_sha256=expected, actual_sha256=actual, match=expected == actual))
assert all(r['match'] for r in source_verification)

# Freeze append-only logs by a single bytes read, discarding only an unfinished last line.
logs = {}; progress = {}
for version, directory in [('v2', V2), ('v3', V3)]:
    p = directory / 'progress.jsonl'; raw = p.read_bytes(); complete = raw[:raw.rfind(b'\n') + 1]
    rows = [json.loads(line) for line in complete.splitlines()]
    logs[version] = dict(path=str(p), captured_prefix_sha256=sha(complete), captured_bytes=len(complete),
                         rows=len(rows), first_step=rows[0]['step'], last_step=rows[-1]['step'])
    for r in rows: progress[r['step']] = r
# Status and log are not atomically synchronized. Select only the status step or earlier.
cutoff = max(step for step in progress if step <= live['data']['step'])

from SecondPass.JointTraining.launch import frames
from SecondPass.JointTraining.core import summarize
specs = {t['id']: t for t in cat['data']['tasks']}
conditions = [dict(task=t, cell=c['id'], kwargs=c['kwargs'], classes=spec['classes'],
                   selection_eligible=c['selection_eligible'], frames=frames(t, c),
                   validation_n=100 if t == 'krauzlis_cued_motion' else 64,
                   v2_final_test_n=200 if t == 'krauzlis_cued_motion' else 128)
              for t, spec in specs.items() for c in spec['conditions']]
evals = []; incomplete = []
candidates = [('v2_final_selected', 117, V2 / 'test_selected.json'),
              ('v2_final_terminal', 715, V2 / 'test_terminal.json'),
              ('v3_validation_baseline', 715, V2 / 'validation_000715.json')]
for p in sorted(V3.glob('validation_[0-9]*.json')):
    if not p.name.endswith('.partial.json'):
        candidates.append(('v3_validation', int(p.stem.split('_')[-1]), p))
for kind in ('terminal', 'selected'):
    p = V3 / f'test_{kind}.json'
    if p.exists():
        report = metadata.get('v3/report.json', {}).get('data', {})
        candidates.append((f'v3_final_{kind}', report.get(f'{kind}_step', live['data']['step'] if kind == 'terminal' else live['data']['best_step']), p))
for role, step, p in candidates:
    source = read(p); d = source['data'].get('results', source['data'])
    if not d.get('complete') or d.get('complete_cells') != 35:
        incomplete.append(dict(role=role, step=step, source=source)); continue
    rows = d['cells']; aggregated = summarize(rows)
    assert aggregated['tasks'] == d['summary']['tasks']
    assert aggregated['groups'] == d['summary']['groups']
    v3key = [sum(t['auc'] for t in aggregated['tasks'].values()) / 13,
             sum(t['chance_normalized_ba'] for t in aggregated['tasks'].values()) / 13]
    accounting = dict(tasks=len(aggregated['tasks']), cells=len(rows), eligible_cells=32,
                      episodes=sum(r['n'] for r in rows), episodes_by_task={t:sum(r['n'] for r in rows if r['task'] == t) for t in specs})
    evals.append(dict(role=role, step=step, source={k:v for k,v in source.items() if k != 'data'},
                      wrapper_metadata={k:v for k,v in source['data'].items() if k != 'results'} if 'results' in source['data'] else None,
                      result=d, accounting=accounting, v3_rule_key_recomputed=v3key))

# Construct architecture on CPU only: no .forward(), no model evaluation, no optimizer.
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline
model = AccumulatorBaseline({t:s['classes'] for t,s in specs.items()}, stack=3, center=True, accumulator='kda').cpu()
params = [dict(name=n, shape=list(p.shape), numel=p.numel(), requires_grad=p.requires_grad) for n,p in model.named_parameters()]
groups = {}
for p in params:
    name = p['name'].split('.')[0]
    groups[name] = groups.get(name, 0) + p['numel']
modules = {n:sum(p.numel() for p in m.parameters()) for n,m in model.named_modules()
           if n and (n.count('.') == 1 or n in ('feat', 'norm', 'gru', 'heads'))}
shapes = []
size = 100
for i,b in enumerate(model.blocks):
    conv = b[0]; size = (size + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
    shapes.append(dict(block=i, input_channels=conv.in_channels, output_channels=conv.out_channels,
                       kernel=conv.kernel_size[0], stride=conv.stride[0], padding=conv.padding[0],
                       output_shape=['B',conv.out_channels,size,size], groupnorm_groups=b[1].num_groups,
                       parameters=sum(p.numel() for p in b.parameters())))
states = [dict(map=[n,n],shape=['B',n,n,2,8,16],scalars_per_example=n*n*2*8*16,
               float32_bytes_per_example=n*n*2*8*16*4) for n in (25,13,7)]
architecture = dict(class_name='AccumulatorBaseline', stack=3, center=True, feature_norm='none', hidden=256,
    total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    parameter_groups=groups, module_parameters=modules, parameter_tensors=params, blocks=shapes,
    state_fields=states, kda_state_scalars_per_example=sum(s['scalars_per_example'] for s in states),
    kda_state_bytes_float32_per_example=sum(s['float32_bytes_per_example'] for s in states),
    gru_state_scalars_per_example=256, stack_history_scalars_per_example=2*3*100*100,
    all_parameters_cpu=all(p.device.type == 'cpu' for p in model.parameters()), model_forward_calls=0,
    torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(), torch_version=torch.__version__)

# Verify archived checkpoint tensor dimensions and digest, without running inference.
checkpoint_paths = [V1/'validation_checkpoint_000117.pt', V2/'terminal.pt']
checkpoint_paths += [V3/f"validation_checkpoint_{e['step']:06d}.pt" for e in evals if e['role'] == 'v3_validation']
checkpoints = []
for p in checkpoint_paths:
    raw = p.read_bytes(); digest = sha(raw)
    saved = torch.load(p, map_location='cpu')
    assert {k:list(v.shape) for k,v in saved['model'].items()} == {k:list(v.shape) for k,v in model.state_dict().items()}
    checks = dict(path=str(p), sha256=digest, bytes=len(raw), step=saved['state']['step'],
                  model_shapes_match=True, tensors_cpu=all(v.device.type == 'cpu' for v in saved['model'].values()),
                  optimizer_parameter_states=len(saved['optimizer']['state']))
    checkpoints.append(checks)
    del saved, raw
assert next(c for c in checkpoints if c['path'] == str(V2/'terminal.pt'))['sha256'] == config['data']['budget_source_sha256']
selected_steps = sorted({117,715,cutoff} | {e['step'] for e in evals if e['step'] in progress})
exposures = [{k:progress[step][k] for k in ('step','utc','cumulative_episodes','cumulative_frames','optimizer_seconds','per_task_exposure')}
             for step in selected_steps]
s = dict(schema='technical_report_saved_artifacts_v1', snapshot_started_utc=started,
         snapshot_completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         scope='Saved JSON statistics and CPU-only model construction/checkpoint-shape audit. Zero forward evaluations, training updates, or accelerator calls.',
         catalog=cat, conditions=conditions, architecture=architecture, source_verification=source_verification,
         live_status=live, metadata=metadata, checkpoints=checkpoints, evaluations=evals,
         excluded_incomplete_evaluations=incomplete,
         ignored_partial_files=[str(p) for p in V3.glob('*.partial.json')],
         training_exposure=dict(logs=logs, live_cutoff_step=cutoff, snapshots=exposures),
         caveats=['Validation fixed draws repeatedly reused for checkpoint selection; not final generalization.',
                  'v2 tests namespace 94292763; v3 final test namespace 94392763; validation namespace 732001. Never pair old test with new validation.',
                  'Delay cells are independent streams, not paired delay manipulations.',
                  'N0 cells are single-class specificity/FPR controls, excluded from BA/AUC.',
                  'Krauzlis target/foil/catch 57/29/14 per 100; side imbalance of one is retained at validation n=100.',
                  'Prior tests seen; same official BSDS500 source identities reused within final-test split; exploratory continuation, one seed.',
                  'No stored trial probabilities here: AUC can be copied and aggregates checked, but cannot be independently reconstructed from confusion.',
                  'No cueing, perturbation, calibration, reaction time or global model-contraction experiment was performed.'])
s['verification'] = verify(s)
(OUT/'metrics_snapshot.json').write_text(json.dumps(s, indent=2, allow_nan=False)+'\n')
(OUT/'metrics_snapshot.json').chmod(0o444)
print(json.dumps(dict(snapshot_utc=s['snapshot_completed_utc'], verification=s['verification'], live_step=cutoff,
                     architecture=architecture['parameter_groups'], total_parameters=architecture['total_parameters'],
                     modules=modules, kda_states=states,
                     evaluations=[dict(role=e['role'],step=e['step'],episodes=e['accounting']['episodes'],key=e['v3_rule_key_recomputed']) for e in evals]),indent=2))
