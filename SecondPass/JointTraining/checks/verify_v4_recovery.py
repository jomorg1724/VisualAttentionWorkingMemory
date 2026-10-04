"""CPU-only independent persisted-state audit; does not initialize MPS."""
import datetime
import json
from pathlib import Path
import subprocess
import torch
from SecondPass.JointTraining import continuation_v4_queue as q
from SecondPass.JointTraining.continuation_v4 import is_joint_worker
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, tree_equal

torch.set_num_threads(2)
directory = Path('SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h').resolve()
manifest = q.load(directory/'queue_manifest.json')
config = q.load(directory/'config.json'); budget = q.load(directory/'budget.json')
q.verify_hashes(manifest['source_hashes']); q.verify_hashes(config['continuation_source_hashes'])
completion = q.verify_predecessor(manifest['source'])
archive = directory/'blocked_attempt_01'
archived = q.load(archive/'archive_sha256.json')
for relative, expected in archived.items():
    assert q.digest(archive/relative) == expected
    assert q.digest(directory/relative) == expected
    assert (archive/relative).stat().st_mode & 0o222 == 0
source = torch.load(completion['source_checkpoint']['path'], map_location='cpu')
receipts = [json.loads(line) for line in (directory/'checkpoints.jsonl').read_text().splitlines()]
first_receipt = next(r for r in receipts if r['step'] == 2861)
migration_receipt = next(r for r in receipts if Path(r['path']).name == 'migration.pt')
latest_receipt = q.load(directory/'latest_checkpoint.json')
loaded = {}
for name, receipt in [('migration', migration_receipt), ('first', first_receipt), ('latest', latest_receipt)]:
    path = Path(receipt['path'])
    assert q.digest(path) == receipt['sha256']
    assert path.stat().st_size == receipt['bytes']
    loaded[name] = torch.load(path, map_location='cpu')
    assert loaded[name]['state']['step'] == receipt['step']
    assert q.digest(path) == receipt['sha256']
migration = loaded['migration']; first = loaded['first']; latest = loaded['latest']
checks = {key:tree_equal(source[key], migration[key]) for key in ['model','optimizer','scheduler','stream','rng']}
assert all(checks.values())
assert migration['state']['step'] == 2860 and first['state']['step'] == 2861
assert first['state']['episodes'] == 91552
changed_model = [k for k in first['model'] if not tree_equal(migration['model'][k], first['model'][k])]
changed_adam = [k for k in first['optimizer']['state'] if not tree_equal(migration['optimizer']['state'][k], first['optimizer']['state'][k])]
advanced_adam = [k for k in first['optimizer']['state'] if first['optimizer']['state'][k]['step'].item() == migration['optimizer']['state'][k]['step'].item()+1]
assert changed_model and changed_adam and advanced_adam
assert set(advanced_adam) == set(changed_adam)
assert tree_equal(migration['optimizer']['param_groups'], first['optimizer']['param_groups'])
assert all(t.dtype == torch.float32 for t in first['model'].values())
scheduler = BalancedScheduler(0); scheduler.load_state_dict(source['scheduler']); expected_task, expected_cell = scheduler.next()
assert tree_equal(scheduler.state_dict(), first['scheduler'])
assert not tree_equal(migration['stream'], first['stream'])
progress = [json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]
assert [r['step'] for r in progress] == list(range(2861, progress[-1]['step']+1))
assert all(r['cumulative_episodes'] == r['step']*32 for r in progress)
assert progress[0]['task'] == expected_task and progress[0]['cell'] == expected_cell
assert first['state']['optimizer_seconds'] > migration['state']['optimizer_seconds']
workers = []
for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
    fields = line.strip().split(None,1)
    if len(fields) == 2 and is_joint_worker(fields[1]): workers.append(dict(pid=int(fields[0]),command=fields[1]))
supervisor = q.load(directory/'supervisor.json')
assert len(workers) == 1 and workers[0]['pid'] == supervisor['worker_pid']
assert q.identity(supervisor['pid']) is not None
assert all(q.identity(pid) is None for pid in [51429,51433,11016])
assert supervisor['deadline'] == budget['deadline'] == config['deadline']
assert budget['deadline'] == budget['cap_started']+28800
assert config['cpu_threads'] == 2 and config['max_steps'] == 5005
assert config['allocation']['additional_updates'] == 2145 and config['allocation']['additional_episodes'] == 68640
assert config['validation_steps'] == [3393,3926,4459,5005]
assert config['final_test_seed_namespace'] == 94492763
result = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cpu_only_independent_readback=True,
    process_handle='proc_b345f6fe2730', supervisor=supervisor, sole_worker=workers[0],
    supervisor_identity=q.identity(supervisor['pid']), worker_identity=q.identity(supervisor['worker_pid']),
    source_checkpoint=completion['source_checkpoint'], migration=migration_receipt, migration_exact_checks=checks,
    first_update=first_receipt, first_model_tensors_changed=len(changed_model), first_adam_states_changed=len(changed_adam),
    first_adam_steps_advanced=len(advanced_adam), exact_next_scheduler=True, native_stream_advanced=True,
    first_optimizer_seconds=first['state']['optimizer_seconds'], carried_optimizer_seconds=migration['state']['optimizer_seconds'],
    progress_rows=len(progress), latest_persisted_progress={k:progress[-1][k] for k in ['utc','step','cumulative_episodes','loss','optimizer_seconds']},
    latest_checkpoint=latest_receipt, archive_and_originals_verified=True, frozen_sources_verified=True,
    budget=budget, deadline_utc=datetime.datetime.fromtimestamp(budget['deadline'],datetime.timezone.utc).isoformat(),
    cap_started_utc=datetime.datetime.fromtimestamp(budget['cap_started'],datetime.timezone.utc).isoformat(),
    plan=dict(target=5005,additional_updates=2145,additional_episodes=68640,validation_steps=config['validation_steps']),
    final_report=str(directory/'report.json'))
atomic_json(directory/'recovery_independent_verification.json', result)
print(json.dumps(result, indent=2))
