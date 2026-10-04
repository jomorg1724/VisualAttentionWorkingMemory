"""Focused CPU checks; no cloud calls and no production training."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch
from SecondPass.SpatialReadout.FreshRun import worker as fresh
from SecondPass.JointTraining.core import tree_equal, cpu_tree


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 2:
            torch.set_num_interop_threads(2)

    def test_exact_restore_and_next_native_draw(self):
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        with tempfile.TemporaryDirectory() as root:
            config = dict(device='cpu', effective_batch=32, microbatch=4,
                          cap_started=time.time(), deadline=time.time()+1000)
            source_session = fresh.Session(Path(root)/'source', config)
            task, cell = source_session.scheduler.next()
            source_session.stream.batch(1, task, cell)
            source_session.state['step'] = 1
            source_session.state['episodes'] = 32
            # Exercise actual Adam moments without an expensive full CPU training run.
            for p in source_session.model.parameters():
                p.grad = torch.full_like(p, .001)
            source_session.optimizer.step()
            source_session.state['best_key'] = [.6, .2]
            source_session.state['best_step'] = 1
            source_session.state['selection_history'] = [dict(step=1, key=[.6,.2], complete=True)]
            # Fresh's first-update verifier intentionally requires a fresh initial
            # receipt; suppress only that unrelated fixture hook.
            checkpoint = cont.cloud.clone(fresh.Session.checkpoint, verify_progress=lambda _: None)
            receipt = checkpoint(source_session, 'terminal.pt')
            payload = fresh.load_verified(receipt)
            resumed = cont.Session(Path(root)/'resumed', config, payload, receipt)
            self.assertTrue(tree_equal(cpu_tree(resumed.model.state_dict()), payload['model']))
            self.assertTrue(tree_equal(cpu_tree(resumed.optimizer.state_dict()), payload['optimizer']))
            self.assertTrue(tree_equal(resumed.state, payload['state']))
            evidence = resumed.verify_resume(payload, receipt)
            self.assertTrue(evidence['next_scheduler_equal'])
            self.assertTrue(evidence['next_native_draw_equal'])
            self.assertTrue(evidence['rng_equal'])
            self.assertTrue(tree_equal(resumed.scheduler.state_dict(), payload['scheduler']))
            self.assertTrue(tree_equal(resumed.stream.state_dict(), payload['stream']))

    def test_budget_plan_fixed_horizon_and_rejection(self):
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        scheduler = fresh.BalancedScheduler(1)
        for _ in range(7): scheduler.next()
        source = dict(state=dict(step=7), scheduler=scheduler.state_dict())
        rows = [dict(task=t, cell=c, seconds=.1, episodes=32, step=i+1)
                for i, (t,c) in enumerate(fresh.cloud.original.all_cells())]
        evaluation = dict(complete=True, cells=[dict(task=t, cell=c, seconds=.1, n=64)
                          for t,c in fresh.cloud.original.all_cells()])
        budget = dict(cap_started=100., hard_deadline=129700., deadline=129100.,
                      wall_cap_seconds=129600., retrieval_reserve_seconds=600.,
                      max_usd=20., additional_updates=104000,
                      origin='Explicit parent transition')
        cont.validate_budget(budget, budget, now=101.)
        plan = cont.measured_plan(source, rows, [evaluation], budget, now=101.)
        self.assertEqual(plan['max_steps'], 104007)
        self.assertEqual(plan['validation_steps'], [7+i for i in range(10400,104001,10400)])
        self.assertEqual(sum(v['updates'] for v in plan['planned_exposure'].values()), 104000)
        self.assertTrue(all(7999 <= v['updates'] <= 8001 for v in plan['planned_exposure'].values()))
        with self.assertRaises(ValueError):
            cont.validate_budget(budget, dict(budget, cap_started=101.), now=101.)
        with self.assertRaises(RuntimeError):
            cont.measured_plan(source, [dict(r, seconds=2.) for r in rows], [evaluation], budget, now=101.)

    def test_actual_evaluator_uses_new_stream(self):
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        from unittest.mock import patch
        class Sentinel(Exception): pass
        seen = []
        class Stream(cont.EvaluationStream):
            def __init__(self, split):
                seen.append((split, self.final_namespace))
                raise Sentinel()
        with patch.object(cont, 'EvaluationStream', Stream):
            with self.assertRaises(Sentinel):
                cont.evaluate(None, 'test', 128, 200, 4, 'cpu', Path('/unused'), float('inf'))
        self.assertEqual(seen, [('test', cont.FINAL_NAMESPACE)])
        stream = cont.EvaluationStream('val')
        old = fresh.FreshStream('val')
        for task, cell in fresh.cloud.original.all_cells():
            self.assertEqual(stream.stream_seed(task,cell), old.stream_seed(task,cell))
        self.assertNotEqual(cont.EvaluationStream('test').stream_seed(task,cell),
                            fresh.FreshStream('test').stream_seed(task,cell))

    def test_real_terminal_cpu_restore_without_forward(self):
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        path = Path('/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_fresh_01/artifacts/terminal.pt')
        if not path.exists(): self.skipTest('Real terminal not present on this host')
        receipt = dict(path=str(path), bytes=16742235,
                       sha256='7b87e8f21e4a972cbb78f8d657786e38c2aab29bf313329167575f9f564d5ff1', step=7739)
        source = cont.cloud.load_verified(receipt)
        with tempfile.TemporaryDirectory() as root:
            config = dict(source['state']['config'], device='cpu')
            resumed = cont.Session(root, config, source, receipt)
            evidence = resumed.verify_resume(source, receipt)
            self.assertTrue(evidence['optimizer_named_state_equal'])
            self.assertEqual(resumed.state['best_step'], 5200)
            self.assertEqual(resumed.state['episodes'], 247648)
            self.assertEqual(resumed.additional_step, 0)

    def test_validation_all_ten_looks_with_parent_history(self):
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        config = dict(validation_steps=[7739+i for i in range(10400,104001,10400)])
        history = [dict(step=5200), dict(step=1)]
        looks = []
        for step in range(7740,111740):
            if cont.validation_due(step, config):
                looks.append(step)
                history.append(dict(step=step))
        self.assertEqual(looks, config['validation_steps'])
        self.assertFalse(cont.validation_due(111738, config))

    def test_cpu_persisted_updates_and_finalization_adapter(self):
        """Real Adam/checkpoints; cheap gradient fixture replaces GPU forward only."""
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        from unittest.mock import patch
        import contextlib
        import io
        path = Path('/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_fresh_01/artifacts/terminal.pt')
        if not path.exists(): self.skipTest('Real terminal not present on this host')
        receipt = dict(path=str(path), bytes=16742235,
                       sha256='7b87e8f21e4a972cbb78f8d657786e38c2aab29bf313329167575f9f564d5ff1', step=7739)
        source = cont.cloud.load_verified(receipt)
        now = time.time()
        budget = dict(cap_started=now, deadline=now+129000, hard_deadline=now+129600,
                      wall_cap_seconds=129600, retrieval_reserve_seconds=600, max_usd=20,
                      additional_updates=13, origin='CPU TEST fixture; never production')
        config = dict(source['state']['config'], **{k:v for k,v in budget.items()})
        config.update(device='cpu', source_checkpoint=receipt, additional_target=13, max_steps=7752,
                      validation_steps=[7740,7752], estimated_one_test_seconds=1,
                      final_reserve_seconds=10, update_estimate_seconds=1,
                      production_cell_seconds=[dict(task=t,cell=c,seconds=.5) for t,c in cont.cloud.original.all_cells()])
        calls = []
        def gradient_update(model, optimizer, stream, task, cell, effective, microbatch, device):
            images, _, _ = stream.batch(1, task, cell)
            optimizer.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if not name.startswith('heads.') or name.startswith('heads.'+task+'.'):
                    p.grad = torch.full_like(p, .001)
            optimizer.step()
            return dict(task=task,cell=cell,loss=1.,seconds=.5,episodes=effective,frames=effective*images.shape[1])
        def fixture_evaluate(model, split, n, kn, micro, device, output, deadline, cells=None):
            calls.append((split,Path(output).name))
            result = dict(complete=True, cells=[dict(task=t,cell=c,n=kn if t=='krauzlis_cued_motion' else n)
                                               for t,c in cont.cloud.original.all_cells()],
                          summary=dict(equal_task_mean_auc=0., tasks={t:dict(chance_normalized_ba=0.) for t in TASKS}))
            cont.atomic_json(output,result)
            return result
        from SecondPass.TaskSuite.suite import TASKS
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for name,value in [('budget',budget),('config',config),('source_checkpoint',receipt),('parent_best_checkpoint',receipt)]:
                cont.atomic_json(directory/(name+'.json'),value)
            with patch.object(cont, 'ADDITIONAL', 13), patch.object(cont.cloud.original, 'verify_sources', lambda _:None), \
                 patch.object(cont.cloud, 'update', gradient_update), patch.object(cont, 'evaluate', fixture_evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cont.run(directory),0)
            self.assertEqual([s for s,_ in calls], ['val','val','test','test'])
            for added in (1,13):
                check = json.loads((directory/f'persisted_progress_{added:06d}.json').read_text())
                self.assertTrue(check['verified'])
                self.assertEqual(check['additional_step'],added)
            report = json.loads((directory/'report.json').read_text())
            self.assertEqual(report['terminal_step'],7752)
            self.assertEqual(report['selected_step'],5200)
            self.assertEqual(len(report['selection_history']),3)
            self.assertTrue(report['final_coverage_complete'])
            self.assertEqual(cont.cloud.digest(path),receipt['sha256'])
            # Completion adapter must publish the verified carried best, even
            # when the old absolute path is unavailable on this CPU host.
            copied = cont.copy_verified(receipt,directory/'parent_best.pt')
            cont.atomic_json(directory/'parent_best_checkpoint.json',copied)
            config['parent_best_receipt'] = dict(receipt,path=source['state']['best_checkpoint'])
            cont.atomic_json(directory/'config.json',config)
            cont.atomic_json(directory/'allocation.json',dict(max_steps=7752))
            cont.completion(directory,'complete')
            manifest = json.loads((directory/'cloud_completion.json').read_text())
            self.assertEqual(manifest['actual_additional_updates'],13)
            self.assertEqual(manifest['actual_updates'],13)
            self.assertEqual(manifest['actual_cumulative_updates'],7752)
            for item in manifest['artifact_manifest']:
                self.assertEqual(cont.cloud.digest(item['path']),item['sha256'])
            # Corruption must fail closed rather than publishing persisted progress.
            wrong = dict(receipt, sha256='0'*64)
            with self.assertRaises(ValueError): cont.cloud.load_verified(wrong)

    def test_config_drops_stale_fresh_allocation(self):
        from SecondPass.SpatialReadout.ContinuationRun import worker as cont
        from unittest.mock import patch
        config = dict(effective_batch=32,microbatch=4,eval_microbatch=4,all_trainable=True,
                      lr=1e-4,clipping=None,bptt='full',precision='fp32',tf32=False,
                      source_hashes={},target_requested=10400,total_episodes=332800,updates_per_task=800)
        with patch.object(cont.cloud.original,'verify_sources',lambda _: None):
            current = cont.config_for(dict(state=dict(config=config)), {}, None, dict(cap_started=1,origin='test'))
        for key in ('target_requested','total_episodes','updates_per_task'):
            self.assertNotIn(key,current)
        with self.assertRaises(ValueError):
            cont.read_source(dict(sha256='0'*64,step=7739,bytes=16742235))


if __name__ == '__main__':
    unittest.main()
