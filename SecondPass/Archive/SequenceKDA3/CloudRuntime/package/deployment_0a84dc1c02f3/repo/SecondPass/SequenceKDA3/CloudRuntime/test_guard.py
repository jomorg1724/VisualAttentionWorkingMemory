import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).parent

def load():
    assert (ROOT/'pod_guard.py').exists(), 'completion-stop guard missing'
    spec=importlib.util.spec_from_file_location('guard90',ROOT/'pod_guard.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class GuardTests(unittest.TestCase):
    def test_finalized_stops_without_retrieval(self):
        g=load()
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)/'run';run.mkdir()
            def put(name,data): (run/name).write_text(json.dumps(data))
            cells=[dict(task='krauzlis_cued_motion',cell=c,n=200,events={e:dict(n=n) for e,n in [('target',114),('foil',58),('catch',28)]}) for c in ('B12','B20','B28')]
            report=dict(failure=None,final_coverage_complete=True,terminal_step=1000,selected_step=1000,config=dict(max_steps=1000))
            put('report.json',report)
            for role in ('terminal','selected'): put('test_'+role+'.json',dict(complete=True,cells=cells))
            put('supervisor_result.json',dict(status='complete',returncode=0,supervisor_pid=880001,worker_pid=880002))
            put('owner_run.json',dict(status='complete',pid=880003))
            for f in ('selected.pt','terminal.pt'): (run/f).write_bytes(b'cpu-test-fixture-not-weights')
            rows=[dict(relative_path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in run.iterdir()]
            put('cloud_completion.json',dict(status='complete',artifact_manifest=rows,actual_updates=1000,pinned_updates=1000))
            with patch.object(g,'process_absent',return_value=True):
                self.assertIsNone(g.stop_reason(100,400,0,run))
                self.assertEqual(g.stop_reason(220,400,0,run),'verified_finalization_disk_retained')
                (run/'selected.pt').write_bytes(b'corrupt')
                self.assertIsNone(g.stop_reason(100,200,0,run))
            self.assertEqual(g.stop_reason(200,200,0,run),'wall_deadline')
            put('DEPLOYMENT_FAILED',dict(stop=True,reason='setup_or_worker_failure'))
            self.assertEqual(g.stop_reason(100,200,0,run),'setup_or_worker_failure')
    def test_failed_publication_never_keeps_completed_gpu(self):
        g=load()
        class Provider:
            def __init__(self): self.stops=0
            def get(self): return dict(id='newpod')
            def patch_status(self,body): raise OSError('offline')
            def stop(self): self.stops+=1; return 204
        with tempfile.TemporaryDirectory() as t:
            p=Provider(); guard=g.Guard(t,'newpod',200,p)
            with patch.object(g,'stop_reason',return_value='verified_finalization_disk_retained'):
                guard.tick(100)
            self.assertEqual(p.stops,1)
    def test_real_finalizer_to_guard_contract(self):
        g=load()
        import subprocess,sys
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); run=root/'run'; run.mkdir()
            def put(name,obj): (run/name).write_text(json.dumps(obj))
            pids=[]
            for _ in range(2):
                child=subprocess.Popen([sys.executable,'-c','pass']); pids.append(child.pid); child.wait()
            cells=[dict(task='krauzlis_cued_motion',cell=c,n=200,events={e:dict(n=n) for e,n in [('target',114),('foil',58),('catch',28)]}) for c in ('B12','B20','B28')]
            put('report.json',dict(failure=None,final_coverage_complete=True,terminal_step=1000,selected_step=500,config=dict(max_steps=1000)))
            put('supervisor_result.json',dict(status='complete',returncode=0,worker_pid=pids[0],supervisor_pid=pids[1]))
            for role in ('selected','terminal'):
                put('test_'+role+'.json',dict(complete=True,cells=cells)); (run/(role+'.pt')).write_bytes(b'fixture')
            rows=[dict(relative_path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in run.iterdir()]
            put('cloud_completion.json',dict(status='complete',artifact_manifest=rows,pinned_updates=1000,actual_updates=1000))
            subprocess.run([sys.executable,'-c','from pathlib import Path; import remote_owner; remote_owner.finalize(Path('+repr(t)+'))'],cwd=ROOT,check=True)
            self.assertTrue(g.verified_finalization(run))
            self.assertIsNone(g.stop_reason(100,400,0,run))
            self.assertEqual(g.stop_reason(220,400,0,run),'verified_finalization_disk_retained')
            (run/'test_terminal.json').write_text('{}')
            self.assertFalse(g.verified_finalization(run))
    def test_retrieval_marker_stops_and_deadline_overrides_grace(self):
        g=load()
        with tempfile.TemporaryDirectory() as t:
            run=Path(t); completion=b'{"status":"complete"}'
            (run/'cloud_completion.json').write_bytes(completion)
            with patch.object(g,'verified_finalization',return_value=True):
                self.assertIsNone(g.stop_reason(100,200,0,run))
                self.assertEqual(g.stop_reason(200,200,0,run),'wall_deadline')
                marker=dict(complete=True,completion_sha256=hashlib.sha256(completion).hexdigest())
                (run/'retrieval_verified.json').write_text(json.dumps(marker))
                self.assertEqual(g.stop_reason(110,200,0,run),'verified_finalization_retrieved')
    def test_deadline_stop_precedes_publication(self):
        g=load();calls=[]
        class Provider:
            def get(self): return dict(id='newpod')
            def stop(self): calls.append('stop');return 204
            def patch_status(self,body): calls.append('publish');raise OSError('offline')
        with tempfile.TemporaryDirectory() as t:
            guard=g.Guard(t,'newpod',100,Provider(),boot=0)
            guard.tick(100)
            self.assertEqual(calls,['stop','publish'])
    def test_healthy_early_cap_requires_unchanged_pinned_and_explicit_reason(self):
        g=load()
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)
            def put(name,data): (run/name).write_text(json.dumps(data))
            cells=[dict(task='krauzlis_cued_motion',cell=c,n=200,events={e:dict(n=n) for e,n in [('target',114),('foil',58),('catch',28)]}) for c in ('B12','B20','B28')]
            report=dict(failure=None,final_coverage_complete=True,terminal_step=900,selected_step=450,config=dict(max_steps=1000),
                stop_reason='wall_budget_reserve',cap_limited=True,exposure_completed=False)
            for role in ('terminal','selected'):
                put('test_'+role+'.json',dict(complete=True,cells=cells));(run/(role+'.pt')).write_bytes(b'fixture')
            put('supervisor_result.json',dict(status='complete',returncode=0,supervisor_pid=880001,worker_pid=880002))
            put('owner_run.json',dict(status='complete',pid=880003))
            def manifest(actual=900,pinned=1000):
                rows=[dict(relative_path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in run.iterdir() if p.name!='cloud_completion.json']
                put('cloud_completion.json',dict(status='complete',artifact_manifest=rows,actual_updates=actual,pinned_updates=pinned))
            with patch.object(g,'process_absent',return_value=True):
                put('report.json',report);manifest();self.assertTrue(g.verified_finalization(run))
                for change in (dict(stop_reason='signal'),dict(cap_limited=False),dict(exposure_completed=True)):
                    put('report.json',dict(report,**change));manifest();self.assertFalse(g.verified_finalization(run))
                missing=dict(report);missing.pop('cap_limited');put('report.json',missing);manifest();self.assertFalse(g.verified_finalization(run))
                put('report.json',report);manifest(actual=901);self.assertFalse(g.verified_finalization(run))
                manifest(pinned=900);self.assertFalse(g.verified_finalization(run))
    def test_live_process_blocks_completion(self):
        g=load()
        import os
        self.assertFalse(g.process_absent(os.getpid()))

if __name__=='__main__': unittest.main()
