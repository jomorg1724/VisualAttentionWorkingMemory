"""Focused offline contracts; no provider, accelerator or training calls."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import deploy,pod_guard,mirror_watch

class RuntimeTests(unittest.TestCase):
    def test_price_and_no_parallel(self):
        quote=deploy.price_contract(dict(id='NVIDIA A40',availability='HIGH',price=dict(secure=.49)),1,24,100)
        self.assertAlmostEqual(8*quote['compute_hourly_usd']+quote['storage_reserve_usd'],4.92)
        deploy.require_no_parallel([])
        with self.assertRaises(ValueError):deploy.require_no_parallel([dict(status='RUNNING',id='old')])
        with self.assertRaises(ValueError):deploy.require_no_parallel([dict(id='unknown')])
    def test_startup_requires_92_named_adam_updates(self):
        cp=dict(step=3,sha256='a'*64)
        data={'latest_checkpoint.json':cp,'progress':dict(step=3),'persisted_progress_verification.json':dict(verified=True,step=3,checkpoint=cp,changed_parameter_tensors=92,all_trainable_parameters_changed=True,all_named_adam_steps_verified=True)}
        self.assertTrue(deploy.production_ready(data))
        data['persisted_progress_verification.json']['changed_parameter_tensors']=91
        self.assertFalse(deploy.production_ready(data))
    def test_deadline_stop_first_and_cleanup_wrongpod_rejected(self):
        calls=[]
        class Provider:
            def stop(self): calls.append('stop');return 204
            def get(self):return dict(id='new')
            def patch_status(self,body):calls.append('status');raise OSError('offline')
        with tempfile.TemporaryDirectory() as tmp:
            guard=pod_guard.Guard(tmp,'new',100,Provider(),boot=0)
            guard.tick(100);self.assertEqual(calls,['stop','status'])
            root=Path(tmp);(root/'pod.json').write_text(json.dumps(dict(id='new',remote_root=deploy.REMOTE)))
            with patch.object(mirror_watch,'ROOT',root),patch.object(mirror_watch,'api') as api:
                with self.assertRaises(ValueError):mirror_watch.cleanup_verified_ephemeral(dict(complete=True,pod='old'))
                api.assert_not_called()
    def test_two_checkpoint_finalization_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            def put(name,obj): (run/name).write_text(json.dumps(obj))
            cells=[dict(task='krauzlis_cued_motion',cell=c,n=200,events={e:dict(n=n) for e,n in [('target',114),('foil',0),('catch',86)]}) for c in ('B12','B20','B28')]
            put('report.json',dict(failure=None,final_coverage_complete=True,terminal_step=330,selected_step=330,config=dict(max_steps=330)))
            for role in ('selected','terminal'):put('test_'+role+'.json',dict(complete=True,cells=cells))
            for name in ('best.pt','latest.pt'):(run/name).write_bytes(b'fixture')
            put('supervisor_result.json',dict(returncode=0,status='complete',worker_pid=900001,supervisor_pid=900002))
            put('owner_run.json',dict(status='complete',pid=900003))
            rows=[dict(relative_path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in run.iterdir()]
            put('cloud_completion.json',dict(status='complete',actual_updates=330,pinned_updates=330,artifact_manifest=rows))
            with patch.object(pod_guard,'process_absent',return_value=True):
                self.assertTrue(pod_guard.verified_finalization(run))
                (run/'latest.pt').write_bytes(b'corrupt');self.assertFalse(pod_guard.verified_finalization(run))

if __name__=='__main__':unittest.main()
