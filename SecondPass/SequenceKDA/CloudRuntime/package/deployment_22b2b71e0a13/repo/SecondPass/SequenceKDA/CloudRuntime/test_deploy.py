"""CPU-only deployment delta tests. No live API transport."""
import importlib.util
from pathlib import Path
import unittest

class DeployTests(unittest.TestCase):
    def test_live_quote_includes_storage_and_no_parallel_pods(self):
        path=Path(__file__).with_name('deploy.py')
        self.assertTrue(path.exists(), 'Missing executable parent deployment')
        spec=importlib.util.spec_from_file_location('deploy',path)
        d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
        q=d.price_contract({'id':'NVIDIA A40','price':{'secure':0.44},'availability':'HIGH'},1.0,24,100)
        self.assertLessEqual(8*q['compute_hourly_usd']+q['storage_reserve_usd'],5)
        with self.assertRaises(ValueError): d.price_contract({'id':'NVIDIA A40','price':{'secure':0.60},'availability':'HIGH'},1.0,24,100)
        with self.assertRaises(ValueError): d.price_contract({'id':'NVIDIA A40','price':{'secure':0.44},'availability':'NONE'},1.0,24,100)
        with self.assertRaises(ValueError): d.require_no_parallel([{'id':'other','status':'RUNNING'}])
        d.require_no_parallel([{'id':'other','status':'EXITED'}])

    def test_production_receipt_must_be_verified(self):
        import deploy as d
        self.assertTrue(hasattr(d,'production_ready'), 'Missing semantic progress gate')
        data={'latest_checkpoint.json':{'step':3,'sha256':'test-digest'},'progress':{'step':3},'persisted_progress_verification.json':{'verified':False,'step':3},'first_cycle_timing.json':{'matched_profile_ratio':1.0,'material_slowdown':False}}
        self.assertFalse(d.production_ready(data))
        data['persisted_progress_verification.json']['verified']=True
        data['persisted_progress_verification.json']['checkpoint']=data['latest_checkpoint.json']
        self.assertTrue(d.production_ready(data))
        data['first_cycle_timing.json']['material_slowdown']=True
        data['first_cycle_timing.json']['matched_profile_ratio']=1.73
        self.assertTrue(d.production_ready(data))
        for ratio in (float('nan'),float('inf'),0,-1,None):
            data['first_cycle_timing.json']['matched_profile_ratio']=ratio
            self.assertFalse(d.production_ready(data))
        data['first_cycle_timing.json']['matched_profile_ratio']=1.0
        data['first_cycle_timing.json'].pop('material_slowdown')
        self.assertFalse(d.production_ready(data))

    def test_unknown_pod_blocks_and_exact_price_cap(self):
        import deploy as d
        with self.assertRaises(ValueError): d.require_no_parallel([{'id':'other'}])
        q=d.price_contract({'id':'NVIDIA A40','price':{'secure':.49},'availability':'HIGH'},1,24,100)
        self.assertAlmostEqual(8*q['compute_hourly_usd']+q['storage_reserve_usd'],4.92)
        with self.assertRaises(ValueError): d.price_contract({'id':'NVIDIA A40','price':{'secure':.49},'availability':'HIGH'},.001,24,100)
    def test_source_archive_freeze_and_tamper(self):
        import deploy as d
        import hashlib,io,json,tarfile,tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'package').mkdir()
            archive=root/'bundle.tar.gz';payload=b'fresh-source'
            manifest={'checkpoints_included':False,'files':[{'path':'repo/model.py','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
            with tarfile.open(archive,'w:gz') as tf:
                for name,data in [('repo/model.py',payload),('deployment_manifest.json',json.dumps(manifest).encode())]:
                    row=tarfile.TarInfo(name);row.size=len(data);tf.addfile(row,io.BytesIO(data))
            digest=hashlib.sha256(archive.read_bytes()).hexdigest()
            (root/'package/bundle_receipt.json').write_text(json.dumps(dict(archive=str(archive),sha256=digest)))
            source=root/'source.py';source.write_text('fresh')
            frozen=dict(approved_archive_sha256=digest,source_hashes={'source.py':hashlib.sha256(source.read_bytes()).hexdigest()},inherited_source_hashes={})
            (root/'runtime_manifest.json').write_text(json.dumps(frozen))
            runtime=root/'runtime.tar.gz';runtime.write_bytes(b'fixture')
            (root/'runtime_bundle_receipt.json').write_text(json.dumps(dict(archive=str(runtime),sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())))
            key=root/'ssh_key';key.write_text('fake');key.with_suffix('.pub').write_text('fake')
            with patch.object(d,'ROOT',root),patch.object(d,'KEY',key):
                self.assertEqual(d.check()[0],archive)
                source.write_text('tampered')
                with self.assertRaises(ValueError):d.check()
                source.write_text('fresh');archive.write_bytes(b'tampered')
                with self.assertRaises(ValueError):d.check()
    def test_status_and_retrieve_commands_exist(self):
        import deploy
        import monitor
        import retrieve_once
        self.assertTrue(callable(monitor.status))
        self.assertTrue(callable(retrieve_once.main))

    def test_failed_setup_stops_and_retains_disk(self):
        import deploy as d
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'.runpod').mkdir()
            (root/'.runpod/config.toml').write_text('[default]\napi_key="fake-test-only"\n')
            (root/'pod_guard.py').write_text('pass')
            (root/'ssh_key.pub').write_text('ssh-ed25519 fake-test')
            calls=[];created={}
            def fake_api(path,method='GET',body=None):
                calls.append((path,method))
                if method=='POST' and path=='/v2/pods':
                    created.update(body);return {'id':'newpod'}
                if path=='/v2/pods/newpod':
                    return dict(id='newpod',cost=.49,status='RUNNING',gpu={'id':'NVIDIA A40','count':1},entrypoint=created['entrypoint'],env=created['env'])
                raise AssertionError(path)
            with patch.object(d,'ROOT',root),patch.object(d,'KEY',root/'ssh_key'),patch.object(d.Path,'home',return_value=root),patch.object(d,'frozen_archive_sha',return_value='f'*64),patch.object(d,'check',return_value=(root/'archive',root/'runtime','sha')),patch.object(d,'list_pods',return_value=[]),patch.object(d,'live_quote',return_value={'observed':__import__('time').time(),'compute_hourly_usd':.49}),patch.object(d,'create_status',return_value={'id':'record'}),patch.object(d,'api',side_effect=fake_api),patch.object(d,'install_and_launch',side_effect=RuntimeError('setup failed')),patch.object(d,'capture_diagnostics'),patch.object(d,'stop_and_verify') as stop:
                with self.assertRaises(RuntimeError): d.deploy(1,24,True)
                stop.assert_called_once_with('newpod')
                self.assertNotIn('fake-test-only',(root/'pod.json').read_text())
                self.assertFalse(any('/delete' in path or method=='DELETE' for path,method in calls))

if __name__=='__main__': unittest.main()
