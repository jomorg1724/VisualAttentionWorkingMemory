"""Offline retrieval integrity and no-restart checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import retrieve_once
import mirror_watch

class MirrorTests(unittest.TestCase):
    def test_manifest_integrity_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'checkpoint.pt';p.write_bytes(b'fixture')
            rows=[dict(relative_path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())]
            retrieve_once.verify_local(root,rows)
            p.write_bytes(b'damaged')
            with self.assertRaises(ValueError): retrieve_once.verify_local(root,rows)
            for name in ('../escape','/absolute','bad\nname'):
                with self.assertRaises(ValueError): retrieve_once.validate_items([dict(rows[0],relative_path=name)])
    def test_stopped_pod_never_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pod.json').write_text(json.dumps(dict(id='stopped')))
            with patch.object(retrieve_once,'ROOT',root),patch.object(retrieve_once,'api',return_value=dict(status='EXITED')) as api:
                with self.assertRaises(RuntimeError):retrieve_once.retrieve()
                api.assert_called_once_with('/v2/pods/stopped')
    def test_cleanup_rejects_wrong_pod_or_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pod.json').write_text(json.dumps(dict(id='newpod',remote_root='/workspace/vawm_two_frame_rvit_01')))
            with patch.object(mirror_watch,'ROOT',root),patch.object(mirror_watch,'api') as api:
                for receipt in (dict(complete=False,pod='newpod'),dict(complete=True,pod='oldpod')):
                    with self.assertRaises(ValueError):mirror_watch.cleanup_verified_ephemeral(receipt)
                api.assert_not_called()
    def test_cleanup_complete_stopped_only_exact_pod(self):
        import deploy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dest=root/'artifacts';dest.mkdir()
            (root/'pod.json').write_text(json.dumps(dict(id='newpod',remote_root=deploy.REMOTE)))
            rows=[];checks={}
            for role in ('selected','terminal'):
                p=dest/(role+'.pt');p.write_bytes(b'fixture');h=hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append(dict(relative_path=p.name,sha256=h,bytes=p.stat().st_size))
                checks[role]=dict(verified=True,path=str(p),sha256=h)
            complete=dest/'cloud_completion.json';complete.write_text(json.dumps(dict(artifact_manifest=rows)))
            receipt=dict(complete=True,pod='newpod',completion_sha256=hashlib.sha256(complete.read_bytes()).hexdigest(),cpu_checkpoint_reload=checks)
            with patch.object(mirror_watch,'ROOT',root),patch.object(mirror_watch,'api',return_value=dict(id='newpod',status='STOPPED')) as api,patch.object(deploy,'list_pods',return_value=[]),patch.object(mirror_watch,'save'):
                result=mirror_watch.cleanup_verified_ephemeral(receipt)
                self.assertTrue(result['deleted'])
                self.assertEqual(api.call_args_list[-1].args,('/v2/pods/newpod','DELETE'))
    def test_mirror_exits_when_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pod.json').write_text(json.dumps(dict(id='stopped',hard_deadline=1000)))
            with patch.object(mirror_watch,'ROOT',root),patch.object(mirror_watch.time,'time',return_value=100),patch.object(mirror_watch,'api',return_value=dict(status='STOPPED')) as api,patch.object(mirror_watch.monitor,'status'),patch.object(mirror_watch,'save') as save:
                mirror_watch.watch()
                api.assert_called_once_with('/v2/pods/stopped')
                self.assertEqual(save.call_args[0][1]['status'],'stopped')

if __name__=='__main__':unittest.main()
