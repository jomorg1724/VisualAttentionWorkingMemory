"""Focused queue gates; no cloud calls or training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import queue_after as q


class QueueTests(unittest.TestCase):
    def test_waits_then_requires_exact_verified_final_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def save(name, value):
                (root/name).write_text(json.dumps(value))
            save('pod.json',dict(id=q.PREDECESSOR_POD))
            self.assertFalse(q.predecessor_ready(root))
            save('mirror_result.json',dict(status='complete_verified_deleted'))
            save('cleanup_verified.json',dict(pod=q.PREDECESSOR_POD,deleted=True,artifacts_verified=True))
            (root/'artifacts').mkdir()
            save('artifacts/cloud_completion.json',dict(status='complete'))
            receipt=dict(pod=q.PREDECESSOR_POD,complete=True,
                completion_sha256=hashlib.sha256((root/'artifacts/cloud_completion.json').read_bytes()).hexdigest(),
                cpu_checkpoint_reload={role:dict(verified=True,step=4216) for role in ('selected','terminal')})
            save('retrieval_verified.json',receipt)
            self.assertTrue(q.predecessor_ready(root))
            receipt['cpu_checkpoint_reload']['terminal']['verified']=False
            save('retrieval_verified.json',receipt)
            with self.assertRaises(ValueError):q.predecessor_ready(root)
            save('pod.json',dict(id='different'))
            with self.assertRaises(ValueError):q.predecessor_ready(root)

    def test_failed_predecessor_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'pod.json').write_text(json.dumps(dict(id=q.PREDECESSOR_POD)))
            (root/'mirror_result.json').write_text(json.dumps(dict(status='deadline_elapsed')))
            with self.assertRaises(ValueError):q.predecessor_ready(root)

    def test_slow_first_ssh_probe_is_retried(self):
        import deploy
        import subprocess
        success=subprocess.CompletedProcess([],0,stdout='',stderr='')
        with patch.object(deploy,'api',return_value=dict(publicIp='example',portMappings={'22':22})), \
                patch.object(deploy.subprocess,'run',side_effect=[subprocess.TimeoutExpired('ssh',15),success]) as run, \
                patch.object(deploy.time,'sleep'),patch.object(deploy,'save') as save:
            deploy.wait_ssh('newpod')
            self.assertEqual(run.call_count,2)
            self.assertEqual(save.call_args.args[0],'ssh_endpoint.json')


if __name__=='__main__':unittest.main()
