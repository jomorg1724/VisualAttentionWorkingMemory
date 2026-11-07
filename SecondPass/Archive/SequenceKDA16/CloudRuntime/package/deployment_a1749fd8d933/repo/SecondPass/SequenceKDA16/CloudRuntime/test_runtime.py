import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).parent

def load():
    assert (ROOT/'remote_owner.py').exists(), 'one-shot remote launcher missing'
    spec=importlib.util.spec_from_file_location('owner90',ROOT/'remote_owner.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class OwnerTests(unittest.TestCase):
    def test_quote_and_budget(self):
        m=load(); b=dict(cap_started=100,deadline=28300,hard_deadline=28900,wall_cap_seconds=28800,retrieval_reserve_seconds=600,max_usd=5)
        q=dict(gpu='NVIDIA A40',gpu_count=1,compute_hourly_usd=.4,storage_reserve_usd=1,verified_live_rates=True,observed=99)
        m.validate_contract(b,q,100)
        for change in [dict(hard_deadline=29000),dict(max_usd=8)]:
            with self.assertRaises(ValueError):m.validate_contract(dict(b,**change),q,100)
        with self.assertRaises(ValueError):m.validate_contract(b,dict(q,compute_hourly_usd=.6),100)
        with self.assertRaises(ValueError):m.validate_contract(b,dict(q,observed=-4000),100)
    def test_no_restart_and_failed_setup(self):
        m=load()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'run').mkdir(); (root/'run/owner_claim.json').write_text('{}')
            with self.assertRaises(FileExistsError):m.claim(root/'run')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'run').mkdir()
            with patch.object(m,'preflight',side_effect=ValueError('missing guard')):
                self.assertEqual(m.execute(root),1)
            self.assertTrue(json.loads((root/'run/DEPLOYMENT_FAILED').read_text())['stop'])
    def test_chain_waits_and_manifests_before_return(self):
        m=load()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'run').mkdir(); calls=[]
            def command(root,*args): calls.append(args[0])
            with patch.object(m,'preflight'),patch.object(m,'command',side_effect=command),patch.object(m,'finalize') as finalize:
                self.assertEqual(m.execute(root),0)
            self.assertEqual(calls,['prepare','supervise-profile','pin','supervise-run'])
            finalize.assert_called_once()

if __name__=='__main__':unittest.main()
