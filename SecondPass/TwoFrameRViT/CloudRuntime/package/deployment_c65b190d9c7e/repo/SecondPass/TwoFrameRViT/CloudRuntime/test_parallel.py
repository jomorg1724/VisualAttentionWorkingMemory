"""Exact authorized parallel pod gate, no provider calls."""
import unittest
import deploy
class ParallelTest(unittest.TestCase):
    def test_only_exact_existing_kda_is_allowed(self):
        deploy.require_no_parallel([dict(id='pa0ko8f2qirisy',status='RUNNING')])
        for pod in (dict(id='other',status='RUNNING'),dict(id='pa0ko8f2qirisy',status='UNKNOWN'),dict(id='other')):
            with self.assertRaises(ValueError):deploy.require_no_parallel([pod])
if __name__=='__main__':unittest.main()
