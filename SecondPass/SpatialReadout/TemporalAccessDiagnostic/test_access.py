"""CPU routing contract: no fits, accelerator calls, or actual test episodes."""
import importlib.util
import unittest
import numpy as np

class AccessContract(unittest.TestCase):
    def test_final_only_routes_every_component_to_report(self):
        name='SecondPass.SpatialReadout.TemporalAccessDiagnostic.run'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Matched diagnostic implementation missing')
        from .run import component_feature
        a={'gru':np.arange(2*5*64*7*7,dtype=np.float32).reshape(2,5,64,7,7)}
        expected=a['gru'][:,-1].reshape(2,-1)
        for task in ['orientation_cued','spatial_binding']:
            for component in ['sample','probe','location','sign']:
                np.testing.assert_array_equal(component_feature(a,'gru','final_only',task,component,0),expected)
                short={'gru':a['gru'][:,-1:]}
                np.testing.assert_array_equal(component_feature(short,'gru','final_only',task,component,0),expected)
        for component,t in [('sample',2),('probe',-1),('location',3)]:
            np.testing.assert_array_equal(component_feature(a,'gru','time_separated','spatial_binding',component,0),a['gru'][:,t].reshape(2,-1))

if __name__=='__main__':unittest.main()
