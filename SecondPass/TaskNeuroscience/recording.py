"""Instrument the authoritative forward without changing its equations.

The generated function is saved with the analysis provenance. Exact native
parity and prefix causality are mandatory before production extraction.
"""
import inspect
import textwrap
import torch
from torch.nn import functional as F
from SecondPass.MemoryGuidedSearch.model import ExpandedSuite
from SecondPass.StabilizedAttentionKrauzlis.model import StabilizedAttentionKrauzlis, warp, causal_average


SOURCE = textwrap.dedent(inspect.getsource(StabilizedAttentionKrauzlis.forward))
SOURCE = SOURCE.replace('def forward(self,images,return_details=False,attention_bias=None):',
    'def recorded_forward(self,images,task,return_details=False,attention_bias=None,record=False,lesion=None):')
SOURCE = SOURCE.replace('history=[]; gate_history=[]; errors=[]; previous_flow=None',
    'history=[]; gate_history=[]; errors=[]; previous_flow=None\n    recordings={k:[] for k in ("appearance","fusion","spatial","cls","comparison","probabilities")}')
SOURCE = SOURCE.replace("bias=attention_bias[step] if attention_bias.ndim==2 else attention_bias\n            if bias.shape!=(100,): raise ValueError('Intervention must be [100] or [time,100]')",
    "bias=attention_bias[:,step] if attention_bias.ndim==3 else attention_bias[step] if attention_bias.ndim==2 else attention_bias\n            if bias.shape not in ((100,),(b,100)): raise ValueError('Bad analysis intervention shape')")
SOURCE = SOURCE.replace('if return_details: gate_history.append',
    '''if lesion is not None and step==lesion['time']:
            if lesion['state']=='spatial':field=field*lesion['gain']
            elif lesion['state']=='cls':cls=cls*lesion['gain']
            elif lesion['state']=='comparison':comparison_memory=comparison_memory*lesion['gain']
            else:raise ValueError(lesion)
        if record:
            for name,v in [('appearance',app[:,step]),('fusion',visual),('spatial',field.flatten(2).transpose(1,2)),('cls',cls),('comparison',comparison_memory)]:
                recordings[name].append(v.detach().cpu())
            recordings['probabilities'].append(self.heads[task](self.readout(torch.cat((cls,comparison_memory),-1))).softmax(-1).detach().cpu())
        if return_details: gate_history.append''')
SOURCE = SOURCE.replace('output=self.readout(torch.cat((cls,comparison_memory),-1))',
    'output=self.heads[task](self.readout(torch.cat((cls,comparison_memory),-1)))')
SOURCE = SOURCE.replace('prediction_loss=prediction_loss,cls=cls,temperature=tau,comparison_memory=comparison_memory,',
    'recordings={k:torch.stack(v,1) for k,v in recordings.items()} if record else {},\n            prediction_loss=prediction_loss,cls=cls,temperature=tau,comparison_memory=comparison_memory,')
_namespace=dict(torch=torch,F=F,warp=warp,causal_average=causal_average)
exec(compile(SOURCE,'<audited-analysis-forward>','exec'),_namespace)


class RecordedSuite(ExpandedSuite):
    forward=_namespace['recorded_forward']
