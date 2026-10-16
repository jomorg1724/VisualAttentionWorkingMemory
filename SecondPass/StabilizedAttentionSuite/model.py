"""Unchanged stabilized backbone and shared FFN, with task-specific output layers."""
from torch import nn
from SecondPass.StabilizedAttentionKrauzlis.model import StabilizedAttentionKrauzlis
from SecondPass.TaskSuite.suite import task_classes

class StabilizedAttentionSuite(StabilizedAttentionKrauzlis):
    def __init__(self,width=64):
        super().__init__(width)
        self.readout=self.readout[:-1]
        self.heads=nn.ModuleDict({task:nn.Linear(64,classes) for task,classes in task_classes().items()})

    def forward(self,images,task,return_details=False,attention_bias=None):
        result=super().forward(images,return_details=return_details,attention_bias=attention_bias)
        if return_details:
            features,details=result
            return self.heads[task](features),details
        return self.heads[task](result)
