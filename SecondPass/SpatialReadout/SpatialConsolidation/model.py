"""Single terminal spatial transformer; recurrence and dense decoder unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from SecondPass.SpatialReadout.model import SpatialReadout


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(64))
        self.eps=1e-6

    def forward(self,x):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)*self.weight


class TerminalSpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm=RMSNorm()
        self.ffn_norm=RMSNorm()
        self.final_norm=RMSNorm()
        self.qkv=nn.Linear(64,192,bias=False)
        self.attention_out=nn.Linear(64,64,bias=False)
        self.gate=nn.Linear(64,176,bias=False)
        self.up=nn.Linear(64,176,bias=False)
        self.down=nn.Linear(176,64,bias=False)
        # Token index = row*7+column; 16 frequencies per axis.
        row,col=torch.meshgrid(torch.arange(7),torch.arange(7),indexing='ij')
        omega=10000.0**(-torch.arange(16,dtype=torch.float32)/16)
        r=row.flatten()[:,None]*omega
        c=col.flatten()[:,None]*omega
        self.register_buffer('position',torch.cat((r.sin(),r.cos(),c.sin(),c.cos()),-1)[None])

    def forward(self,field):
        if field.ndim!=4 or tuple(field.shape[1:])!=(64,7,7):
            raise ValueError('Expected B,64,7,7 final memory')
        x=field.flatten(2).transpose(1,2)+self.position
        q,k,v=self.qkv(self.attention_norm(x)).reshape(x.shape[0],49,3,4,16).permute(2,0,3,1,4).unbind(0)
        attention=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False)
        x=x+self.attention_out(attention.transpose(1,2).reshape(x.shape[0],49,64))
        z=self.ffn_norm(x)
        x=x+self.down(F.silu(self.gate(z))*self.up(z))
        return self.final_norm(x).transpose(1,2).reshape(field.shape)


class SpatialConsolidation(SpatialReadout):
    def __init__(self,task_classes):
        super().__init__(task_classes)
        self.consolidation=TerminalSpatialBlock()

    def forward(self,images,task):
        hidden=self.recurrent_states(images)[-1]
        hidden=self.consolidation(hidden)
        return self.heads[task](self.readout(hidden.flatten(1)).relu())
