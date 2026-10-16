"""Patch-local gate normalization and bounded attention-dependent memory writes."""
import torch
from torch import nn

GATE_NAMES=tuple(f'{module}_{metric}' for module in ('spatial','cls') for metric in
    ('write_mean','write_low','write_high','reset_low','reset_high','effective_write_mean','candidate_saturated'))

class PatchNorm(nn.Module):
    def __init__(self,width):
        super().__init__(); self.norm=nn.LayerNorm(width)

    def forward(self,x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)

def statistics(write,reset,effective,candidate):
    w,r,e,c=(v.detach() for v in (write,reset,effective,candidate))
    return torch.stack((w.mean(),(w<.01).float().mean(),(w>.99).float().mean(),
        (r<.01).float().mean(),(r>.99).float().mean(),e.mean(),(c.abs()>.99).float().mean()))

class SpatialGRU(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.gates=nn.Conv2d(2*width,2*width,3,padding=1)
        self.candidate=nn.Conv2d(2*width,width,3,padding=1)
        self.write_norm=PatchNorm(width);self.reset_norm=PatchNorm(width)
        self.candidate_norm=PatchNorm(width)
        nn.init.zeros_(self.gates.bias);nn.init.zeros_(self.candidate.bias)

    def forward(self,visual,previous,attention_gate,return_details=False):
        w,r=self.gates(torch.cat((visual,previous),1)).chunk(2,1)
        write=self.write_norm(w).sigmoid();reset=self.reset_norm(r).sigmoid()
        candidate=self.candidate_norm(self.candidate(torch.cat((visual,reset*previous),1))).tanh()
        effective=attention_gate*write
        result=previous+effective*(candidate-previous)
        return result,statistics(write,reset,effective,candidate) if return_details else None

class CLSGRU(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.gates=nn.Linear(2*width,2*width);self.candidate=nn.Linear(2*width,width)
        self.write_norm=nn.LayerNorm(width);self.reset_norm=nn.LayerNorm(width)
        self.candidate_norm=nn.LayerNorm(width)
        nn.init.zeros_(self.gates.bias);nn.init.zeros_(self.candidate.bias)

    def forward(self,visual,previous,return_details=False):
        w,r=self.gates(torch.cat((visual,previous),-1)).chunk(2,-1)
        write=self.write_norm(w).sigmoid();reset=self.reset_norm(r).sigmoid()
        candidate=self.candidate_norm(self.candidate(torch.cat((visual,reset*previous),-1))).tanh()
        result=previous+write*(candidate-previous)
        return result,statistics(write,reset,write,candidate) if return_details else None
