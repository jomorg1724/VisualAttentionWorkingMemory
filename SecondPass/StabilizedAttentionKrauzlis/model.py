"""Explicit local temporal comparisons routed by one shared spatial-attention head.

Single CLS-query recurrent attention; no second head or ungated decision path.
All maps and optional logit interventions refer to a uniform10x10 visual grid.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .memory import SpatialGRU, CLSGRU


class LocalMotionBank(nn.Module):
    def __init__(self):
        super().__init__()
        w=torch.hann_window(24,periodic=False)
        self.register_buffer('window',w[:,None]*w[None,:])
        d=torch.arange(-4,5); dy,dx=torch.meshgrid(d,d,indexing='ij')
        self.register_buffer('indices',(dy.remainder(32)*32+dx.remainder(32)).flatten())
        self.register_buffer('offsets',torch.stack((dx,dy),-1).flatten(0,1).float())

    def forward(self,gray):
        b,t,_,h,w=gray.shape
        patches=F.unfold(F.pad(gray.reshape(b*t,1,h,w)-.5,(7,7,7,7)),24,stride=10)
        patches=patches.transpose(1,2).reshape(b,t,100,24,24)
        patches=(patches-patches.mean((-1,-2),keepdim=True))*self.window
        spectra=torch.fft.rfft2(patches,s=(32,32))
        energy=patches.square().sum((-1,-2))
        costs=[]; valid=[]
        for lag in (1,2,4):
            ix=(torch.arange(t,device=gray.device)-lag).clamp_min(0)
            corr=torch.fft.irfft2(spectra[:,ix].conj()*spectra,s=(32,32))
            norm=(energy[:,ix]*energy).sqrt()
            costs.append(corr.flatten(-2)[...,self.indices]/norm[...,None].clamp_min(1e-8))
            valid.append(norm/(norm+.01))
        return torch.stack(costs,2),torch.stack(valid,2)


def warp(image,flow):
    b,_,h,w=image.shape
    flow=F.interpolate(flow,(h,w),mode='bilinear',align_corners=False)
    yy,xx=torch.meshgrid(torch.arange(h,device=image.device),torch.arange(w,device=image.device),indexing='ij')
    grid=torch.stack((2*(xx-flow[:,0])/(w-1)-1,2*(yy-flow[:,1])/(h-1)-1),-1)
    return F.grid_sample(image,grid,align_corners=True,padding_mode='border')


def causal_average(population,reliability,width,gap):
    """Normalized per-location causal boxcar; gap counts excluded recent frames.

    recent=width4/gap0, reference=width8/gap8 gives four intervening frames.
    This runs at EVERY timestep and receives no task phases or event times.
    """
    weighted=population*reliability[...,None]
    numerator=torch.cat((torch.zeros_like(weighted[:,:1]),weighted.cumsum(1)),1)
    denominator=torch.cat((torch.zeros_like(reliability[:,:1]),reliability.cumsum(1)),1)
    end=(torch.arange(population.shape[1],device=population.device)+1-gap).clamp_min(0)
    start=(end-width).clamp_min(0)
    mass=(denominator[:,end]-denominator[:,start]).clamp_min(0)
    average=(numerator[:,end]-numerator[:,start])/mass[...,None].clamp_min(1e-6)
    return average,mass/width


class StabilizedAttentionKrauzlis(nn.Module):
    def __init__(self,width=64):
        super().__init__(); self.width=width
        self.motion=LocalMotionBank()
        self.refine=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.GELU(),nn.Conv2d(4,1,3,padding=1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
        self.log_temperature=nn.Parameter(torch.full((3,),math.log(.04)))
        self.trace_rate=nn.Parameter(torch.tensor(math.log(.1/.9)))
        self.appearance=nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((10,10)))
        #32 appearance + RGB means/contrast4 +3x81 motion +3 confidence
        # +3 local population differences +3 inferred xy velocities +1 prediction error.
        self.fusion=nn.Linear(32+4+243+3+3+6+1,width)
        self.position=nn.Parameter(torch.randn(1,100,width)*.02)
        self.cls_token=nn.Parameter(torch.randn(1,width)*.02)
        self.key_norm=nn.LayerNorm(width); self.memory_norm=nn.LayerNorm(width)
        self.query_norm=nn.LayerNorm(width)
        self.attention_scale_logit=nn.Parameter(torch.tensor(math.log(7/8)))
        self.query=nn.Linear(width,width,bias=False); self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width); self.memory_value=nn.Linear(width,width,bias=False)
        self.ffn=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),nn.Linear(256,width))
        self.spatial=SpatialGRU(width); self.cls=CLSGRU(width)
        self.comparison_width=3*3*81+6
        self.readout=nn.Sequential(nn.Linear(width+self.comparison_width,64),nn.GELU(),nn.Linear(64,2))

    def forward(self,images,return_details=False,attention_bias=None):
        if images.ndim!=5 or images.shape[2] not in (1,3) or tuple(images.shape[-2:])!=(100,100):
            raise ValueError('Need a native RGB/grayscale100x100 movie')
        b,t=images.shape[:2]
        rgb=images.expand(-1,-1,3,-1,-1) if images.shape[2]==1 else images
        gray=rgb.mean(2,keepdim=True)
        # Fixed comparisons have no learned parameters. They are computed once
        # per movie; every lag is causal and first-frame startup repeats frame0.
        costs,valid=self.motion(gray)
        corr=costs.reshape(-1,1,9,9)
        refined=(corr+.1*self.refine(corr)).reshape(b,t,3,100,81)
        tau=self.log_temperature.exp().clamp(.015,.2)
        pop=(refined/tau[None,None,:,None,None]).softmax(-1)
        reference,reference_conf=causal_average(pop,valid,8,8)
        observation,observation_conf=causal_average(pop,valid,4,0)
        confidence=(reference_conf*observation_conf).clamp_min(0).sqrt()
        difference=(observation-reference).abs()
        # Compare at each location BEFORE attention pools locations. These
        # uncompressed channels share exactly the same alpha as visual values.
        local_comparison=torch.cat((
            (10*reference*confidence[...,None]).transpose(2,3).flatten(3),
            (10*observation*confidence[...,None]).transpose(2,3).flatten(3),
            (10*difference*confidence[...,None]).transpose(2,3).flatten(3),
            (difference.sum(-1)*confidence).transpose(2,3),
            ((1-F.cosine_similarity(reference,observation,dim=-1))*confidence).transpose(2,3)),dim=-1)
        displacement=torch.einsum('btlsk,kd->btlsd',pop,self.motion.offsets)
        velocity=displacement/gray.new_tensor([1.,2.,4.])[None,None,:,None,None]
        app=self.appearance((rgb-.5).reshape(b*t,3,100,100)).reshape(b,t,32,100).transpose(2,3)
        means=F.avg_pool2d((rgb-.5).reshape(b*t,3,100,100),10).reshape(b,t,3,100).transpose(2,3)
        energy=F.avg_pool2d((gray-.5).square().reshape(b*t,1,100,100),10).reshape(b,t,100)
        raw=torch.cat((means,energy.sqrt()[...,None]),-1)
        trace=pop.new_full((b,3,100,81),1/81)
        field=pop.new_zeros(b,self.width,10,10); cls=pop.new_zeros(b,self.width)
        history=[]; gate_history=[]; errors=[]; previous_flow=None
        comparison_memory=pop.new_zeros(b,self.comparison_width)
        for step in range(t):
            if step:
                prediction=warp(gray[:,step-1],previous_flow)
                mse=F.avg_pool2d((prediction-gray[:,step]).square(),10).flatten(1)
                denominator=(.5*(energy[:,step-1]+energy[:,step])).clamp_min(1e-4)
                error=mse/denominator
                # Pixel-derived reliability only; no event or phase metadata.
                weight=(energy[:,step-1]*energy[:,step]).sqrt()
                weight=weight/(weight+1e-4)
                errors.append((error*weight).mean())
            else:
                error=energy[:,step]*0
            difference=(pop[:,step]-trace).abs().sum(-1).transpose(1,2)
            motion_features=(10*pop[:,step]*valid[:,step,...,None]).transpose(1,2).flatten(2)
            features=torch.cat((app[:,step],raw[:,step],motion_features,
                valid[:,step].transpose(1,2),difference,
                velocity[:,step].transpose(1,2).flatten(2),error[...,None]),-1)
            visual=self.fusion(features)+self.position
            old=field.flatten(2).transpose(1,2)
            key=self.key(self.key_norm(visual)+self.memory_norm(old))
            query=self.query(self.query_norm(cls+self.cls_token))
            key=F.normalize(key,dim=-1,eps=1e-6)
            query=F.normalize(query,dim=-1,eps=1e-6)
            attention_scale=1+15*self.attention_scale_logit.sigmoid()
            logits=attention_scale*(key*query[:,None]).sum(-1)
            if attention_bias is not None:
                bias=attention_bias[step] if attention_bias.ndim==2 else attention_bias
                if bias.shape!=(100,): raise ValueError('Intervention must be [100] or [time,100]')
                logits=logits+bias
            alpha=logits.softmax(-1)
            value=self.value(visual)+self.memory_value(old)
            value=value+self.ffn(value)
            # The SAME alpha routes all appearance/motion channels to the CLS
            # and modulates spatial writes. No global comparison bypass exists.
            selected=(alpha[...,None]*value).sum(1)
            selected_comparison=(alpha[...,None]*local_comparison[:,step]).sum(1)
            comparison_memory=.8*comparison_memory+.2*selected_comparison
            write_attention=(alpha/(alpha+1/alpha.shape[-1])).reshape(b,1,10,10)
            field,spatial_stats=self.spatial(value.transpose(1,2).reshape(b,self.width,10,10),field,write_attention,return_details)
            cls,cls_stats=self.cls(selected,cls,return_details)
            if return_details: gate_history.append(torch.cat((spatial_stats,cls_stats)))
            trace=trace+self.trace_rate.sigmoid()*valid[:,step,...,None]*(pop[:,step]-trace)
            previous_flow=velocity[:,step].mean(1).transpose(1,2).reshape(b,2,10,10)
            if return_details: history.append(alpha)
        output=self.readout(torch.cat((cls,comparison_memory),-1))
        prediction_loss=torch.stack(errors).mean() if errors else output.sum()*0
        if return_details:
            return output,dict(attention=torch.stack(history,1).reshape(b,t,10,10),
                prediction_loss=prediction_loss,cls=cls,temperature=tau,comparison_memory=comparison_memory,
                gate_statistics=torch.stack(gate_history).mean(0),attention_scale=attention_scale)
        return output
