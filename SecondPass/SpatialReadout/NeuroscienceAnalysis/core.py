"""Frozen hook-only recording and transient emitted-field interventions.
No optimizer, no replacement of deployed computations, no metadata into model.
Arrays: trial, time, y, x, head; coefficient source/read axes explicit.
"""
import numpy as np
import torch
from torch.nn import functional as F
from WorkingMemory.SpatialTaskBattery.stimuli import CENTERS


def masks(side):
    """Translated compact smooth kernels: identical support, mass and peak.
    Feature coordinates use cell centers over 100-pixel raster (approximate RF).
    Last site is central background; all measured sites are retained.
    """
    radius=2 if side==25 else 1
    sigma=1.2 if side==25 else .8
    yy,xx=np.mgrid[:side,:side];out=[]
    for cx,cy in list(CENTERS)+[(50,50)]:
        ix=int(round(cx*side/100-.5));iy=int(round(cy*side/100-.5))
        d=(xx-ix)**2+(yy-iy)**2
        out.append(np.exp(-d/(2*sigma**2))*(d<=radius**2))
    return np.asarray(out,dtype=np.float32)


def epoch_frames(meta,epoch):
    if epoch=='encoding':return [2] # one update, last sample
    if epoch=='retention':
        pure=meta.get('blank_frames',[])[2:]
        return pure[-1:] # one genuinely stimulus-free update
    if epoch=='probe':return [meta['probe_frame']]
    if epoch=='query':return meta['cue_frames'][-1:]
    raise ValueError(epoch)


def observe(model,x,task,meta,record=False,intervention=None,localizer=False):
    """Run native final-only head. Hooks change only finest KDA output emission.
    Its own returned KDA state remains untouched, but downstream recurrence can
    propagate the transient effect into later frames and coarser scales.
    """
    handles=[];pack=[[] for _ in range(3)];reads=[[] for _ in range(3)]
    grug=[];gruh=[];emissions=[];counter=[0];effects=[]
    if record:
        for scale,m in enumerate(model.acc):
            def packed_hook(module,args,y,s=scale):pack[s].append(y.detach().cpu())
            def read_hook(module,args,s=scale):reads[s].append(args[0].detach().cpu())
            handles.append(m.inputs.register_forward_hook(packed_hook))
            handles.append(m.output.register_forward_pre_hook(read_hook))
        handles.append(model.spatial_gru.gates.register_forward_hook(lambda m,a,y:grug.append(y.sigmoid().detach().cpu())))
        handles.append(model.spatial_gru.register_forward_hook(lambda m,a,y:gruh.append(y.detach().cpu())))
    mm=torch.as_tensor(masks(25),device=x.device)
    sites=[]
    if intervention:
        for row in meta:
            loc=row.get('target_location',0)
            sites.append(loc if intervention['site']=='cued' else (loc+1)%4 if intervention['site']=='foil' else 4)
        mask=mm[sites,None]
        active=[set(epoch_frames(row,intervention['epoch'])) for row in meta]
    def emitted_hook(module,args,y):
        t=counter[0];counter[0]+=1
        if localizer and t==2:
            # B,site,channel; local activation RMS uses unpooled values.
            emissions.append(torch.einsum('bchw,shw->bsc',y,mm[:4])/mm[:4].sum((-1,-2))[None,:,None])
            emissions.append((torch.einsum('bchw,shw->bs',y.square(),mm[:4])/(32*mm[:4].sum((-1,-2))[None])).sqrt())
        if not intervention or not intervention['dose']:return y
        on=torch.as_tensor([t in times for times in active],device=y.device,dtype=y.dtype)[:,None,None,None]
        m=on*mask
        if intervention['kind']=='inhibit':delta=-intervention['dose']*m*y
        else:
            direction=torch.as_tensor(intervention['direction'],device=y.device,dtype=y.dtype)[None,:,None,None]
            delta=intervention['dose']*intervention['rms']*m*direction
        effects.append(dict(frame=t,delta_rms=float(delta.square().mean().sqrt().cpu()),local_delta_rms=float((delta.square().sum((1,2,3))/(32*m.square().sum((1,2,3)).clamp_min(1e-9))).sqrt().mean().cpu())))
        return y+delta
    if intervention or localizer:handles.append(model.acc[0].output.register_forward_hook(emitted_hook))
    try:
        with torch.inference_mode():logits=model(x,task)
    finally:
        for h in handles:h.remove()
    data={}
    if localizer:
        data['local_features']=emissions[0].cpu().numpy();data['local_rms']=emissions[1].cpu().numpy()
    if intervention:data['effects']=effects
    if record:
        with torch.inference_mode():
            for scale in range(3):
                pp=torch.stack(pack[scale],1);B,T,_,H,W=pp.shape
                p=pp.reshape(B,T,2,41,H,W).permute(0,1,4,5,2,3)
                q=F.normalize(p[...,:8],dim=-1,eps=1e-6);k=F.normalize(p[...,8:16],dim=-1,eps=1e-6)
                v=p[...,16:32];alpha=p[...,32:40].sigmoid();beta=p[...,40:41].sigmoid()
                # Reverse row propagation computes final-read x all sources O(T).
                r=q[:,-1];coeff=[]
                for tau in reversed(range(T)):
                    dot=(r*k[:,tau]).sum(-1,keepdim=True)
                    coeff.append((beta[:,tau]*dot)[...,0])
                    r=(r-beta[:,tau]*k[:,tau]*dot)*alpha[:,tau]
                coeff=torch.stack(coeff[::-1],1)
                reconstruction=(coeff[...,None]*v).sum(1)
                observed=reads[scale][-1].reshape(B,2,16,H,W).permute(0,3,4,1,2)
                error=(reconstruction-observed).abs().flatten(1).max(1).values
                # Fixed source=min(2,T-1), every read. Pre-source coefficients zero.
                source=min(2,T-1);vec=beta[:,source]*k[:,source];allread=[]
                for t in range(T):
                    if t<source:allread.append(torch.zeros_like(beta[:,t,...,0]));continue
                    if t>source:
                        vec=alpha[:,t]*vec;vec=vec-beta[:,t]*k[:,t]*(k[:,t]*vec).sum(-1,keepdim=True)
                    allread.append((q[:,t]*vec).sum(-1))
                pre=f's{scale}_'
                data.update({pre+'beta':beta[...,0].numpy(),pre+'alpha_mean8':alpha.mean(-1).numpy(),pre+'coeff_final_read':coeff.numpy(),pre+'coeff_source2_all_reads':torch.stack(allread,1).numpy(),pre+'reconstruction_error':error.numpy(),pre+'source_frame':np.array(source),pre+'read_frame':np.array(T-1),pre+'masks':masks(H)})
            gates=torch.stack(grug,1);hidden=torch.stack(gruh,1)
            data['gru_write']=gates[:,:,:64].mean(2).numpy();data['gru_reset']=gates[:,:,64:].mean(2).numpy();data['gru_state_energy']=hidden.square().mean(2).numpy()
    return logits,data
