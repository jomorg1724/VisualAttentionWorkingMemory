"""Attention analysis for a trained KDA-accumulator model (Section 6 of SecondPass/KDA_paper): gate maps, implicit
attention weights over past frames, state probes, and interventions. Works on `AccumulatorBaseline` with accumulator='kda';
gate maps and interventions also run on 'convgru' (its gates are write/reset) where noted.

Usage: python -m WorkingMemory.PlainBaseline.analysis.kda_probe --checkpoint <terminal.pt> --out <dir> [--n 128] [--delays 0,4,12,24]
Writes <out>/kda_probe.json, <out>/gates_D<d>.npz (per-frame maps) and <out>/kda_probe.md.

Definitions (per site, per head; k, q unit vectors in R^8, alpha in (0,1)^8, beta in (0,1), S in R^{8x16}):
  A_t = (I - beta_t k_t k_t^T) Diag(alpha_t)              transition applied to the old state
  S_t = A_t S_{t-1} + beta_t k_t v_t^T ;  o_t = S_t^T q_t
  w_{t,tau} = beta_tau q_t^T (A_t ... A_{tau+1}) k_tau      implicit attention of frame t on frame tau (exact)
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from WorkingMemory.PlainBaseline.analysis.psychometric import load_model
from WorkingMemory.PlainBaseline.analysis.psych_stream import PsychOrientationStream
from WorkingMemory.SpatialTaskBattery.stimuli import CENTERS
from WorkingMemory.BatteryAudit.observers import balanced_accuracy

def site_masks(h,w):
    """Boolean masks (h,w) for each of the four Gabor locations at a map of size h x w (100 px -> h)."""
    yy,xx=np.mgrid[:h,:w];sx=100/w;sy=100/h;masks=[]
    for (cx,cy) in CENTERS:masks.append(((xx+.5)*sx-cx)**2+((yy+.5)*sy-cy)**2<=12**2)
    return masks

class Recorder(torch.nn.Module):
    """Wraps a SpatialKDA module to record q,k,v,alpha,beta and state at each call, and to apply interventions."""
    def __init__(self,module):
        super().__init__();self.m=module;self.frames=[];self.clamp_beta_frames=set();self.clamp_alpha_frames=set();self.reset_before=set();self.t=0;self.enabled=True
    def forward(self,current,state=None):
        m=self.m;batch,_,height,width=current.shape
        packed=m.inputs(current).view(batch,2,41,height,width).permute(0,3,4,1,2)
        q=F.normalize(packed[...,:8],dim=-1,eps=1e-6);k=F.normalize(packed[...,8:16],dim=-1,eps=1e-6);v=packed[...,16:32]
        alpha=packed[...,32:40].sigmoid();beta=packed[...,40:41].sigmoid()
        if self.t in self.clamp_beta_frames:beta=torch.zeros_like(beta)
        if self.t in self.clamp_alpha_frames:alpha=torch.ones_like(alpha)
        if state is None or self.t in self.reset_before:state=current.new_zeros(batch,height,width,2,8,16)
        from PreAttentiveVision.TemporalIntegration.accumulators import kda_update
        result,state=kda_update(state,q,k,v,alpha,beta)
        if self.enabled:self.frames.append(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),alpha=alpha.detach().cpu(),beta=beta.detach().cpu(),S=state.detach().cpu()))
        self.t+=1
        result=result.permute(0,3,4,1,2).reshape(batch,32,height,width)
        return m.output(result),state

def implicit_attention(frames,t):
    """w_{t,tau} for all tau<=t: tensor (B,H,W,heads,t+1). Exact unroll with 8x8 transitions."""
    q=frames[t]['q'];B,H,W,nh,_=q.shape;out=torch.zeros(B,H,W,nh,t+1)
    for tau in range(t+1):
        vec=frames[tau]['k']*frames[tau]['beta']          # beta_tau k_tau  (B,H,W,nh,8)
        for s in range(tau+1,t+1):
            a=frames[s]['alpha'];k=frames[s]['k'];b=frames[s]['beta']
            vec=a*vec;vec=vec-b*k*(k*vec).sum(-1,keepdim=True)   # Diag(alpha) then (I - beta k k^T)
        out[...,tau]=(q*vec).sum(-1)
    return out

def run_condition(model,rec_scales,delay,n,device,intervention=None):
    stream=PsychOrientationStream();xs=[];ys=[];metas=[];logits=[]
    for rec in rec_scales:rec.frames=[];rec.t=0;rec.clamp_beta_frames=set();rec.clamp_alpha_frames=set();rec.reset_before=set()
    x,y,meta=stream.batch(n,'orientation_cued',dict(delay=delay));T=x.shape[1];blanks=set(meta[0]['blank_frames']);probe=meta[0]['probe_frame']
    for rec in rec_scales:
        if intervention=='reset_before_probe':rec.reset_before={probe}
        if intervention=='beta_zero_blanks':rec.clamp_beta_frames=blanks
        if intervention=='alpha_one_blanks':rec.clamp_alpha_frames=blanks
    with torch.no_grad():lg=model(x.to(device),'orientation_cued').float().cpu()
    return x,y,meta,lg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);ap.add_argument('--n',type=int,default=128);ap.add_argument('--delays',default='0,4,12,24')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model,args=load_model(a.checkpoint,a.device)
    if args.get('encoder')!='accum' or args.get('accumulator') not in ('kda',):raise SystemExit('this probe needs an accum/kda checkpoint (convgru support: gate maps only, not implemented here)')
    recs=[Recorder(m) for m in model.acc]
    for s,rec in enumerate(recs):model.acc[s]=rec  # ModuleList item replaced by a callable wrapper
    res=dict(checkpoint=a.checkpoint,n=a.n,delays=[],interventions={});report=[]
    for delay in [int(d) for d in a.delays.split(',')]:
        x,y,meta,lg=run_condition(model,recs,delay,a.n,a.device);pred=lg.argmax(1).numpy();ba=balanced_accuracy(y.numpy(),pred,2)
        probe=meta[0]['probe_frame'];blanks=meta[0]['blank_frames'];samples=meta[0]['sample_frames'];targets=np.array([m['target_location'] for m in meta])
        entry=dict(delay=delay,ba=ba,scales=[])
        for s,rec in enumerate(recs):
            fr=rec.frames;B,H,W=fr[0]['beta'].shape[:3];masks=site_masks(H,W);gabor=np.any(masks,0);cued=np.stack([masks[t] for t in targets]);uncued=np.stack([gabor&~masks[t] for t in targets]);background=np.broadcast_to(~gabor,(B,H,W)).copy()
            beta=torch.stack([f['beta'][...,0].mean(-1) for f in fr]);alpha=torch.stack([f['alpha'].mean((-1,-2)) for f in fr])  # (T,B,H,W)
            def region_mean(field,region):return float((field*torch.as_tensor(region)).sum()/(torch.as_tensor(region).sum()*field.shape[0]+1e-9))
            stats={}
            for name,idx in (('sample',samples),('blank',blanks),('probe',[probe])):
                if not idx:continue
                stats[name]=dict(beta_cued=region_mean(beta[idx],cued),beta_uncued=region_mean(beta[idx],uncued),beta_background=region_mean(beta[idx],background),
                                 alpha_cued=region_mean(alpha[idx],cued),alpha_uncued=region_mean(alpha[idx],uncued),alpha_background=region_mean(alpha[idx],background))
            w=implicit_attention(fr,probe)  # (B,H,W,heads,T)
            prof=lambda region:[float((w[...,tau].abs().mean(-1)*torch.as_tensor(region)).sum()/(torch.as_tensor(region).sum()+1e-9)) for tau in range(probe+1)]
            attention=dict(cued=prof(cued),uncued=prof(uncued),background=prof(background))
            # State probe: linear regression from S at the cued site (mean over the cued mask) to the sample orientation, per frame.
            angles=np.array([m['sample_angles_radians'][m['target_location']] for m in meta]) if 'sample_angles_radians' in meta[0] else None
            probe_r2=None
            if angles is not None and B>=32:
                Y=np.stack([np.cos(2*angles),np.sin(2*angles)],1);probe_r2=[]
                for t in range(len(fr)):
                    S=fr[t]['S'].numpy().reshape(B,H,W,-1);feat=np.stack([S[b][masks[targets[b]]].mean(0) for b in range(B)]);feat=np.c_[feat,np.ones(B)]
                    half=B//2;coef,_,_,_=np.linalg.lstsq(feat[:half],Y[:half],rcond=None);pred_=feat[half:]@coef;r2=1-((Y[half:]-pred_)**2).sum()/((Y[half:]-Y[half:].mean(0))**2).sum();probe_r2.append(float(r2))
            entry['scales'].append(dict(scale=s,map=(H,W),gates=stats,attention_abs_mean=attention,state_probe_r2_by_frame=probe_r2))
            np.savez_compressed(out/f'gates_D{delay}_scale{s}.npz',beta=beta.numpy(),alpha=alpha.numpy(),attention_probe=w.numpy(),targets=targets)
        res['delays'].append(entry)
        inter={}
        for name in ('reset_before_probe','beta_zero_blanks','alpha_one_blanks'):
            if delay==0 and name!='reset_before_probe':continue
            for rec in recs:rec.enabled=False
            x2,y2,m2,lg2=run_condition(model,recs,delay,a.n,a.device,intervention=name);inter[name]=balanced_accuracy(y2.numpy(),lg2.argmax(1).numpy(),2)
            for rec in recs:rec.enabled=True
        res['interventions'][f'D{delay}']=dict(baseline=ba,**inter)
        line=f"D{delay}: BA {ba:.3f}; interventions {inter}; scale0 gates: "+'; '.join(f"{k}: beta cued {v['beta_cued']:.2f} bg {v['beta_background']:.2f}, alpha cued {v['alpha_cued']:.2f}" for k,v in entry['scales'][0]['gates'].items())
        att=entry['scales'][0]['attention_abs_mean']['cued'];line+=f"; probe attention on cued site by frame {[round(v,3) for v in att]}"
        if entry['scales'][0]['state_probe_r2_by_frame']:line+=f"; state->angle R2 by frame {[round(v,2) for v in entry['scales'][0]['state_probe_r2_by_frame']]}"
        print(line,flush=True);report.append(line)
    (out/'kda_probe.json').write_text(json.dumps(res,indent=1),encoding='utf-8');(out/'kda_probe.md').write_text('# KDA probe\n\n'+'\n\n'.join(report)+'\n',encoding='utf-8');print('wrote',out)

if __name__=='__main__':main()
