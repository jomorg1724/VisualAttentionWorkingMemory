"""One-step training-logic diagnostics for the AV-context v2 recipe (handoff section 4.6). No optimizer step is taken.

For a model state (scratch init or a checkpoint) and one fixed microbatch per task:
  1. per-task gradient norms (whole model and per module) under the recipe's loss/5 scaling, before the shared clip;
  2. the norm of the summed five-task gradient, the clip factor, and each task's share of it;
  3. bit-identity of the activation-checkpointed forward/backward against the direct forward/backward;
  4. E/I state norm per timestep on a D24 orientation trial, i.e. how much of the memory survives 24 blank frames.

Usage: python -m WorkingMemory.BatteryAudit.training_diagnostics [--checkpoint path] [--device cuda]
Writes training_diagnostics_<tag>.json next to this file.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from WorkingMemory.AttentionContextComparator.V2.model import ARM,initialize_scratch
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

HERE=Path(__file__).resolve().parent
CFG_PATH=HERE.parents[0]/'AttentionContextComparator'/'V2'/'runs'/'v2_local_20260915_220124'/'fixed_config.json'
MODULES=('encoder','accumulators','attention','memory_input','memory','readout','priority_readout')
CELLS={'orientation_cued':dict(delay=0),'motion_duration_cued':dict(delay=0),'krauzlis_cued_motion':dict(baseline_transitions=20),'spatial_binding':dict(delay=0),'image_recognition':dict(load=4,probe_hold=4)}

def module_of(name):
    for m in MODULES:
        if name.startswith(m+'.'):return m
    return 'other'

def grad_vector(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.parameters()])

def per_module_norms(model):
    acc={}
    for n,p in model.named_parameters():
        if p.grad is None:continue
        acc[module_of(n)]=acc.get(module_of(n),0.)+float(p.grad.square().sum())
    return {k:float(np.sqrt(v)) for k,v in acc.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default=None);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');ap.add_argument('--seed',type=int,default=63973001);a=ap.parse_args()
    cfg=json.loads(CFG_PATH.read_text())['config'];device=torch.device(a.device)
    torch.manual_seed(cfg['model_seed']);np.random.seed(cfg['model_seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,optimizer,lineage=initialize_scratch(ARM,cfg)
    tag='scratch';step=0
    if a.checkpoint:
        saved=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(saved['model'],strict=True);step=int(saved['step']);tag=f'step{step}'
        ckpt_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    else:ckpt_sha=None
    model.to(device);model.train()
    n_params=sum(p.numel() for p in model.parameters());per_module_params={m:sum(p.numel() for n,p in model.named_parameters() if module_of(n)==m) for m in MODULES+('other',)}
    stream=SpatialBatteryStream(a.seed,'val')
    batches={task:stream.batch(cfg['batch_size'],task,cond) for task,cond in CELLS.items()}
    res=dict(tag=tag,step=step,checkpoint=a.checkpoint,checkpoint_sha256=ckpt_sha,device=str(device),batch_size=cfg['batch_size'],clip=cfg['clip'],params=n_params,params_per_module=per_module_params,
        optimizer_groups=[dict(lr=g['lr'],weight_decay=g['weight_decay'],params=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],params_in_optimizer=sum(p.numel() for g in optimizer.param_groups for p in g['params']),cells=CELLS)

    # 1-2. Per-task gradients under the recipe scaling (loss/5), then the sum.
    grads={};info={}
    for task,(x,y,_) in batches.items():
        model.zero_grad(set_to_none=True);t=time.time()
        logits=model(x.to(device),task);loss=torch.nn.functional.cross_entropy(logits,y.to(device));(loss/5).backward()
        g=grad_vector(model).detach().clone();grads[task]=g
        info[task]=dict(loss=float(loss),chance_loss=float(np.log(logits.shape[1])),frames=int(x.shape[1]),grad_norm_scaled=float(g.norm()),grad_norm_unscaled=float(5*g.norm()),
            per_module=per_module_norms(model),seconds=time.time()-t,
            logit_spread=float(logits.detach().std(0).mean()),pred_counts=np.bincount(logits.argmax(1).cpu().numpy(),minlength=logits.shape[1]).tolist())
    total=sum(grads.values());tn=float(total.norm())
    for task,g in grads.items():
        info[task]['share_of_sum_cosine']=float((g@total)/(g.norm()*tn+1e-30));info[task]['projection_onto_sum_fraction']=float((g@total)/(tn*tn+1e-30))
    pair={f'{a_}|{b_}':float((grads[a_]@grads[b_])/(grads[a_].norm()*grads[b_].norm()+1e-30)) for i,a_ in enumerate(grads) for b_ in list(grads)[i+1:]}
    res['per_task']=info;res['summed_gradient']=dict(norm=tn,clip_factor=float(min(1.,cfg['clip']/(tn+1e-6))),effective_lr_scale_after_clip=float(min(1.,cfg['clip']/(tn+1e-6))),pairwise_cosine=pair)

    # 3. Activation-checkpoint identity on the orientation batch.
    x,y,_=batches['orientation_cued'];x=x.to(device);y=y.to(device)
    def run(flag):
        model.activation_checkpoint=flag;model.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
            torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
            logits=model(x,'orientation_cued');torch.nn.functional.cross_entropy(logits,y).backward()
        return logits.detach().clone(),grad_vector(model).detach().clone()
    l1,g1=run(True);l0,g0=run(False);model.activation_checkpoint=cfg['activation_checkpoint']
    res['activation_checkpoint']=dict(logits_max_abs_diff=float((l1-l0).abs().max()),grad_max_abs_diff=float((g1-g0).abs().max()),grad_rel_diff=float((g1-g0).norm()/(g0.norm()+1e-30)),bit_identical=bool(torch.equal(l1,l0) and torch.equal(g1,g0)))

    # 4. E/I state through a 24-frame blank (orientation D24: cue+samples at frames 0-2, blanks 3-26, probe 27).
    xb,yb,mb=stream.batch(4,'orientation_cued',dict(delay=24));model.eval()
    with torch.no_grad():
        _,diag=model(xb.to(device),'orientation_cued',True)
    rates=[float(s[0].norm()) for s in diag['states']];adapt=[float(s[1].norm()) for s in diag['states']];inactive=[r.get('inactive_fraction') for r in diag['records']]
    meta=mb[0];last_sample=meta['sample_frames'][-1];last_blank=meta['blank_frames'][-1]
    mem=model.memory;tau=(1+31*mem.raw_tau_r.sigmoid()).detach().cpu();ta=(4+124*mem.raw_tau_a.sigmoid()).detach().cpu()
    res['ei_blank_decay']=dict(rate_norm_per_frame=rates,adaptation_norm_per_frame=adapt,inactive_fraction_per_frame=inactive,
        rate_norm_ratio_last_blank_over_last_sample=float(rates[last_blank]/(rates[last_sample]+1e-30)),
        tau_r=dict(min=float(tau.min()),median=float(tau.median()),max=float(tau.max())),tau_a=dict(min=float(ta.min()),median=float(ta.median()),max=float(ta.max())),
        passive_decay_24_frames_at_median_tau_r=float((1-(-torch.expm1(-1/tau.median())))**24),
        fixed_trace_decay_24_frames=dict(fast=0.25**24,slow=0.75**24))
    model.train()
    out=HERE/f'training_diagnostics_{tag}.json';out.write_text(json.dumps(res,indent=1),encoding='utf-8');print('wrote',out)
    for task,i in info.items():print(f"{task:22s} loss {i['loss']:.3f} (chance {i['chance_loss']:.3f}) |g|/5 {i['grad_norm_scaled']:.4f} share {i['projection_onto_sum_fraction']:.3f} preds {i['pred_counts']}")
    print('sum norm',f'{tn:.4f}','clip factor',f"{res['summed_gradient']['clip_factor']:.3f}");print('checkpoint identity',res['activation_checkpoint']);print('blank decay ratio',f"{res['ei_blank_decay']['rate_norm_ratio_last_blank_over_last_sample']:.4f}",'tau_r',res['ei_blank_decay']['tau_r'])

if __name__=='__main__':main()
