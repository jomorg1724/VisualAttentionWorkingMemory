"""Gradient signal-to-noise at initialisation: how consistent is the loss gradient across independent batches?

For each task, K independent batches of B episodes from the validation stream; per-batch gradient g_k (all parameters
except the heads' bias). SNR = |mean_k g_k| / sqrt(mean_k |g_k - mean g|^2 / K) is the norm of the expected gradient in units of
its standard error; a value near 1 means the expected gradient is not distinguishable from noise at this batch size.
Also reports the mean pairwise cosine between batch gradients. Usage: python -m WorkingMemory.PlainBaseline.gradient_snr [--stack 3] [--center]
"""
import argparse,json,time
import numpy as np,torch
from torch.nn import functional as F
from WorkingMemory.PlainBaseline.baseline import PlainBaseline,RUNG1_CELLS,ALL_TASK_CLASSES,make_stream
from WorkingMemory.SpatialTaskBattery import stimuli as S
ap=argparse.ArgumentParser();ap.add_argument('--stack',type=int,default=3);ap.add_argument('--center',action='store_true');ap.add_argument('--batches',type=int,default=16);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--tasks',default='all');ap.add_argument('--chunk',type=int,default=8);a=ap.parse_args()
dev='cuda';out={}
tasks=list(S.TASK_CLASSES) if a.tasks=='all' else a.tasks.split(',')
for task in tasks:
    torch.manual_seed(a.seed);m=PlainBaseline(ALL_TASK_CLASSES,stack=a.stack,center=a.center).to(dev);m.train()
    params=[p for n,p in m.named_parameters() if not n.startswith('heads.') or n.startswith(f'heads.{task}.weight')]
    stream=make_stream(63973001,'val',task);cells=RUNG1_CELLS[task];G=[];t=time.time()
    D=[];perm_rng=np.random.default_rng(a.seed)
    def grad(x,y):
        m.zero_grad()
        for i in range(0,a.batch,a.chunk):
            (F.cross_entropy(m(x[i:i+a.chunk].to(dev),task),y[i:i+a.chunk].to(dev))*(len(y[i:i+a.chunk])/a.batch)).backward()
        return torch.cat([p.grad.flatten() for p in params]).detach().cpu().double()
    for k in range(a.batches):
        x,y,_=stream.batch(a.batch,task,cells[k%len(cells)]);g=grad(x,y);G.append(g)
        # Label-shuffle control: the same images with permuted labels. g - g_shuffled keeps only the label-informative part.
        ys=y[torch.as_tensor(perm_rng.permutation(len(y)),dtype=torch.long)];D.append(g-grad(x,ys))
    G=torch.stack(G);mu=G.mean(0);dev_=G-mu;se=float(torch.sqrt((dev_.square().sum(1)).mean()/a.batches))
    cos=[float(F.cosine_similarity(G[i],G[j],dim=0)) for i in range(a.batches) for j in range(i+1,a.batches)]
    D=torch.stack(D);muD=D.mean(0);seD=float(torch.sqrt(((D-muD).square().sum(1)).mean()/a.batches))
    cosD=[float(F.cosine_similarity(D[i],D[j],dim=0)) for i in range(a.batches) for j in range(i+1,a.batches)]
    label_signal=dict(snr=float(muD.norm()/seD),mean_norm=float(muD.norm()),per_batch_norm=float(D.norm(dim=1).mean()),pairwise_cosine_mean=float(np.mean(cosD)),pairwise_cosine_sd=float(np.std(cosD)),
        fraction_of_total_gradient=float(D.norm(dim=1).mean()/G.norm(dim=1).mean()))
    # Same statistic restricted to the head weights (first-order linear signal) and to the CNN.
    names=[n for n,p in m.named_parameters() if not n.startswith('heads.') or n.startswith(f'heads.{task}.weight')];sizes=[p.numel() for n,p in m.named_parameters() if n in names]
    idx=np.cumsum([0]+sizes);sub={}
    for label,pred in (('head',lambda n:n.startswith('heads.')),('cnn',lambda n:n.startswith('cnn.')),('gru',lambda n:n.startswith('gru.'))):
        sel=torch.cat([torch.arange(idx[i],idx[i+1]) for i,n in enumerate(names) if pred(n)]);g=G[:,sel];mu_=g.mean(0);se_=float(torch.sqrt(((g-mu_).square().sum(1)).mean()/a.batches))
        sub[label]=dict(snr=float(mu_.norm()/se_),mean_norm=float(mu_.norm()),per_batch_norm=float(g.norm(dim=1).mean()))
    out[task]=dict(snr=float(mu.norm()/se),mean_grad_norm=float(mu.norm()),per_batch_grad_norm=float(G.norm(dim=1).mean()),pairwise_cosine_mean=float(np.mean(cos)),pairwise_cosine_sd=float(np.std(cos)),by_module=sub,label_signal=label_signal,seconds=time.time()-t)
    print(f"{task:22s} SNR {out[task]['snr']:.2f} |E g| {out[task]['mean_grad_norm']:.4f} per-batch |g| {out[task]['per_batch_grad_norm']:.4f} cos {np.mean(cos):+.3f}+-{np.std(cos):.3f} | head SNR {sub['head']['snr']:.2f} cnn SNR {sub['cnn']['snr']:.2f} gru SNR {sub['gru']['snr']:.2f} | LABEL part: frac {label_signal['fraction_of_total_gradient']:.2f} SNR {label_signal['snr']:.2f} cos {label_signal['pairwise_cosine_mean']:+.3f}+-{label_signal['pairwise_cosine_sd']:.3f}",flush=True)
json.dump(dict(args=vars(a),results=out),open(f'WorkingMemory/PlainBaseline/gradient_snr_stack{a.stack}{"_center" if a.center else ""}.json','w'),indent=1)
