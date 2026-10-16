"""Bounded CPU behavior checks and a pixel-only correspondence diagnostic."""
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import generate, training_pair
from .model import CorrespondencePredictiveMemory, LocalCorrespondence, warp
from .loss import objective
from SecondPass.OutcomeContrastiveRViT.train import summary


def main():
    torch.set_num_threads(2); torch.manual_seed(194211)
    front = LocalCorrespondence()
    frame = generate(0)[0][0:1]
    moved = torch.roll(frame, (2, -3), (-2, -1))
    movies = torch.stack((frame, moved), 1)
    corr = front(movies).mean((-1,-2))[:,0]
    estimate = front.offset_xy[corr.argmax(1)]
    assert torch.equal(estimate, torch.tensor([[-3., 2.]])), estimate
    flow = torch.tensor([-3.,2.]).reshape(1,2,1,1).expand(1,2,4,4)
    assert torch.allclose(warp(frame,flow),moved,atol=3e-6)
    # Identical evaluation nuisance draws across change/no-change; training views differ.
    a=generate(12000000,'val',True)[0]; b=generate(12000001,'val',True)[0]
    assert torch.equal(a[:4],b[:4]) and not torch.equal(a[4:],b[4:])
    rows=[training_pair(49000000+i) for i in range(4)]
    stacked=torch.stack([r[0] for r in rows]); frames=torch.cat((stacked[:,0],stacked[:,1]))
    assert not torch.equal(stacked[:,0],stacked[:,1])
    labels=torch.tensor([r[1] for r in rows])
    model=CorrespondencePredictiveMemory(); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    initial={k:v.clone() for k,v in model.named_parameters()}
    for _ in range(2):
        opt.zero_grad(); logits,cls,details=model(frames,True); loss,parts=objective(logits,details,labels)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    changed=[n for n,p in model.named_parameters() if not torch.equal(p,initial[n])]
    assert len(changed)==len(initial), set(initial)-set(changed)
    with torch.no_grad():
        _,_,d1=model(frames[:2],True)
        altered=frames[:2].clone(); altered[:,2:]=.5
        _,_,d2=model(altered,True)
        assert torch.equal(d1['flow'][:,0],d2['flow'][:,0])
    # No fitting of the front-end or classifier: fixed pixel correspondence only.
    # Validation selects threshold; independent test uses it unchanged.
    def scores(split,n):
        out=[]; labels=[]
        with torch.no_grad():
            for start in range(0,n,8):
                rows=[generate((60000000 if split=='val' else 72000000)+i,split,True) for i in range(start,start+8)]
                cost=front(torch.stack([r[0] for r in rows]))
                p=(cost/.04).softmax(2).mean((-1,-2))
                pre,post=p[:,:2].mean(1),p[:,3:].mean(1)
                out.extend((pre-post).abs().sum(1).tolist()); labels.extend(r[1] for r in rows)
        return np.array(out),np.array(labels)
    vs,vy=scores('val',192)
    candidates=np.r_[-1.,np.unique(vs),3.]
    threshold=max(candidates,key=lambda t: summary(vs-t,vy)['balanced_accuracy'])
    ts,ty=scores('test',384)
    result=dict(status='passed',device='cpu',parameter_count=sum(p.numel() for p in model.parameters()),
        learned_tensors=len(initial),all_parameters_changed_after_two_steps=True,
        local_correlation_and_warp_sign_correct=True,first_prediction_causal=True,
        paired_nuisances_verified=True,finite_full_objective=float(loss.detach()),
        diagnostic=dict(description='Fixed pixel-correspondence population difference; not a trained model result',
            val_n=192,test_n=384,threshold=float(threshold),test=summary(ts-threshold,ty)))
    Path(__file__).with_name('check_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
