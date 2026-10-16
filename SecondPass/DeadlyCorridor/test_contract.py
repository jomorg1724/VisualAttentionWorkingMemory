import hashlib,copy
import numpy as np
import torch
from .encoder import StreamingEncoder
from .environment import Corridor,ACTIONS
from .agent import Agent,soft_targets,quantile_loss
from .replay import Replay
from SecondPass.StabilizedAttentionVDA.train import PARENT

torch.set_num_threads(2)

def test_streaming_matches_original_full_prefix_and_reset():
    model=StreamingEncoder().load_suite(PARENT).eval();torch.manual_seed(321);x=torch.rand(1,21,3,100,100);state=None
    with torch.no_grad():
        for t in range(x.shape[1]):
            h,state,a=model.step(x[:,t],state)
            if t in (0,1,4,8,15,20):
                reference,details=model(x[:,:t+1],True)
                torch.testing.assert_close(h,reference,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(a,details['attention'][:,-1],atol=2e-6,rtol=2e-6)
        assert len(state['gray'])==5 and len(state['pop'])==16
        reset,_,_=model.step(x[:,0]);torch.testing.assert_close(reset,model(x[:,:1]),atol=1e-6,rtol=1e-6)

def test_terminal_targets_and_action_mixture_preserved():
    prob=torch.tensor([[.25,.75]]);logp=prob.log();q1=torch.tensor([[[0.,2.],[10.,12.]]]);q2=q1+3
    y,w=soft_targets(torch.tensor([5.]),torch.tensor([.9]),torch.tensor([1.]),prob,logp,q1,q2,.2)
    assert torch.all(y==5) and torch.allclose(w,torch.tensor([[.125,.125,.375,.375]]))
    y,w=soft_targets(torch.tensor([0.]),torch.tensor([1.]),torch.tensor([0.]),prob,logp,q1,q2,0.)
    assert torch.equal(y,torch.tensor([[0.,2.,10.,12.]]))
    pred=torch.tensor([[1.,2.]],requires_grad=True);loss=quantile_loss(pred,y,w);loss.backward();assert torch.isfinite(pred.grad).all()
    # One quantile at its target gives zero loss.
    assert quantile_loss(torch.tensor([[2.]]),torch.tensor([[2.]]),torch.ones(1,1))==0

def test_replay_alignment_and_episode_boundaries():
    replay=Replay(capacity=6)
    for n in (3,4):replay.add(dict(observations=np.zeros((n+1,3,100,100),np.uint8),actions=[0]*n,rewards=[0.]*n,done=[False]*(n-1)+[True],tics=[4]*n))
    assert replay.transitions==4 and len(replay.episodes)==1
    for ep,start,length in replay.sample(20,8):assert start+length<=len(ep['actions']) and len(ep['observations'])==len(ep['actions'])+1
    assert len(ACTIONS)==54 and len({tuple(a) for a in ACTIONS})==54
    assert not np.any(ACTIONS[:,0]*ACTIONS[:,1]) and not np.any(ACTIONS[:,3]*ACTIONS[:,4]) and not np.any(ACTIONS[:,5]*ACTIONS[:,6])

def test_real_engine_shared_representation_and_optimizer_update():
    before=hashlib.sha256(PARENT.read_bytes()).hexdigest();env=Corridor();agent=Agent(PARENT,quantiles=8)
    observations=[];record=dict(observations=[],actions=[],rewards=[],done=[],tics=[]);state=None
    try:
        obs=env.reset(202777);record['observations'].append(obs)
        for i in range(6):
            action,state,attention=agent.act(torch.tensor(obs).float()[None]/255,state)
            obs,r,done,info=env.step(action);assert info['tics']==4
            record['observations'].append(np.zeros((3,100,100),np.uint8) if obs is None else obs)
            for k,v in dict(actions=action,rewards=r,done=done,tics=info['tics']).items():record[k].append(v)
            if done:break
        shared=[]
        hooks=[module[0].register_forward_pre_hook(lambda m,args:shared.append(args[0].detach().clone())) for module in (agent.actor,agent.q1,agent.q2)]
        h,_,_=agent.encoder.step(torch.tensor(record['observations'][0]).float()[None]/255)
        agent.actor(h);agent.q(agent.q1,h);agent.q(agent.q2,h)
        for hook in hooks:hook.remove()
        assert len(shared)==3 and all(torch.equal(shared[0],z) for z in shared[1:])
        old={n:p.detach().clone() for n,p in agent.named_parameters()}
        metrics=agent.update([(record,1,2)])
        for prefix in ('encoder.','actor.','q1.','q2.'):
            assert any(not torch.equal(p,old[n]) for n,p in agent.named_parameters() if n.startswith(prefix)),prefix
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(torch.isfinite(p).all() for p in agent.parameters())
    finally:env.close()
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==before
