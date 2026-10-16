"""Discrete SAC with twin quantile action-value distributions.

The action mixture is retained in the distributional Bellman target, rather than
collapsing all possible next actions into a mean before quantile regression.
"""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .encoder import StreamingEncoder,detach_state
from .environment import ACTIONS

def mlp(outputs):return nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,outputs))

def quantile_loss(prediction,targets,weights,kappa=1.):
    """prediction[B,N], targets[B,M], normalized target-mixture weights[B,M]."""
    delta=targets.detach()[:,None,:]-prediction[:,:,None]
    huber=torch.where(delta.abs()<=kappa,.5*delta.square(),kappa*(delta.abs()-.5*kappa))
    tau=(torch.arange(prediction.shape[-1],device=prediction.device,dtype=prediction.dtype)+.5)/prediction.shape[-1]
    loss=(tau[None,:,None]-(delta.detach()<0).to(delta.dtype)).abs()*huber/kappa
    return (loss*weights.detach()[:,None,:]).sum(-1).mean()

def soft_targets(reward,discount,done,probabilities,log_probabilities,q1,q2,alpha):
    # Select the WHOLE lower-mean critic distribution separately for each action.
    # Pointwise min of quantiles would introduce a different artificial law.
    choose=q1.mean(-1)<=q2.mean(-1)
    z=torch.where(choose[...,None],q1,q2)
    samples=reward[:,None,None]+discount[:,None,None]*(1-done[:,None,None])*(z-alpha*log_probabilities[...,None])
    weights=probabilities[...,None].expand_as(z)/z.shape[-1]
    return samples.flatten(1),weights.flatten(1)

class Agent(nn.Module):
    def __init__(self,checkpoint=None,quantiles=32,lr=3e-4):
        super().__init__();self.encoder=StreamingEncoder();self.quantiles=quantiles;self.actions=len(ACTIONS)
        if checkpoint:self.encoder.load_suite(checkpoint)
        self.actor=mlp(self.actions);self.q1=mlp(self.actions*quantiles);self.q2=mlp(self.actions*quantiles)
        self.target_encoder=copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_q1=copy.deepcopy(self.q1).requires_grad_(False);self.target_q2=copy.deepcopy(self.q2).requires_grad_(False)
        self.log_alpha=nn.Parameter(torch.tensor(math.log(.1)))
        self.critic_opt=torch.optim.Adam(list(self.encoder.parameters())+list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=lr);self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=lr)
        self.target_entropy=.5*math.log(self.actions)

    def q(self,net,h):return net(h).reshape(-1,self.actions,self.quantiles)

    @torch.no_grad()
    def act(self,image,state=None,deterministic=False):
        h,state,attention=self.encoder.step(image,state)
        distribution=torch.distributions.Categorical(logits=self.actor(h))
        action=distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()),detach_state(state),attention

    def encode_segment(self,encoder,episode,start,length,gradient=False):
        device=next(self.parameters()).device;state=None;features=[]
        # Replay observations from the ACTUAL episode start with current weights.
        # No stale collection-state embeddings and no arbitrary memory resets.
        with torch.no_grad():
            for i in range(start):
                x=torch.as_tensor(episode['observations'][i],device=device,dtype=torch.float32)[None]/255
                _,state,_=encoder.step(x,state)
        state=detach_state(state)
        with torch.set_grad_enabled(gradient):
            for i in range(start,start+length+1):
                x=torch.as_tensor(episode['observations'][i],device=device,dtype=torch.float32)[None]/255
                h,state,_=encoder.step(x,state);features.append(h)
        return torch.cat(features)

    def update(self,segments,reward_scale=1.,gamma=.99,polyak=.005,clip=100.):
        device=next(self.parameters()).device
        hs=[];next_online=[];next_target=[];actions=[];rewards=[];discounts=[];dones=[]
        for episode,start,length in segments:
            h=self.encode_segment(self.encoder,episode,start,length,True);hs.append(h[:-1]);next_online.append(h[1:].detach())
            next_target.append(self.encode_segment(self.target_encoder,episode,start,length,False)[1:])
            sl=slice(start,start+length);actions.extend(episode['actions'][sl]);rewards.extend(episode['rewards'][sl]);dones.extend(episode['done'][sl])
            discounts.extend([gamma**(n/4) for n in episode['tics'][sl]])
        h=torch.cat(hs);nh=torch.cat(next_online);th=torch.cat(next_target)
        act=torch.tensor(actions,device=device);r=torch.tensor(rewards,device=device)*reward_scale
        dis=torch.tensor(discounts,device=device);done=torch.tensor(dones,device=device,dtype=torch.float32)
        with torch.no_grad():
            logp=self.actor(nh).log_softmax(-1);prob=logp.exp();alpha=self.log_alpha.exp()
            targets,weights=soft_targets(r,dis,done,prob,logp,self.q(self.target_q1,th),self.q(self.target_q2,th),alpha)
        batch=torch.arange(len(h),device=device)
        q1=self.q(self.q1,h)[batch,act];q2=self.q(self.q2,h)[batch,act]
        critic_loss=quantile_loss(q1,targets,weights)+quantile_loss(q2,targets,weights)
        self.critic_opt.zero_grad(set_to_none=True);critic_loss.backward()
        critic_params=list(self.encoder.parameters())+list(self.q1.parameters())+list(self.q2.parameters())
        norm=torch.nn.utils.clip_grad_norm_(critic_params,clip,error_if_nonfinite=True);self.critic_opt.step()
        # Actor sees the same 64D h. Its gradients update its own head; the shared
        # visual/recurrent encoder learns through both critics, avoiding actor/Q
        # gradients pulling the shared representation in opposing directions.
        detached=h.detach();logp=self.actor(detached).log_softmax(-1);prob=logp.exp()
        with torch.no_grad():q=torch.minimum(self.q(self.q1,detached).mean(-1),self.q(self.q2,detached).mean(-1))
        actor_loss=(prob*(self.log_alpha.exp().detach()*logp-q)).sum(-1).mean()
        self.actor_opt.zero_grad(set_to_none=True);actor_loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),clip,error_if_nonfinite=True);self.actor_opt.step()
        entropy=-(prob.detach()*logp.detach()).sum(-1).mean()
        # If entropy is below the target, gradient descent increases alpha.
        alpha_loss=self.log_alpha*(entropy-self.target_entropy)
        self.alpha_opt.zero_grad(set_to_none=True);alpha_loss.backward();self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10,2)
            for online,target in ((self.encoder,self.target_encoder),(self.q1,self.target_q1),(self.q2,self.target_q2)):
                for p,tp in zip(online.parameters(),target.parameters()):tp.lerp_(p,polyak)
        return dict(critic_loss=float(critic_loss.detach()),actor_loss=float(actor_loss.detach()),alpha=float(self.log_alpha.exp().detach()),entropy=float(entropy),gradient_norm=float(norm),transitions=len(h))
