"""Bounded-history streaming form of the unchanged stabilized computation."""
import torch
from torch.nn import functional as F
from SecondPass.StabilizedAttentionKrauzlis.model import StabilizedAttentionKrauzlis,warp

def detach_state(state):
    if state is None:return None
    if torch.is_tensor(state):return state.detach()
    if isinstance(state,dict):return {k:detach_state(v) for k,v in state.items()}
    if isinstance(state,list):return [detach_state(v) for v in state]
    return state

class StreamingEncoder(StabilizedAttentionKrauzlis):
    def __init__(self):
        super().__init__();self.readout=self.readout[:-1]

    def load_suite(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        self.load_state_dict({k:v for k,v in saved['model'].items() if not k.startswith('heads.')},strict=True)
        return self

    def step(self,rgb,state=None):
        """One observed RGB frame; return the exact 64D pre-belief representation.

        State resets only at episode boundaries. Temporal windows count observed
        frames, not skipped Doom tics. No health, labels or game metadata enter.
        """
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,100,100):raise ValueError('B,3,100,100 required')
        b=len(rgb);gray=rgb.mean(1,keepdim=True)
        if state is None:
            state=dict(gray=[],pop=[],valid=[],trace=rgb.new_full((b,3,100,81),1/81),
                field=rgb.new_zeros(b,self.width,10,10),cls=rgb.new_zeros(b,self.width),
                comparison=rgb.new_zeros(b,self.comparison_width),flow=None,energy=None)
        gray_history=(state['gray']+[gray])[-5:]
        costs,valid_all=self.motion(torch.stack(gray_history,1));cost=costs[:,-1];valid=valid_all[:,-1]
        corr=cost.reshape(-1,1,9,9);refined=(corr+.1*self.refine(corr)).reshape(b,3,100,81)
        tau=self.log_temperature.exp().clamp(.015,.2);pop=(refined/tau[None,:,None,None]).softmax(-1)
        populations=(state['pop']+[pop])[-16:];validities=(state['valid']+[valid])[-16:]
        def average(start,end,width):
            ps=populations[start:end];vs=validities[start:end]
            if not ps:return torch.zeros_like(pop),torch.zeros_like(valid)
            mass=torch.stack(vs).sum(0);num=torch.stack([p*v[...,None] for p,v in zip(ps,vs)]).sum(0)
            return num/mass[...,None].clamp_min(1e-6),mass/width
        reference,rc=average(-16,-8,8);observation,oc=average(-4,None,4)
        confidence=(rc*oc).clamp_min(0).sqrt();delta=(observation-reference).abs()
        local=torch.cat(((10*reference*confidence[...,None]).transpose(1,2).flatten(2),
            (10*observation*confidence[...,None]).transpose(1,2).flatten(2),
            (10*delta*confidence[...,None]).transpose(1,2).flatten(2),
            (delta.sum(-1)*confidence).transpose(1,2),
            ((1-F.cosine_similarity(reference,observation,dim=-1))*confidence).transpose(1,2)),-1)
        displacement=torch.einsum('blsk,kd->blsd',pop,self.motion.offsets)
        velocity=displacement/rgb.new_tensor([1.,2.,4.])[None,:,None,None]
        app=self.appearance(rgb-.5).flatten(2).transpose(1,2)
        means=F.avg_pool2d(rgb-.5,10).flatten(2).transpose(1,2)
        energy=F.avg_pool2d((gray-.5).square(),10).flatten(1)
        raw=torch.cat((means,energy.sqrt()[...,None]),-1)
        if state['flow'] is None:error=energy*0
        else:
            prediction=warp(state['gray'][-1],state['flow'])
            mse=F.avg_pool2d((prediction-gray).square(),10).flatten(1)
            error=mse/(.5*(state['energy']+energy)).clamp_min(1e-4)
        diff=(pop-state['trace']).abs().sum(-1).transpose(1,2)
        motion_features=(10*pop*valid[...,None]).transpose(1,2).flatten(2)
        features=torch.cat((app,raw,motion_features,valid.transpose(1,2),diff,velocity.transpose(1,2).flatten(2),error[...,None]),-1)
        visual=self.fusion(features)+self.position;old=state['field'].flatten(2).transpose(1,2)
        key=F.normalize(self.key(self.key_norm(visual)+self.memory_norm(old)),dim=-1,eps=1e-6)
        query=F.normalize(self.query(self.query_norm(state['cls']+self.cls_token)),dim=-1,eps=1e-6)
        alpha=((1+15*self.attention_scale_logit.sigmoid())*(key*query[:,None]).sum(-1)).softmax(-1)
        value=self.value(visual)+self.memory_value(old);value=value+self.ffn(value)
        selected=(alpha[...,None]*value).sum(1)
        comparison=.8*state['comparison']+.2*(alpha[...,None]*local).sum(1)
        gate=(alpha/(alpha+1/100)).reshape(b,1,10,10)
        field,_=self.spatial(value.transpose(1,2).reshape(b,self.width,10,10),state['field'],gate)
        cls,_=self.cls(selected,state['cls'])
        trace=state['trace']+self.trace_rate.sigmoid()*valid[...,None]*(pop-state['trace'])
        flow=velocity.mean(1).transpose(1,2).reshape(b,2,10,10)
        new=dict(gray=gray_history,pop=populations,valid=validities,trace=trace,field=field,cls=cls,comparison=comparison,flow=flow,energy=energy)
        h=self.readout(torch.cat((cls,comparison),-1))
        return h,new,alpha.reshape(b,10,10)
