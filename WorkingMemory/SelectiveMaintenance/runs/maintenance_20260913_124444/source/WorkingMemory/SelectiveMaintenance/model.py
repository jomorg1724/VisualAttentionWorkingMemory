"""Architecture-only additive feedback; all teaching and output routes unchanged."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from WorkingMemory.SpatialComparison.model import Competitor,groups as parent_groups,logit
ARMS=('continuation','controller_feedback')
VERSION='spatial_additive_controller_v1'

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(2*64*3*3,32,bias=False),nn.LayerNorm(32))
        self.raw_recurrent=nn.Parameter(torch.empty(32,32));self.bias=nn.Parameter(torch.full((32,),.05))
        self.raw_tau_r=nn.Parameter(logit((torch.logspace(math.log10(2),math.log10(8),32)-1)/31))
        self.raw_tau_a=nn.Parameter(logit((torch.logspace(math.log10(16),math.log10(64),32)-4)/124))
        self.raw_adaptation=nn.Parameter(torch.full((32,),math.log(.1/.9)))
        self.register_buffer('signs',torch.cat((torch.ones(26),-torch.ones(6))))
        self.raw_feedback=nn.Parameter(torch.empty(64,32))
        self.raw_basis=nn.Parameter(torch.empty(32,13,13))
        self.raw_coupling=nn.Parameter(torch.tensor(math.log(math.expm1(.01))))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input[0].weight,gain=.4)
            for raw in (self.raw_recurrent,self.raw_feedback):
                v=torch.empty_like(raw).uniform_(.5,1.5)
                for a,b in ((0,26),(26,32)):v[:,a:b]*=.6/v[:,a:b].sum(1,keepdim=True)
                raw.copy_(torch.log(torch.expm1(v)))
            self.raw_basis.normal_(math.log(math.expm1(1)),.02)
    def forward(self,h,r,state=None):
        if state is None:state=(h.new_zeros(len(h),32),h.new_zeros(len(h),32))
        old,adapt=state
        # Entire fields contribute; no cue crop, task ID, phase or target input.
        z=self.input(torch.cat((F.adaptive_avg_pool2d(h,3).flatten(1),F.adaptive_avg_pool2d(r,3).flatten(1)),1))
        alpha=-torch.expm1(-1/(1+31*self.raw_tau_r.sigmoid()));beta=-torch.expm1(-1/(4+124*self.raw_tau_a.sigmoid()))
        drive=z+F.linear(old,F.softplus(self.raw_recurrent)*self.signs[None,:])-.5*self.raw_adaptation.sigmoid()*adapt+self.bias
        return ((1-alpha)*old+alpha*drive.relu(),(1-beta)*adapt+beta*old)
    def feedback(self,old):
        # Fixed source identity across every output channel and location.
        basis=F.softplus(self.raw_basis);basis=basis/basis.mean((1,2),keepdim=True)
        return F.softplus(self.raw_coupling)*torch.einsum('bc,kc,chw->bkhw',old*self.signs,F.softplus(self.raw_feedback),basis)

class Maintenance(Competitor):
    def __init__(self,arm,seed=41973001,activation_checkpoint=True):
        super().__init__('spatial_ei',activation_checkpoint=activation_checkpoint)
        if arm not in ARMS:raise ValueError(arm)
        self.maintenance_arm=arm
        if arm=='controller_feedback':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed);self.controller=Controller()
    def forward(self,images,task,diagnostic=False,feedback_off_frames=None):
        if self.maintenance_arm=='continuation':return super().forward(images,task,diagnostic)
        traces=();state=None;control=None;states=[];controls=[];records=[];first_field=None
        for t in range(images.shape[1]):
            if self.training and self.activation_checkpoint and torch.is_grad_enabled():res=checkpoint(self._sensory,images[:,t],*traces,use_reentrant=False,preserve_rng_state=True)
            else:res=self._sensory(images[:,t],*traces)
            field,traces=res[0],res[1:];z=self.memory_input(field)
            old=torch.zeros_like(z) if state is None else state[0];adapt=torch.zeros_like(z) if state is None else state[1]
            old_control=field.new_zeros(len(field),32) if control is None else control[0]
            feedback=self.controller.feedback(old_control)
            # This externally supplied intervention is evaluation-only. Normal training
            # and inference never receive a frame mask or task-dependent controller flag.
            if feedback_off_frames is not None:
                if self.training:raise ValueError('Acute interruption is inference only')
                if t in feedback_off_frames:feedback=feedback*0
            next_control=self.controller(field,old,control)
            local=self.comparator(torch.cat((old,field),1));comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1)
            m=self.memory;tau=1+31*m.raw_tau_r.sigmoid();ta=4+124*m.raw_tau_a.sigmoid()
            alpha=(-torch.expm1(-1/tau))[None,:,None,None];beta=(-torch.expm1(-1/ta))[None,:,None,None]
            drive=m.input(z)+F.conv2d(old,m.recurrent_weight(),padding=1)-(.5*m.raw_adaptation.sigmoid())[None,:,None,None]*adapt+m.bias[None,:,None,None]
            r=(1-alpha)*old+alpha*(drive+feedback).relu();a=(1-beta)*adapt+beta*old;state=(r,a);control=next_control
            if diagnostic:
                if t==0:field.retain_grad();first_field=field
                for tensor in (*state,*control):
                    if tensor.requires_grad:tensor.retain_grad()
                states.append(state);controls.append(control)
                records.append(dict(inactive_fraction=float((r<1e-6).float().mean()),feedback_rms=float(feedback.detach().square().mean().sqrt()),drive_rms=float(drive.detach().square().mean().sqrt()),coupling=float(F.softplus(self.controller.raw_coupling).detach())))
        sensory=self.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
        retained=torch.cat((r.mean((2,3)),r.amax((2,3))),1)
        # Exactly the inherited output route: no controller-to-classifier connection.
        logits=self.classify(sensory+self.memory_output(retained)+self.comparison_output(comp),task)
        return (logits,dict(states=states,first_field=first_field,records=records,controls=controls)) if diagnostic else logits

def groups(model,cfg):
    result={}
    for name,p in model.named_parameters():
        new=name.startswith(('memory.','memory_input.','memory_output.','comparator.','comparison_output.','readout.heads.orientation_binding.','controller.'))
        wd=cfg['weight_decay'] if p.ndim>=2 and name.endswith('weight') and 'norm' not in name and not name.startswith('controller.input.1.') else 0.
        result.setdefault((cfg['new_lr'] if new else cfg['parent_lr'],wd),[]).append(p)
    return [dict(params=v,lr=k[0],weight_decay=k[1]) for k,v in result.items()]

def migrate(parent,arm,cfg):
    assert parent['version']=='spatial_dense_ei_comparison_v1' and parent['arm']=='spatial_ei' and parent['step']==4400
    old=Competitor('spatial_ei',activation_checkpoint=False);old.load_state_dict(parent['model'],strict=True)
    oldopt=torch.optim.Adam(parent_groups(old,parent['config']),eps=parent['config']['adam_eps']);oldopt.load_state_dict(parent['optimizer'])
    model=Maintenance(arm,cfg['model_seed']);state=model.state_dict()
    for name,value in parent['model'].items():
        assert name in state and state[name].shape==value.shape,name
        state[name]=value
    missing=set(state)-set(parent['model']);assert all(n.startswith('controller.') for n in missing)
    model.load_state_dict(state,strict=True);opt=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);oldnames=dict(old.named_parameters());copied=[]
    for name,p in model.named_parameters():
        if name in oldnames and oldnames[name] in oldopt.state:opt.state[p]=copy.deepcopy(oldopt.state[oldnames[name]]);copied.append(name)
    return model,opt,dict(version=VERSION,parent_step=4400,parent_version=parent['version'],retained_tensors=list(parent['model']),adam_states_by_name=copied,new_parameters=sorted(missing),sampler='exact parent stream and RNG continuation; global schedule offset4400',extra_state_entries=64 if arm=='controller_feedback' else 0)
