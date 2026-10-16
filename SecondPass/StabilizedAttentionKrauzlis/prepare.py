"""Create the isolated stability variant from the preserved original sources."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'LocalComparisonKrauzlis'

def main():
    for name in ('model.py','dataset.py','train.py','launch.py'):
        assert not (HERE/name).exists(), 'Do not overwrite an existing experiment'
        text=(OLD/name).read_text(encoding='utf-8')
        text=text.replace('LocalComparisonKrauzlis','StabilizedAttentionKrauzlis')
        if name=='dataset.py':
            text=text.replace('196101','197101').replace('196102','197102').replace('196103','197103')
        if name=='model.py':
            text=text.replace('from SecondPass.OutcomeContrastiveRViT.model import SpatialGRU, CLSGRU',
                              'from .memory import SpatialGRU, CLSGRU')
            text=text.replace('self.query=nn.Linear',
                'self.attention_scale_logit=nn.Parameter(torch.tensor(math.log(7/8)))\n        self.query=nn.Linear')
            text=text.replace('history=[]; errors=[]; previous_flow=None',
                'history=[]; gate_history=[]; errors=[]; previous_flow=None')
            text=text.replace('logits=(key*query[:,None]).sum(-1)/math.sqrt(self.width)',
                'key=F.normalize(key,dim=-1,eps=1e-6)\n            query=F.normalize(query,dim=-1,eps=1e-6)\n            attention_scale=1+15*self.attention_scale_logit.sigmoid()\n            logits=attention_scale*(key*query[:,None]).sum(-1)')
            text=text.replace('field=self.spatial((100*alpha[...,None]*value).transpose(1,2).reshape(b,self.width,10,10),field)\n            cls=self.cls(selected,cls)',
                'write_attention=(alpha/(alpha+1/alpha.shape[-1])).reshape(b,1,10,10)\n            field,spatial_stats=self.spatial(value.transpose(1,2).reshape(b,self.width,10,10),field,write_attention,return_details)\n            cls,cls_stats=self.cls(selected,cls,return_details)\n            if return_details: gate_history.append(torch.cat((spatial_stats,cls_stats)))')
            text=text.replace('prediction_loss=prediction_loss,cls=cls,temperature=tau,comparison_memory=comparison_memory)',
                'prediction_loss=prediction_loss,cls=cls,temperature=tau,comparison_memory=comparison_memory,\n                gate_statistics=torch.stack(gate_history).mean(0),attention_scale=attention_scale)')
        if name=='train.py':
            text=text.replace("'SecondPass/OutcomeContrastiveRViT/model.py',",'')
            text=text.replace('INIT_SEED,SHUFFLE_SEED=196211,196212','INIT_SEED,SHUFFLE_SEED=197211,197212')
            text=text.replace('totals=dict(loss=0.,decision_loss=0.,prediction_loss=0.,accuracy=0.,cls_std=0.)',
                "totals=dict(loss=0.,decision_loss=0.,prediction_loss=0.,accuracy=0.,cls_std=0.,attention_entropy=0.,attention_peak=0.,attention_scale=0.,**{k:0. for k in GATE_NAMES})")
            text=text.replace("for k,v in values.items(): totals[k]+=weight*float(v.detach())",
                "alpha=detail['attention'].flatten(2).detach()\n        values.update(attention_entropy=-(alpha*alpha.clamp_min(1e-12).log()).sum(-1).mean(),\n            attention_peak=alpha.max(-1).values.mean(),attention_scale=detail['attention_scale'])\n        values.update(zip(GATE_NAMES,detail['gate_statistics']))\n        for k,v in values.items(): totals[k]+=weight*float(v.detach())")
            text=text.replace('opt.step(); totals.update(batch_size=len(indices),gradient_norm=float(norm))',
                "torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True)\n    clipped_norm=torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()\n    opt.step(); totals.update(batch_size=len(indices),gradient_norm=float(norm),clipped_gradient_norm=float(clipped_norm),gradient_clipped=bool(norm>100))")
            start=text.index('    tiny=dataset.render')
            end=text.index('    clock=time.time();rows=dataset.pool(0)',start)
            text=text[:start]+'    del model,opt\n'+text[end:]
            text=text.replace('tiny_fit_last=trace[-1],','')
            text=text.replace('range(100,target+1,100))|{66,target}','range(50,target+1,50))|{33,target}')
            text=text.replace("clipping=None","clipping=100.,gradient_renormalization=False")
            text=text.replace('validation_every=100','validation_every=50,first_validation=33')
            text=text.replace("version='native_krauzlis_local_comparison_shared_attention_v1'","version='native_krauzlis_stabilized_attention_v1'")
            text=text.replace("architecture='same shared attention routes visual values and explicit local temporal comparisons; spatialGRU, CLSGRU and selected comparison trace'",
                "architecture='one shared attention; QK L2 normalization with sigmoid-bounded scale1..16; per-patch separate gate/candidate LayerNorm in spatialGRU and CLSGRU; alpha/(alpha+1/N) gates spatial update; original local comparisons',\n        memory_normalization='gate/candidate preactivations across channels; stored state is not renormalized'")
            text=text.replace("state['step']%100==0","state['step']%50==0")
            text=text.replace('default=6600','default=2244')
            text=text.replace('from .model import StabilizedAttentionKrauzlis',
                'from .model import StabilizedAttentionKrauzlis\nfrom .memory import GATE_NAMES')
        if name=='launch.py':
            text=text.replace('user approved targeted local temporal comparison variant after frozen-model failure diagnostics',
                'user requested normalization, gradient clipping100, and suggested attention stabilization; fresh local run')
        (HERE/name).write_text(text,encoding='utf-8')
    (HERE/'__init__.py').write_text('',encoding='utf-8')

if __name__=='__main__': main()
