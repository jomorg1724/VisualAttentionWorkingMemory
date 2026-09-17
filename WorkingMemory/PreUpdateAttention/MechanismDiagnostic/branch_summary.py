"""Additional descriptive statistics from already-saved motion branch logits."""
from common import *
import numpy as np
def main():
    groups={}
    for line in (HERE/'run/predictions.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['family']=='motion_direction':groups.setdefault(r['cell']+'/'+r['arm']+'/'+r['variant'],[]).append(r)
    result={}
    for name,rows in groups.items():
        p=np.array([r['branch_logits'] for r in rows]);centered=p-p.mean(-1,keepdims=True);labels=np.array([r['label'] for r in rows]);bias=np.array(rows[0]['head_bias']);entry={}
        for i,branch in enumerate(('sensory','postupdate_memory','oldmemory_currentfield_comparator')):
            entry[branch]=dict(raw_mean=p[:,i].mean(0).tolist(),raw_between_episode_sd=p[:,i].std(0,ddof=1).tolist(),centered_mean=centered[:,i].mean(0).tolist(),centered_between_episode_sd=centered[:,i].std(0,ddof=1).tolist(),centered_mean_by_true_direction={str(j):centered[labels==j,i].mean(0).tolist() for j in range(4)},up_minus_down_mean=float((p[:,i,1]-p[:,i,3]).mean()),up_minus_down_sd=float((p[:,i,1]-p[:,i,3]).std(ddof=1)))
        result[name]=dict(n=len(rows),branch_statistics=entry,head_bias=bias.tolist(),centered_head_bias=(bias-bias.mean()).tolist(),direction_order=['right','up','left','down'])
    write(HERE/'branch_statistics.json',dict(status='completed',groups=result,interpretation='Descriptive exact head projections on realized features. Centering removes class-independent common shifts. SD is between independent episodes, not standard error. Branch decomposition is not unique under reparameterization and does not identify the training cause.'))
    print(json.dumps({name:{branch:dict(centered_mean=v['centered_mean'],centered_sd=v['centered_between_episode_sd'],up_minus_down_mean=v['up_minus_down_mean']) for branch,v in value['branch_statistics'].items()} for name,value in result.items() if name.endswith('/baseline')},indent=2))
if __name__=='__main__':main()
