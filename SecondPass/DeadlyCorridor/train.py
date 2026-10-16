"""Explicit-launch, finite local runner. Preparation does not invoke this module."""
import argparse,hashlib,json,random,time,copy
from pathlib import Path
import numpy as np
import torch
from .agent import Agent
from .environment import Corridor,ACTIONS
from .replay import Replay
from SecondPass.StabilizedAttentionVDA.train import PARENT
from SecondPass.PixelContrastiveVAE.train import GPULock,atomic_json,append_json,cpu_tree,replace_with_retry,configure

@torch.no_grad()
def episode(agent,env,seed,device,deadline,warmup=False):
    obs=env.reset(seed);state=None;total=0.;record=dict(observations=[obs],actions=[],rewards=[],done=[],tics=[])
    while True:
        if time.time()>=deadline:raise TimeoutError('Finite local allocation reached')
        # Even during random exploration, advance the same visual/recurrent state.
        action,state,_=agent.act(torch.as_tensor(obs,device=device).float()[None]/255,state)
        if warmup:action=random.randrange(len(ACTIONS))
        obs,reward,done,info=env.step(action);total+=reward
        record['observations'].append(np.zeros((3,100,100),np.uint8) if obs is None else obs)
        for k,v in dict(actions=action,rewards=reward,done=done,tics=info['tics']).items():record[k].append(v)
        if done:return record,dict(raw_return=total,steps=len(record['actions']),tics=sum(record['tics']),dead=info['dead'])

@torch.no_grad()
def evaluate(agent,env,device,deadline,seed_base,count):
    rows=[]
    for seed in range(seed_base,seed_base+count):
        obs=env.reset(seed);state=None;total=0.;steps=0
        while True:
            if time.time()>=deadline:raise TimeoutError('Allocation reached in evaluation')
            action,state,_=agent.act(torch.as_tensor(obs,device=device).float()[None]/255,state,True)
            obs,r,done,info=env.step(action);total+=r;steps+=1
            if done:break
        rows.append(dict(seed=seed,raw_return=total,steps=steps,dead=info['dead'],tics=info['episode_tics']))
    return dict(episodes=rows,mean_return=float(np.mean([r['raw_return'] for r in rows])),count=count)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',type=int,default=1000000)
    p.add_argument('--wall-seconds',type=int,default=28800);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.steps<1 or not 0<a.wall_seconds<=28800:raise ValueError('Positive target and finite <=8h allocation required')
    if not a.resume:a.run.mkdir(parents=True,exist_ok=False)
    elif not (a.run/'latest.pt').exists():raise FileNotFoundError('No existing episode-boundary checkpoint')
    configure(a.device);random.seed(202211);np.random.seed(202211);torch.manual_seed(202211)
    lock=GPULock(PARENT.parents[4]/'LocalRuntime/local_gpu_worker.lock');env=None
    try:
        agent=Agent(PARENT).to(a.device);replay=Replay();steps=updates=episodes=0;next_eval=5000;best=-float('inf')
        env=Corridor();started=time.time();deadline=started+a.wall_seconds
        config=dict(version='corridor_quantile_sac_v1',steps=a.steps,started=started,deadline=deadline,parent=str(PARENT),
            parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),environment=env.provenance,quantiles=32,
            gamma_per_four_tics=.99,reward_scale=.01,lr=3e-4,polyak=.005,gradient_clip=100.,warmup=2000,
            replay_capacity=20000,batch_segments=2,unroll=8,context='exact episode prefix recomputed with current weights under no_grad',
            representation='same64D post-GELU representation used by suite belief heads',encoder_learning='critic gradients only; actor head has detached shared representation',
            seeds=dict(train=202101,val=212101,test=222101),no_auxiliary_pixel_loss=True,
            source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
        if a.resume:
            saved=torch.load(a.run/'latest.pt',map_location=a.device,weights_only=False)
            if saved['config']['environment']!=config['environment']:raise ValueError('Environment version or scenario mismatch')
            if saved['config']['parent_sha256']!=config['parent_sha256']:raise ValueError('Parent checkpoint mismatch')
            config=saved['config'];deadline=config['deadline']
            if time.time()>=deadline:raise TimeoutError('Saved finite allocation expired; no automatic extension')
            if config['source_sha256']!={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}:raise ValueError('Source mismatch')
            agent.load_state_dict(saved['agent'])
            for name in ('critic_opt','actor_opt','alpha_opt'):getattr(agent,name).load_state_dict(saved[name])
            replay=saved['replay'];steps=saved['steps'];updates=saved['updates'];episodes=saved['episodes'];next_eval=saved['next_eval'];best=saved['best']
            torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
            if a.device=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        else:atomic_json(a.run/'config.json',config)
        def save():
            # Called at episode boundaries only; no claim to serialize Doom state.
            payload=cpu_tree(dict(agent=agent.state_dict(),critic_opt=agent.critic_opt.state_dict(),actor_opt=agent.actor_opt.state_dict(),
                alpha_opt=agent.alpha_opt.state_dict(),steps=steps,updates=updates,episodes=episodes,next_eval=next_eval,best=best,config=config,
                torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
            payload['replay']=replay
            tmp=a.run/'latest.pt.tmp';torch.save(payload,tmp);replace_with_retry(tmp,a.run/'latest.pt')
        reason='planned_complete';discarded_partial_episode=False;phase='idle'
        try:
            while steps<config['steps']:
                if time.time()>deadline-120 or (a.run/'stop_requested.json').exists():reason='time_cap' if time.time()>deadline-120 else 'user_stop';break
                phase='collecting'
                record,metrics=episode(agent,env,config['seeds']['train']+episodes,a.device,deadline-90,steps<config['warmup'])
                phase='updating'
                replay.add(record);steps+=metrics['steps'];episodes+=1
                episode_metrics={('episode_steps' if key=='steps' else key):value for key,value in metrics.items()}
                append_json(a.run/'episodes.jsonl',dict(episode=episodes,steps=steps,**episode_metrics))
                # The encoder is fixed during each collected episode. Updates occur
                # between episodes, so online memory never spans a weight update.
                if steps>=config['warmup']:
                    for _ in range(min(32,max(1,metrics['steps']//4))):
                        if time.time()>deadline-90:reason='time_cap';break
                        losses=agent.update(replay.sample(config['batch_segments'],config['unroll']));updates+=1
                        append_json(a.run/'updates.jsonl',dict(update=updates,steps=steps,**losses))
                atomic_json(a.run/'status.json',dict(phase='training',steps=steps,episodes=episodes,updates=updates,deadline=deadline))
                if steps>=next_eval and time.time()<deadline-300:
                    phase='validating'
                    result=evaluate(agent,env,a.device,deadline-90,config['seeds']['val'],10);atomic_json(a.run/f'validation_{steps:08d}.json',result)
                    next_eval=steps+5000
                    if result['mean_return']>best:
                        best=result['mean_return'];tmp=a.run/'best.pt.tmp';torch.save(cpu_tree(dict(agent=agent.state_dict(),steps=steps,validation=result,config=config)),tmp);replace_with_retry(tmp,a.run/'best.pt')
                save()
                if reason=='time_cap':break
            if reason=='planned_complete':
                phase='final_evaluation'
                result=evaluate(agent,env,a.device,deadline-30,config['seeds']['val'],10);atomic_json(a.run/'validation_final.json',result)
                if result['mean_return']>best:
                    best=result['mean_return'];torch.save(cpu_tree(dict(agent=agent.state_dict(),steps=steps,validation=result,config=config)),a.run/'best.pt')
                save();selected=Agent().to(a.device);selected.load_state_dict(torch.load(a.run/'best.pt',map_location=a.device,weights_only=False)['agent'])
                result=evaluate(selected,env,a.device,deadline,config['seeds']['test'],30);atomic_json(a.run/'final_test.json',result)
        except TimeoutError:
            reason='time_cap';discarded_partial_episode=phase=='collecting';save()
        atomic_json(a.run/'exit.json',dict(reason=reason,steps=steps,updates=updates,episodes=episodes,discarded_partial_episode=discarded_partial_episode,
            parent_unchanged=hashlib.sha256(PARENT.read_bytes()).hexdigest()==config['parent_sha256'],complete=reason=='planned_complete'))
    finally:
        if env is not None:env.close()
        lock.close()

if __name__=='__main__':main()
