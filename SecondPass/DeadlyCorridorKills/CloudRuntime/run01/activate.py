"""Disposable checks precede one explicit fresh production launch."""
import json,subprocess,time,random
from pathlib import Path
import numpy as np
import torch
from SecondPass.DeadlyCorridorKills.agent import Agent
from SecondPass.DeadlyCorridorKills.environment import Corridor,ACTIONS

RUN=Path('/workspace/vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01')
PY='/opt/doom-venv/bin/python'

def main():
    assert json.loads((RUN/'upload_verified.json').read_text())['all_hashes_match']
    guard=json.loads((RUN/'guard.json').read_text());assert guard['status']=='armed'
    assert 'cloud_control.py' in Path(f"/proc/{guard['pid']}/cmdline").read_bytes().decode()
    subprocess.run([PY,'-m','pytest','SecondPass/DeadlyCorridorKills/test_contract.py','-q'],cwd='/workspace/vawm',check=True)
    torch.set_num_threads(2);torch.manual_seed(99121)
    agent=Agent().cuda();env=Corridor();record=dict(observations=[],actions=[],rewards=[],done=[],tics=[])
    try:
        obs=env.reset(231112);record['observations'].append(obs)
        action=int(np.flatnonzero(np.all(ACTIONS==[0,0,1,1,0,0,0],axis=1))[0])
        while True:
            obs,r,done,info=env.step(action)
            record['observations'].append(np.zeros((3,100,100),np.uint8) if obs is None else obs)
            for k,v in dict(actions=action,rewards=r,done=done,tics=info['tics']).items():record[k].append(v)
            if done:break
        assert sum(record['rewards'])==info['kills'] and info['kills']>=1
        n=min(8,len(record['actions']));metrics=agent.update([(record,0,n),(record,len(record['actions'])-n,n)])
        assert all(np.isfinite(float(v)) for v in metrics.values())
        assert all(torch.isfinite(p).all() for p in agent.parameters())
        receipt=dict(tests_passed=8,disposable_cuda_update=metrics,disposable_weights_discarded=True,
            actual_kills=info['kills'],reward_sum=sum(record['rewards']),torch=torch.__version__,time=time.time())
        (RUN/'preparation_verified.json').write_text(json.dumps(receipt,indent=2))
    finally:env.close()
    del agent;torch.cuda.empty_cache()
    assert not (RUN/'worker.json').exists()
    with (RUN/'supervisor.log').open('w') as f:
        child=subprocess.Popen([PY,'-u',str(RUN/'cloud_control.py'),'supervise'],cwd='/workspace/vawm',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(3);assert child.poll() is None,child.returncode
    print('Supervisor',child.pid,flush=True);print((RUN/'worker.json').read_text(),flush=True)

if __name__=='__main__':main()
