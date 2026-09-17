"""One existing EI retention-learning continuation versus untouched parent; finite local cap."""
import os,sys,time,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha
from WorkingMemory.RecurrentComparison.Retention.protocol import protocol
PARENT=ROOT/'WorkingMemory/RecurrentComparison/ReadoutRefit/runs/readout_20260912_220701/readout_refit/checkpoint_009840.pt'
def run(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Cannot renew existing allowance')
    cells,cycle=protocol();prior=read(ROOT/'WorkingMemory/RecurrentComparison/ReadoutRefit/runs/readout_20260912_220701/fixed_config.json')
    cfg=dict(prior['recipe'],arm='ei_adaptive',batch_size=8,activation_checkpoint=False,purpose='existing_core_retention_learning',cells=cells,cycle=cycle,scheduler_seed=22973001,optimizer='preserved_Adam_states_same_parameter_inventory',training_protocol='paired_blank_retention_v1')
    names=list(read(ROOT/'WorkingMemory/RecurrentComparison/ReadoutRefit/runs/readout_20260912_220701/budget.json')['source_hashes'])
    names+=['WorkingMemory/RecurrentComparison/Retention/'+n for n in ('train.py','sweep.py','protocol.py')]
    hashes={n:sha(ROOT/n) for n in names}
    for name in names:
        dst=root/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    digest=sha(PARENT)
    if digest!='6b3542ee4c679ccf1990556377937d0d5f36a5ba85e89b3d9ad8d68bb2c09ce9':raise RuntimeError('Parent identity mismatch')
    now=time.time();deadline=now+14400
    ledger=dict(started_unix=now,deadline_unix=deadline,hard_seconds=14400,status='profiling',active=None,events=[],source_hashes=hashes)
    aggregate=dict(status='profiling',run_root=str(root),config=dict(parent_checkpoint=str(PARENT),parent_sha256=digest,parent_step=9840,allowance_seconds=14400,recipe=cfg,cells=cells),profiles=[],training={},baseline=None,validation=[],parent_test=None,test=None)
    def publish():write(root/'aggregate.json',aggregate);write(HERE/'results.json',aggregate);write(root/'budget.json',ledger)
    def launch(kind,out,**kw):
        number=len(ledger['events']);result=root/f'job_{number:03d}_result.json';job=root/f'job_{number:03d}.json';log=root/f'worker_{number:03d}.log'
        write(job,dict(kind=kind,config=cfg,out=str(out),deadline=deadline-60,result=str(result),source_hashes=hashes,parent_checkpoint=str(PARENT),parent_sha256=digest,**kw));tick=time.time()
        with log.open('w') as f:
            p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(HERE/'train.py'),str(job)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            ledger['active']=dict(pid=p.pid,kind=kind,started_unix=tick,log=str(log));publish()
            try:code=p.wait(timeout=max(.01,deadline-time.time()-60))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=time.time()-tick));ledger['active']=None;publish()
        if code:raise RuntimeError(f'{kind} exit{code}: {log}')
        return read(result)
    publish()
    try:
        profile_cells=[(cells[n]['family'],cells[n]['condition']) for n in ['motion_D24','orientation_D24','motion_D12','orientation_D12','motion_D4','orientation_D4','motion_D0','orientation_D0','motion_direction_anchor','orientation_anchor']]
        profile=launch('profile',root/'separate_profile',profile_cells=profile_cells);aggregate['profiles'].append(profile)
        lookup={(m['family'],json.dumps(m['condition'],sort_keys=True)):m for m in profile['measurements']}
        prices={n:lookup[(c['family'],json.dumps(c['condition'],sort_keys=True))] for n,c in cells.items()}
        average=sum(prices[n]['step_seconds'] for n in cycle)/len(cycle)
        reserve=600+1.4*sum(m['eval_seconds'] for m in prices.values())/8*(5*128+2*512)
        steps=next((n for n in range(4960,0,-80) if 1.3*n*average+reserve<deadline-time.time()-90),None)
        if steps is None:raise RuntimeError('No complete-cycle exposure fits allowance')
        targets=[9840+round(steps*f) for f in (.25,.5,.75,1.)]
        aggregate['config'].update(additional_updates=steps,additional_episodes=steps*8,targets=targets,validation_n_per_cell=128,test_n_per_cell=512,validation_seed=23973001,test_seed=24973001,
            estimated_training_seconds=1.3*steps*average,reserved_evaluation_reporting_seconds=reserve,trainable_parameters=profile['trainable_parameters'],profile_training_episodes=80,profile_eval_episodes=80,
            inference_reference='untouched EI9840 paired across models and delays on identical fresh evidence/probe movies',selection='mean eight primary task-by-delay validation OVR-AUC, exact ties earlier; anchors descriptive')
        write(root/'fixed_config.json',aggregate['config']);print('FIXED_ALLOCATION '+json.dumps(aggregate['config']),flush=True)
        aggregate['status']=ledger['status']='baseline';publish()
        aggregate['baseline']=launch('eval',root/'baseline_validation',split='val',eval_seed=23973001,n_per_cell=128,resamples=0);publish()
        aggregate['status']=ledger['status']='training';publish();cp=None;best=None
        for target in targets:
            result=launch('train',root/'retention',target=target,checkpoint=cp);aggregate['training']=result;cp=result['checkpoint'];publish()
            if result['step']!=target:raise RuntimeError('Fixed target interrupted by deadline')
            val=launch('eval',root/f'validation_{target:06d}',checkpoint=cp,split='val',eval_seed=23973001,n_per_cell=128,resamples=0);aggregate['validation'].append(val)
            if best is None or val['selection_mean_auc']>best['selection_mean_auc']:best=val;aggregate.update(selected_checkpoint=cp,selected_step=target)
            publish()
        aggregate['status']=ledger['status']='final_test';publish()
        aggregate['parent_test']=launch('eval',root/'parent_test',split='test',eval_seed=24973001,n_per_cell=512,resamples=0);publish()
        aggregate['test']=launch('eval',root/'refit_test',checkpoint=aggregate['selected_checkpoint'],split='test',eval_seed=24973001,n_per_cell=512,resamples=0);publish()
        aggregate['status']='completed'
    except BaseException as error:aggregate.update(status='failed',error=repr(error));raise
    finally:
        aggregate['wall_seconds']=time.time()-now;ledger['status']=aggregate['status'];publish()
        write(root/'exit.json',dict(status=aggregate['status'],wall_seconds=time.time()-now,deadline_unix=deadline,error=aggregate.get('error')))
if __name__=='__main__':run(sys.argv[1])
