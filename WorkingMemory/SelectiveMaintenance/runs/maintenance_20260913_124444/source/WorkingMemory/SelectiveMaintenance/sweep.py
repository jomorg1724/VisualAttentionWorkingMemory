"""Sequential two-arm finite-budget comparison; no allowance renewal."""
import sys,os,time,json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha
PARENT=ROOT/'WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/spatial_ei/checkpoint_004400.pt'
def protocol():
    cells={}
    for family,label in [('orientation_single','single'),('orientation_binding','binding'),('motion_direction','motion')]:
        for d in ((0,24) if label=='motion' else (0,4,12,24)):cells[f'{label}_D{d}']=dict(family=family,condition=dict(delay=d,spacing='mixed'))
    cycle=[name for name in cells for _ in range(4 if name.startswith('motion') else 9)]
    return cells,cycle
def run(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Cannot renew existing budget')
    cells,cycle=protocol();arms=('controller_feedback','continuation')
    cfg=dict(cells=cells,cycle=cycle,batch_size=8,train_seed=32973001,scheduler_seed=33973001,model_seed=41973001,new_lr=.0003,parent_lr=.00003,adam_eps=1e-10,weight_decay=.0001,clip=1.,activation_checkpoint=True)
    names=list(read(PARENT.parents[1]/'budget.json')['source_hashes'])+['WorkingMemory/SelectiveMaintenance/'+n for n in ('model.py','train.py','sweep.py')]
    hashes={n:sha(ROOT/n) for n in names}
    for name in names:
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    digest=sha(PARENT);assert digest=='9a3ffb08829246e1a3622d2658cd76aed46d89c1d84e57fa9a2b334b9a66b90b'
    start=time.time();deadline=start+14400;ledger=dict(started_unix=start,deadline_unix=deadline,status='profiling',active=None,events=[],source_hashes=hashes)
    agg=dict(status='profiling',run_root=str(root),config=dict(recipe=cfg,parent=str(PARENT),parent_sha256=digest,parent_step=4400,arms=arms,allowance_seconds=14400),profiles={},runs={})
    def publish():write(root/'budget.json',ledger);write(root/'aggregate.json',agg);write(HERE/'results.json',agg)
    def launch(kind,arm,out,**kw):
        i=len(ledger['events']);job=root/f'job_{i:03d}.json';result=root/f'job_{i:03d}_result.json';log=root/f'worker_{i:03d}.log'
        write(job,dict(kind=kind,arm=arm,config=cfg,out=str(out),result=str(result),deadline=deadline-60,parent=str(PARENT),parent_sha256=digest,source_hashes=hashes,**kw));tick=time.time()
        with log.open('w') as f:
            p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(HERE/'train.py'),str(job)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);ledger['active']=dict(pid=p.pid,arm=arm,kind=kind,log=str(log));publish()
            try:code=p.wait(timeout=max(.01,deadline-time.time()-60))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=time.time()-tick));ledger['active']=None;publish()
        if code:raise RuntimeError(f'{arm}/{kind} failed exit{code}: {log}')
        return read(result)
    publish()
    try:
        profile_names=['binding_D24','single_D24','motion_D24','binding_D0','single_D0','motion_D0']
        for arm in arms:agg['profiles'][arm]=launch('profile',arm,root/f'profile_{arm}',profile_names=profile_names);publish()
        times={};evalcost={}
        for arm,prof in agg['profiles'].items():
            rows={r['name']:r for r in prof['rows']};costs={}
            for name in cells:
                family,d=name.split('_D');fraction=int(d)/24;costs[name]={k:(1-fraction)*rows[family+'_D0'][k]+fraction*rows[family+'_D24'][k] for k in ('seconds','eval_seconds')}
            times[arm]=sum(costs[n]['seconds'] for n in cycle)/80
            # Four validation looks over ten cells; final fourteen cells including held-out binding positions.
            evalcost[arm]=(sum(c['eval_seconds'] for c in costs.values())*4*128+sum(c['eval_seconds'] for c in costs.values())*512+sum(costs[n]['eval_seconds'] for n in costs if n.startswith('binding'))*512)/8
        reserve=900+1.6*sum(evalcost.values());steps=next((n for n in range(4960,0,-80) if 1.3*n*sum(times.values())+reserve<deadline-time.time()-120),None)
        if steps is None:raise RuntimeError('No balanced equal exposure fits finite budget')
        targets=[4400+round(steps*f) for f in (.25,.5,.75,1.)]
        agg['config'].update(updates_per_arm=steps,episodes_per_arm=steps*8,targets=targets,val_seed=45973001,test_seed=46973001,validation_n_per_cell=128,test_n_per_cell=512,selection='equal mean eight primary task-by-delay validation AUC; ties earlier',training_estimate_seconds=1.3*steps*sum(times.values()),evaluation_reporting_reserve_seconds=reserve,profile_updates_per_arm=6,profile_exposure_not_continued=True,teaching='unchanged SpatialComparison protocol/loss/schedule, exact parent sampler and RNG continuation',acute_intervention='selected feedback arm only; silence additive feedback during D24 inserted blanks on three standard tasks; OOD diagnostic',run_order=list(arms))
        write(root/'fixed_config.json',agg['config']);print('FIXED_ALLOCATION '+json.dumps(agg['config']),flush=True);agg['status']=ledger['status']='training';publish()
        # The one already-requested actual-code review may finish during profiling.
        # This waits only for that review's recorded completion, never renews time.
        while not (HERE/'review_ready.json').exists():
            if time.time()>deadline-120:raise TimeoutError('Review not resolved inside original allowance')
            time.sleep(1)
        for arm in arms:
            cp=None;best=None;record=dict(validation=[]);agg['runs'][arm]=record
            for target in targets:
                result=launch('train',arm,root/arm,checkpoint=cp,target=target);record['training']=result;cp=result['checkpoint'];publish()
                if result['step']!=target:raise RuntimeError('Production target interrupted')
                v=launch('eval',arm,root/f'{arm}_val_{target:06d}',checkpoint=cp,split='val',eval_seed=45973001,n=128,heldout_locations=False);record['validation'].append(v)
                if best is None or v['selection_mean_auc']>best['selection_mean_auc']:best=v;record.update(selected_checkpoint=cp,selected_step=target)
                publish()
            record['test']=launch('eval',arm,root/f'{arm}_test',checkpoint=record['selected_checkpoint'],split='test',eval_seed=46973001,n=512,heldout_locations=True);record['status']='completed';publish()
        agg['parent_reference']=launch('eval','continuation',root/'parent_reference',checkpoint=None,split='test',eval_seed=46973001,n=512,heldout_locations=True)
        agg['feedback_interruption']=launch('eval','controller_feedback',root/'feedback_interruption',checkpoint=agg['runs']['controller_feedback']['selected_checkpoint'],split='test',eval_seed=46973001,n=512,heldout_locations=False,feedback_off=True)
        agg['status']='completed'
    except BaseException as e:agg.update(status='failed',error=repr(e));raise
    finally:
        ledger['status']=agg['status'];agg['wall_seconds']=time.time()-start;publish();write(root/'exit.json',dict(status=agg['status'],error=agg.get('error'),wall_seconds=agg['wall_seconds'],deadline_unix=deadline))
if __name__=='__main__':run(sys.argv[1])
