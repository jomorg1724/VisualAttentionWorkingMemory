"""Resume the pinned allocation after a status-file sharing violation; no new budget."""
import os,sys,time,json,subprocess,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+f'.recovery-{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(100):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)
def main():
    run=Path(sys.argv[1]).resolve();agg=read(run/'aggregate.json');ledger=read(run/'budget.json');cfg=agg['config']['recipe'];deadline=ledger['deadline_unix']
    assert agg['status']=='failed' and 'PermissionError' in agg.get('error','') and time.time()<deadline
    backup=run/'status_publish_failure';backup.mkdir(exist_ok=False)
    for name in ('budget.json','aggregate.json','exit.json','supervisor_error.log'):shutil.copy2(run/name,backup/name)
    # The already-launched validation completed after its supervisor exited.
    active=ledger['active'];assert active['pid']==18832 and active['kind']=='eval'
    job=read(run/'job_007.json');result=read(job['result']);assert result['status']=='completed' and result['step']==7400
    probe=subprocess.run(['powershell','-NoProfile','-Command',f"Get-CimInstance Win32_Process -Filter 'ProcessId={active['pid']}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],capture_output=True,text=True,check=True)
    assert not probe.stdout.strip(),'Previous worker still exists; do not launch a duplicate'
    ledger['events'].append(dict(active,returncode=0,exit_code_observed=False,completion_evidence='Completed job_007_result.json plus absent process; supervisor lost exit handle',wall_seconds=result['worker_seconds']))
    ledger['active']=None;record=agg['runs']['controller_feedback'];assert len(record['validation'])==2
    record['validation'].append(result)
    if result['selection_mean_auc']>max(v['selection_mean_auc'] for v in record['validation'][:-1]):record.update(selected_step=7400,selected_checkpoint=result['checkpoint'])
    recovery=dict(reason=agg.pop('error'),resume_epoch=time.time(),original_deadline=deadline,original_start=ledger['started_unix'],source_unchanged=True,checkpoint_resumed=record['training']['checkpoint'],training_steps_repeated=0,profiles_repeated=0,supervisor_pid=os.getpid(),adopted_completed_validation='job_007_result.json')
    write(run/'recovery_receipt.json',recovery);agg['recovery']=recovery;agg['status']=ledger['status']='training'
    def publish():write(run/'budget.json',ledger);write(run/'aggregate.json',agg);write(P/'results.json',agg)
    def launch(kind,arm,out,**kw):
        i=len(ledger['events']);path=run/f'job_{i:03d}.json';result_path=run/f'job_{i:03d}_result.json';log=run/f'worker_{i:03d}.log'
        write(path,dict(kind=kind,arm=arm,config=cfg,out=str(out),result=str(result_path),deadline=deadline-60,parent=agg['config']['parent'],parent_sha256=agg['config']['parent_sha256'],source_hashes=ledger['source_hashes'],**kw));tick=time.time()
        with log.open('w') as f:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(P/'train.py'),str(path)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            ledger['active']=dict(pid=child.pid,arm=arm,kind=kind,log=str(log))
            try:
                publish();print(json.dumps(ledger['active']),flush=True)
                code=child.wait(timeout=max(.01,deadline-time.time()-60))
            except subprocess.TimeoutExpired:child.kill();child.wait();code=124
            finally:
                if child.poll() is None:child.kill();child.wait()
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=time.time()-tick));ledger['active']=None;publish()
        if code:raise RuntimeError(f'{arm}/{kind} exited{code}: {log}')
        return read(result_path)
    publish()
    try:
        for arm in agg['config']['arms']:
            rec=agg['runs'].setdefault(arm,dict(validation=[]));cp=rec.get('training',{}).get('checkpoint');current=rec.get('training',{}).get('step',4400)
            for target in agg['config']['targets']:
                if target<=current:continue
                trained=launch('train',arm,run/arm,checkpoint=cp,target=target);assert trained['status']=='completed' and trained['step']==target
                rec['training']=trained;cp=trained['checkpoint'];current=target;publish()
                val=launch('eval',arm,run/f'{arm}_val_{target:06d}',checkpoint=cp,split='val',eval_seed=agg['config']['val_seed'],n=128,heldout_locations=False)
                previous=max((v['selection_mean_auc'] for v in rec['validation']),default=-1);rec['validation'].append(val)
                if val['selection_mean_auc']>previous:rec.update(selected_checkpoint=cp,selected_step=target)
                publish()
            rec['test']=launch('eval',arm,run/f'{arm}_test',checkpoint=rec['selected_checkpoint'],split='test',eval_seed=agg['config']['test_seed'],n=512,heldout_locations=True);rec['status']='completed';publish()
        agg['parent_reference']=launch('eval','continuation',run/'parent_reference',checkpoint=None,split='test',eval_seed=agg['config']['test_seed'],n=512,heldout_locations=True)
        agg['feedback_interruption']=launch('eval','controller_feedback',run/'feedback_interruption',checkpoint=agg['runs']['controller_feedback']['selected_checkpoint'],split='test',eval_seed=agg['config']['test_seed'],n=512,heldout_locations=False,feedback_off=True)
        agg['status']='completed'
    except BaseException as error:agg.update(status='failed',error=repr(error));raise
    finally:
        ledger['status']=agg['status'];agg['wall_seconds']=time.time()-ledger['started_unix'];publish();write(run/'exit.json',dict(status=agg['status'],error=agg.get('error'),wall_seconds=agg['wall_seconds'],deadline_unix=deadline,recovery=recovery))
if __name__=='__main__':main()
