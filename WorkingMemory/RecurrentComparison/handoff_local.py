"""User-directed queue migration: adopt LSTM, exclude local EI, keep deadline."""
import sys,time,json,subprocess,ctypes
from ctypes import wintypes
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write


def run(root,old_pid):
    root=Path(root).resolve();old=psutil.Process(old_pid)
    command=old.cmdline()
    if not any(x.endswith('RecurrentComparison/sweep.py') or x.endswith('RecurrentComparison\\sweep.py') for x in command) or str(root) not in command:
        raise RuntimeError('Exact supervisor identity mismatch')
    old.suspend();terminated=False;handle=None
    try:
        ledger=read(root/'budget.json');aggregate=read(root/'aggregate.json');active=ledger['active'];number=len(ledger['events'])
        job=read(root/f'job_{number:03d}.json')
        if active['arm']!='lstm' or job['config']['arm']!='lstm':raise RuntimeError('Only current LSTM worker may be adopted')
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x00100000|0x1000,False,active['pid'])
        if not handle:raise RuntimeError('Could not retain exact active worker handle')
        old.terminate();old.wait(10);terminated=True
        write(root/'handoff_local.json',dict(status='adopting_lstm',old_supervisor_pid=old_pid,new_controller_pid=psutil.Process().pid,
            adopted_worker=active,deadline_unix=ledger['deadline_unix'],local_ei_excluded=True,
            reason='Explicit user instruction: run neuro arm on RunPod, preserve active local LSTM',runtime_source_edited=False))
    except BaseException:
        if not terminated and old.is_running():old.resume()
        raise
    deadline=ledger['deadline_unix'];cfg=aggregate['config']['recipe'];record=aggregate['runs']['lstm']
    def publish():write(root/'aggregate.json',aggregate);write(HERE/'results.json',aggregate);write(root/'budget.json',ledger)
    def accept(kind,result,wall):
        if kind=='train':record['training']=dict(result,training_wall_seconds=record.get('training',{}).get('training_wall_seconds',0)+wall)
        elif kind=='eval':
            record['validation'].append(result)
            if 'selected_checkpoint' not in record or result['selection_mean_auc']>max(v['selection_mean_auc'] for v in record['validation'][:-1]):
                record.update(selected_checkpoint=result['checkpoint'],selected_step=result['step'])
        publish()
    try:
        # A retained OS process handle prevents PID reuse from changing which
        # process supplies the exit code. The worker itself is uninterrupted.
        while kernel.WaitForSingleObject(handle,1000)==258:
            if time.time()>=deadline:
                psutil.Process(active['pid']).kill();raise TimeoutError('Original absolute deadline reached')
        code=wintypes.DWORD();kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle);handle=None
        wall=time.time()-active['started_unix']
        ledger['events'].append(dict(active,returncode=int(code.value),wall_seconds=wall,adopted_after_supervisor_migration=True));ledger['active']=None
        if code.value:raise RuntimeError('Adopted LSTM worker exit '+str(code.value))
        accept(job['kind'],read(job['result']),wall)

        def launch(kind,out,**kw):
            number=len(ledger['events']);job_path=root/f'job_{number:03d}.json';result=root/f'job_{number:03d}_result.json';log_path=root/f'worker_{number:03d}.log'
            write(job_path,dict(kind=kind,config=dict(cfg,arm='lstm'),out=str(out),deadline=deadline,result=str(result),
                source_hashes=ledger['source_hashes'],parent_checkpoint=aggregate['config']['parent_checkpoint'],parent_sha256=aggregate['config']['parent_sha256'],**kw))
            tick=time.time()
            with log_path.open('w') as log:
                p=subprocess.Popen([sys.executable,'-B',str(HERE/'train.py'),str(job_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                ledger['active']=dict(pid=p.pid,kind=kind,arm='lstm',started_unix=tick,log=str(log_path));publish()
                try:code=p.wait(timeout=max(.01,deadline-time.time()))
                except subprocess.TimeoutExpired:p.kill();p.wait();code=124
                finally:
                    if p.poll() is None:p.kill();p.wait()
            wall=time.time()-tick;ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=wall));ledger['active']=None;publish()
            if code:raise RuntimeError('Local LSTM '+kind+' exit '+str(code))
            return read(result),wall

        for target in aggregate['config']['targets']:
            if record['training'].get('step',0)<target:
                result,wall=launch('train',root/'lstm',target=target,checkpoint=record['training']['checkpoint']);accept('train',result,wall)
            if not any(v['step']==target for v in record['validation']):
                result,wall=launch('eval',root/f'lstm_validation_{target:06d}',checkpoint=record['training']['checkpoint'],split='val',eval_seed=1193001,n_per_cell=128,resamples=0)
                accept('eval',result,wall)
        for reset in (False,True):
            name='reset_test' if reset else 'test'
            if not record.get(name):
                result,_=launch('eval',root/('lstm_'+name),checkpoint=record['selected_checkpoint'],split='test',eval_seed=1293001,n_per_cell=512,resamples=500,reset_memory=reset)
                record[name]=result;publish()
        record['status']='completed';aggregate['status']='local_completed_remote_pending';ledger['status']=aggregate['status'];publish()
        write(root/'local_exit.json',dict(status='completed',wall_seconds=time.time()-ledger['started_unix'],deadline_unix=deadline,
            local_ei_excluded=True,original_supervisor_user_terminated=old_pid,local_model='lstm'))
        write(root/'handoff_local.json',dict(status='local_completed_remote_pending',old_supervisor_pid=old_pid,new_controller_pid=psutil.Process().pid,
            deadline_unix=deadline,local_ei_excluded=True,runtime_source_edited=False))
    except BaseException as e:
        if handle:kernel.CloseHandle(handle)
        aggregate['status']='local_handoff_failed';aggregate['error']=repr(e);publish()
        write(root/'local_exit.json',dict(status='failed',error=repr(e),deadline_unix=deadline));raise

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
