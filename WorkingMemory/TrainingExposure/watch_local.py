"""Finish local saved-data reporting, then combine the retrieved cloud companion."""
import sys,json,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    root=Path(read(HERE/'local_run.json')['run']);completed=set()
    while True:
        budget_path=root/'budget.json'
        if not budget_path.exists():time.sleep(10);continue
        budget=read(budget_path)
        if time.time()>budget['deadline_unix']-10:return
        local=root/'exit.json';cloud=HERE/'Cloud/retrieval_receipt.json';available=[]
        if local.exists() and read(local)['status']=='completed':available.append('control_10')
        if cloud.exists() and read(cloud).get('status')=='verified':
            p=Path(read(cloud)['results'])/'exit.json'
            if p.exists() and read(p)['status']=='completed':available.append('focused_50')
        key=tuple(available)
        if 'control_10' in available and key not in completed:
            proc=subprocess.run([sys.executable,'-B',str(HERE/'analyze.py')],capture_output=True,text=True,timeout=180)
            (HERE/'analysis_worker.log').write_text(proc.stdout+'\n'+proc.stderr)
            (HERE/'analysis_status.json').write_text(json.dumps(dict(arms=available,returncode=proc.returncode,finished_unix=time.time()),indent=2))
            if proc.returncode:raise RuntimeError('Saved-data report failed')
            completed.add(key)
            if len(available)==2:return
        if local.exists() and read(local)['status']!='completed':return
        time.sleep(45)
if __name__=='__main__':main()
