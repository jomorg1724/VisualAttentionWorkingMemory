"""Finish this authorized analysis after inference; never launch or extend inference."""
import json,time
from .design import OUT
def main():
    deadline=json.loads((OUT/'budget.json').read_text())['deadline'];last=-5000;phase=None
    while not (OUT/'completion.json').exists():
        if time.time()>deadline:raise TimeoutError('Original analysis deadline reached')
        if (OUT/'failure.json').exists():raise RuntimeError((OUT/'failure.json').read_text())
        if (OUT/'exit.json').exists():raise RuntimeError('Inference exited without completion')
        s=json.loads((OUT/'status.json').read_text())
        if s['presentations']-last>=5000 or s['phase']!=phase:
            print(json.dumps(s),flush=True);last=s['presentations'];phase=s['phase']
        time.sleep(20)
    while not (OUT/'exit.json').exists():
        if time.time()>deadline:raise TimeoutError('Original analysis deadline reached')
        time.sleep(1)
    assert json.loads((OUT/'exit.json').read_text())['exit_code']==0
    print('Inference complete. Building figures, tables, animations and interactive gallery.',flush=True)
    from .report import main as report
    report()
    from .audit import main as audit
    audit()
if __name__=='__main__':main()
