"""Rebuild the existing task manuscripts with section-level biological context."""
import json,time
from .papers import PAPERS,compile_paper
from .design import OUT,TASKS
from .neuroscience_report import enhance_source

def main():
    for task in TASKS:
        p=PAPERS/f'{task}.tex';source=p.read_text(encoding='utf8')
        revised=enhance_source(task,source)
        assert enhance_source(task,revised)==revised,task
        (OUT/'publication_status.json').write_text(json.dumps(dict(phase='integrating_neuroscience_into_task_papers',task=task,time=time.time())))
        compile_paper(p)
        print(task,'compiled',flush=True)
    print('All 19 task papers rebuilt; QA and volume packaging remain.',flush=True)

if __name__=='__main__':main()
