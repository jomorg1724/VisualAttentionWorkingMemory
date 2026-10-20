"""Resume publication QA from completed experiments without rerunning inference."""
import json,os,time
from .finalize import OUT,TASKS,PAPERS,write,export_csv,compile_paper,audit_data,synthesis,audit_pdfs

def main():
    stop=OUT/'stop_requested.json'
    if stop.exists():
        stop.rename(OUT/f'stop_requested_before_publication_resume_{int(time.time())}.json')
    anchor='Standardization is fitted on the inner fitting subset during selection and on the full fitting set for the final held-out evaluation.'
    addition=' Each standardized full-coordinate vector is additionally divided by the square root of its coordinate count; these penalty values refer to that normalized feature scale. The full-crop reconstruction and candidate-reconstruction regressions use the same scaling convention.'
    for task in TASKS:
        if stop.exists():
            write(OUT/'publication_status.json',dict(phase='paused',time=time.time()))
            return
        write(OUT/'publication_status.json',dict(phase='final_layout_and_methods_review',task=task,time=time.time(),pid=os.getpid()))
        path=OUT/task/'behavior.json'
        data=json.loads(path.read_text(encoding='utf8'))
        for row in data['factors']:
            row.pop('ci95',None)
            row['uncertainty']='Descriptive marginal; repeated-scene uncertainty is supplied by the matched timing analyses and task-level cluster bootstrap.'
        write(path,data);export_csv(task)
        path=PAPERS/f'{task}.tex';source=path.read_text(encoding='utf8')
        assert anchor in source,task
        if addition.strip() not in source:source=source.replace(anchor,anchor+addition)
        path.write_text(source,encoding='utf8');compile_paper(path)
        print(task,'compiled',flush=True)
    audit_data();synthesis();audit_pdfs()
    write(OUT/'publication_status.json',dict(phase='awaiting_final_visual_review',time=time.time(),pid=os.getpid()))

if __name__=='__main__':main()
