"""Small CPU-only integration through the real worker finalization path."""
import json
import time
import torch
from . import worker as w


def test_cpu_run_finalizes_selected_terminal(tmp_path):
    torch.set_num_threads(1)
    start=time.time();b=dict(cap_started=start,hard_deadline=start+28800,deadline=start+28200,wall_cap_seconds=28800,retrieval_reserve_seconds=600)
    cfg=dict(b,device='cpu',source_hashes={},effective_batch=4,microbatch=4,disposable_profile=True,
        max_steps=1,final_reserve_seconds=120,update_estimate_seconds=10,estimated_one_test_seconds=30,
        checkpoint_every=1000,validation_steps=[1])
    for name,obj in [('budget.json',b),('config.json',cfg),('allocation.json',dict(max_steps=1))]:
        (tmp_path/name).write_text(json.dumps(obj))
    # Keep a whole100-event validation block (all classes/subgroups); final tests
    # are only4/cell for this CPU harness test, NOT deployable scientific results.
    def cpu_eval(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
        return w.evaluate(model,split,n,100 if split=='val' else 4,4,'cpu',output,deadline,cells)
    run=w.clone(w._run,evaluate=cpu_eval)
    assert run(tmp_path)==0
    w.completion(tmp_path,'complete')
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['terminal_step']==report['selected_step']==1 and report['final_coverage_complete']
    assert (tmp_path/'selected.pt').is_file()
    complete=json.loads((tmp_path/'cloud_completion.json').read_text())
    assert complete['actual_updates']==1 and complete['status']=='complete'
