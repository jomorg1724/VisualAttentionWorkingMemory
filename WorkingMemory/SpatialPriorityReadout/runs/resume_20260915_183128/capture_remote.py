import csv,json,pathlib,subprocess
r=pathlib.Path("/workspace/vawm_spatial_priority_scratch_resume_1600")
b=json.load(open(r/"spatial_priority_results/budget.json"))
m=list(csv.DictReader(open(r/"spatial_priority_results/spatial_priority_readout_scratch/training/metrics.csv")))
idx=[json.loads(x) for x in open(r/"spatial_priority_results/spatial_priority_readout_scratch/training/checkpoint_index.jsonl")]
print(json.dumps(dict(supervisor_pid=int((r/"remote_supervisor.pid").read_text()),worker_pid=b["active"]["pid"],active=b["active"],first_metric=m[0],latest_metric=m[-1],metric_rows=len(m),checkpoint_index=idx,gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw","--format=csv,noheader,nounits"],text=True).strip())))
