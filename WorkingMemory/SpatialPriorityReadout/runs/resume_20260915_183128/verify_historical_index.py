import hashlib,json,pathlib
p=pathlib.Path("/workspace/vawm_spatial_priority_scratch/spatial_priority_results/spatial_priority_readout_scratch/training")
rows=[json.loads(x) for x in (p/"checkpoint_index.jsonl").read_text().splitlines()]
assert all(hashlib.sha256((p/r["file"]).read_bytes()).hexdigest()==r["sha256"] for r in rows)
print(json.dumps(rows))
