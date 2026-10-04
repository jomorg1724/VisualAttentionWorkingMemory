# Earlier live-checkpoint validation

Read-only CPU snapshots provide earlier performance visibility for the active
sixteen-head KDA and delayed-frame CNN-GRU. They load a hashed immutable saved
checkpoint, freeze all weights, use the same native validation streams and emit
per-condition/event metrics. No training/RNG/optimizer state or official model
selection changes. They use two CPU threads, no GPU, and at most1,200seconds
inside the existing run deadline. A measured longest-movie profile chooses100
trials per condition if feasible, otherwise50before evaluation. Counts and
checkpoint step are explicit; partial cells are not a complete battery.

Outputs are outside active run directories so the extra evaluation cannot race
with final artifact manifests. The KDA cloud snapshot has a bounded parent
mirror for retrieval; it never provisions or restarts any pod. These are live
validation results, not selected models or fresh final tests. The new RViT also
has official in-run validation at100,250,500,...terminal.

```bash
python -m SecondPass.EarlyValidation.snapshot kda16 RUN --output OUTSIDE_RUN/latest.json --deadline UNIX
```

[Completed KDA snapshot](../SequenceKDA16/CloudRuntime/early_validation/latest.json) ·
[Local CNN-GRU snapshot](../DelayedFrameGRU/LocalRuntime/early_validation/latest.json).
