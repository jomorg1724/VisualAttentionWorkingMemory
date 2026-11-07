# Two-frame RViT with repeated native movie pools

Same [RViT architecture](../TwoFrameRViT/README.md), freshly initialized end to end.
Generate1000 native B12/B20/B28 movies (334/333/333, rotate extra per pool). Shuffle
within each length and shuffle condition order in every batch round. Every movie
is processed once per epoch for ten epochs before regeneration. Maxbatch32/micro4;
include13/14-movie tails with loss normalized by actual batch size.33updates per
epoch;330updates/10000presentations per pool. Final binary cross-entropy only,
Adam1e-4, no clipping, FP32 full BPTT, all parameters trainable.

Checkpoints save native pre/post-generation stream states, pool IDs, shuffle RNG
and cursor; raster movies can be regenerated without storing4.4GB in every checkpoint.
Progress reports unique generated/presented movies separately from presentations.
Validation uses independent fixed100/cell and finaltests200/cell. Firstvalidation100,
thenevery250updatesandterminal. No training on evaluation movies.

SameA40pod7f27p6jxpitihn/originaldeadline; seven complete ten-epoch pools pinned:
2310updates/70000presentations/7000unique movies. Originalhardshutdown remains
05:16:04UTC October3 /10:16:04PM PDT October2, no extension. Profile state discarded.
[Run status](RUN_STATUS.md) · [journal](../../LabJournal/krauzlis-rvit-replay.md).
