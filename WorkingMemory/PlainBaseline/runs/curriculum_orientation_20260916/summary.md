# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/curriculum_orientation_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| orientation_cued | orientation_cued_D0 | best | 800 | 51200 | 0.996 | 0.50 | 0.992 | 1.000 | 512 | [[256, 0], [2, 254]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_cued | orientation_cued_D0 | best | 1000 | 64000 | 0.998 | 0.50 | 0.996 | 1.000 | 512 | [[256, 0], [1, 255]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.998 | 0.50 | 0.996 | 1.000 | 512 | [[256, 0], [1, 255]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_cued** (completed, 2197746 params, 2131 s): 200:0.984, 400:0.988, 600:0.996, 800:1.000, 1000:0.996, 1200:0.996, 1400:1.000, 1563:1.000
**orientation_cued** (completed, 2197746 params, 426 s): 200:0.977, 400:0.988, 600:0.996, 800:0.996, 1000:1.000, 1200:0.996, 1400:0.996, 1563:1.000