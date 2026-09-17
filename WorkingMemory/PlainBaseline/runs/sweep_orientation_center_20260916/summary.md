# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/sweep_orientation_center_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| orientation_cued | orientation_cued_D0 | best | 1563 | 100032 | 0.535 | 0.50 | 0.070 | 0.506 | 512 | [[196, 60], [178, 78]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.535 | 0.50 | 0.070 | 0.506 | 512 | [[196, 60], [178, 78]] |
| orientation_cued | orientation_cued_D0 | best | 200 | 12800 | 0.496 | 0.50 | -0.008 | 0.505 | 512 | [[83, 173], [85, 171]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.500 | 0.50 | 0.000 | 0.518 | 512 | [[256, 0], [256, 0]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_cued** (completed, 2196204 params, 901 s): 200:0.531, 400:0.496, 600:0.473, 800:0.496, 1000:0.488, 1200:0.539, 1400:0.480, 1563:0.551
**orientation_cued** (incomplete, 2196204 params, running/killed): 200:0.500, 400:0.500, 600:0.477, 800:0.457, 1000:0.500, 1200:0.500
**orientation_cued** (completed, 2196204 params, 373 s): 200:0.527, 400:0.500, 600:0.500, 800:0.516, 1000:0.500, 1200:0.477, 1400:0.500, 1563:0.500