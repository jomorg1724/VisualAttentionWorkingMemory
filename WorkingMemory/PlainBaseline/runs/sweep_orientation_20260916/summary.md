# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/sweep_orientation_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| orientation_cued | orientation_cued_D0 | best | 400 | 25600 | 0.508 | 0.50 | 0.016 | 0.502 | 512 | [[157, 99], [153, 103]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.498 | 0.50 | -0.004 | 0.499 | 512 | [[184, 72], [185, 71]] |
| orientation_cued | orientation_cued_D0 | best | 200 | 12800 | 0.514 | 0.50 | 0.027 | 0.507 | 512 | [[128, 128], [121, 135]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.498 | 0.50 | -0.004 | 0.516 | 512 | [[254, 2], [255, 1]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_cued** (incomplete, 2191916 params, running/killed): 200:0.500, 400:0.488, 600:0.484, 800:0.496, 1000:0.484, 1200:0.551, 1400:0.500
**orientation_cued** (completed, 2196716 params, 255 s): 200:0.508, 400:0.543, 600:0.516, 800:0.488, 1000:0.500, 1200:0.480, 1400:0.504, 1563:0.469
**orientation_cued** (completed, 2196204 params, 345 s): 200:0.539, 400:0.539, 600:0.492, 800:0.480, 1000:0.500, 1200:0.500, 1400:0.500, 1563:0.500