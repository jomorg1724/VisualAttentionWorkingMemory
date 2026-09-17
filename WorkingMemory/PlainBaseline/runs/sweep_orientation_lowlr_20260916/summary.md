# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/sweep_orientation_lowlr_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| orientation_cued | orientation_cued_D0 | best | 1200 | 76800 | 0.533 | 0.50 | 0.066 | 0.513 | 512 | [[73, 183], [56, 200]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.520 | 0.50 | 0.039 | 0.518 | 512 | [[42, 214], [32, 224]] |
| orientation_cued | orientation_cued_D0 | best | 1563 | 100032 | 0.518 | 0.50 | 0.035 | 0.500 | 512 | [[83, 173], [74, 182]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.518 | 0.50 | 0.035 | 0.500 | 512 | [[83, 173], [74, 182]] |
| orientation_cued | orientation_cued_D0 | best | 1200 | 76800 | 0.510 | 0.50 | 0.020 | 0.509 | 512 | [[143, 113], [138, 118]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.477 | 0.50 | -0.047 | 0.497 | 512 | [[166, 90], [178, 78]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_cued** (completed, 2191916 params, 534 s): 200:0.516, 400:0.535, 600:0.461, 800:0.488, 1000:0.504, 1200:0.539, 1400:0.512, 1563:0.508
**orientation_cued** (completed, 2196716 params, 1055 s): 200:0.457, 400:0.508, 600:0.414, 800:0.504, 1000:0.516, 1200:0.531, 1400:0.504, 1563:0.535
**orientation_cued** (completed, 2196716 params, 529 s): 200:0.504, 400:0.469, 600:0.457, 800:0.457, 1000:0.477, 1200:0.527, 1400:0.504, 1563:0.496
**orientation_cued** (incomplete, 2197232 params, running/killed): 200:0.480, 400:0.484