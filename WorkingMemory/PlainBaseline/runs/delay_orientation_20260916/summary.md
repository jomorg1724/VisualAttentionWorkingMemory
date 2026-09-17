# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/delay_orientation_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| orientation_cued | orientation_cued_D0 | best | 2000 | 128000 | 0.998 | 0.50 | 0.996 | 1.000 | 512 | [[255, 1], [0, 256]] |
| orientation_cued | orientation_cued_D4 | best | 2000 | 128000 | 0.537 | 0.50 | 0.074 | 0.542 | 512 | [[174, 82], [155, 101]] |
| orientation_cued | orientation_cued_D12 | best | 2000 | 128000 | 0.496 | 0.50 | -0.008 | 0.502 | 512 | [[166, 90], [168, 88]] |
| orientation_cued | orientation_cued_D24 | best | 2000 | 128000 | 0.480 | 0.50 | -0.039 | 0.480 | 512 | [[159, 97], [169, 87]] |
| orientation_cued | orientation_cued_D0 | terminal | 3125 | 200000 | 0.996 | 0.50 | 0.992 | 1.000 | 512 | [[254, 2], [0, 256]] |
| orientation_cued | orientation_cued_D4 | terminal | 3125 | 200000 | 0.498 | 0.50 | -0.004 | 0.472 | 512 | [[138, 118], [139, 117]] |
| orientation_cued | orientation_cued_D12 | terminal | 3125 | 200000 | 0.512 | 0.50 | 0.023 | 0.508 | 512 | [[104, 152], [98, 158]] |
| orientation_cued | orientation_cued_D24 | terminal | 3125 | 200000 | 0.490 | 0.50 | -0.020 | 0.491 | 512 | [[95, 161], [100, 156]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_cued** (completed, 2197746 params, 2753 s): 250:0.637, 500:0.612, 750:0.612, 1000:0.618, 1250:0.604, 1500:0.620, 1750:0.629, 2000:0.645, 2250:0.643, 2500:0.606, 2750:0.608, 3000:0.627, 3125:0.629