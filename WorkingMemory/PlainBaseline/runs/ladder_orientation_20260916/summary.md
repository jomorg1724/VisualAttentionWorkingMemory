# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/ladder_orientation_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| orientation_ring | orientation_ring_D0 | best | 1400 | 89600 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_ring | orientation_ring_D0 | terminal | 1563 | 100032 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_ring | orientation_ring_D0 | best | 600 | 38400 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_ring | orientation_ring_D0 | terminal | 1563 | 100032 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_sign | orientation_sign_D0 | best | 1200 | 76800 | 0.994 | 0.50 | 0.988 | 1.000 | 512 | [[256, 0], [3, 253]] |
| orientation_sign | orientation_sign_D0 | terminal | 1563 | 100032 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_single | orientation_single_D0 | best | 600 | 38400 | 0.992 | 0.50 | 0.984 | 1.000 | 512 | [[256, 0], [4, 252]] |
| orientation_single | orientation_single_D0 | terminal | 1563 | 100032 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_single | orientation_single_D0 | best | 600 | 38400 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |
| orientation_single | orientation_single_D0 | terminal | 1563 | 100032 | 1.000 | 0.50 | 1.000 | 1.000 | 512 | [[256, 0], [0, 256]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_ring** (completed, 2197232 params, 421 s): 200:0.477, 400:0.535, 600:0.992, 800:0.996, 1000:0.996, 1200:0.996, 1400:1.000, 1563:1.000
**orientation_ring** (completed, 2197746 params, 408 s): 200:0.484, 400:0.996, 600:1.000, 800:0.992, 1000:1.000, 1200:1.000, 1400:0.996, 1563:1.000
**orientation_sign** (completed, 2197746 params, 201 s): 200:0.484, 400:0.488, 600:0.500, 800:0.566, 1000:0.988, 1200:1.000, 1400:1.000, 1563:1.000
**orientation_single** (completed, 2197232 params, 1930 s): 200:0.938, 400:0.992, 600:1.000, 800:1.000, 1000:1.000, 1200:0.996, 1400:1.000, 1563:1.000
**orientation_single** (completed, 2197232 params, 451 s): 200:0.988, 400:0.996, 600:1.000, 800:1.000, 1000:1.000, 1200:1.000, 1400:1.000, 1563:1.000