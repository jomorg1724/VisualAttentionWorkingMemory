# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/long_orientation_20260916`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**orientation_cued** (incomplete, 2197232 params, running/killed): 500:0.457, 1000:0.492, 1500:0.492, 2000:0.465, 2500:0.500, 3000:0.527, 3500:0.477, 4000:0.504, 4500:0.484, 5000:0.531, 5500:0.461, 6000:0.512, 6500:0.500, 7000:0.492, 7500:0.434, 8000:0.496, 8500:0.512