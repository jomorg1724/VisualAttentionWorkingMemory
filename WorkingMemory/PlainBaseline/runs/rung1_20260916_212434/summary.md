# Rung 1 summary: plain baseline, one task per run

Run directory `WorkingMemory/PlainBaseline/runs/rung1_20260916_212434`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).

| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| image_recognition | image_recognition_N4_H3 | best | 200 | 12800 | 0.514 | 0.50 | 0.027 | 0.477 | 512 | [[206, 50], [199, 57]] |
| image_recognition | image_recognition_N4_H4 | best | 200 | 12800 | 0.512 | 0.50 | 0.023 | 0.485 | 512 | [[216, 40], [210, 46]] |
| image_recognition | image_recognition_N4_H5 | best | 200 | 12800 | 0.484 | 0.50 | -0.031 | 0.538 | 512 | [[228, 28], [236, 20]] |
| image_recognition | image_recognition_N4_H3 | terminal | 313 | 20032 | 0.525 | 0.50 | 0.051 | 0.514 | 512 | [[45, 211], [32, 224]] |
| image_recognition | image_recognition_N4_H4 | terminal | 313 | 20032 | 0.500 | 0.50 | 0.000 | 0.498 | 512 | [[68, 188], [68, 188]] |
| image_recognition | image_recognition_N4_H5 | terminal | 313 | 20032 | 0.504 | 0.50 | 0.008 | 0.523 | 512 | [[101, 155], [99, 157]] |
| krauzlis_cued_motion | krauzlis_cued_motion_B12 | best | 100 | 6400 | 0.500 | 0.50 | 0.000 | 0.494 | 512 | [[0, 217], [0, 295]] |
| krauzlis_cued_motion | krauzlis_cued_motion_B20 | best | 100 | 6400 | 0.500 | 0.50 | 0.000 | 0.506 | 512 | [[0, 222], [0, 290]] |
| krauzlis_cued_motion | krauzlis_cued_motion_B28 | best | 100 | 6400 | 0.500 | 0.50 | 0.000 | 0.515 | 512 | [[0, 223], [0, 289]] |
| krauzlis_cued_motion | krauzlis_cued_motion_B12 | terminal | 313 | 20032 | 0.500 | 0.50 | 0.000 | 0.522 | 512 | [[0, 217], [0, 295]] |
| krauzlis_cued_motion | krauzlis_cued_motion_B20 | terminal | 313 | 20032 | 0.500 | 0.50 | 0.000 | 0.518 | 512 | [[0, 222], [0, 290]] |
| krauzlis_cued_motion | krauzlis_cued_motion_B28 | terminal | 313 | 20032 | 0.500 | 0.50 | 0.000 | 0.534 | 512 | [[0, 223], [0, 289]] |
| motion_duration_cued | motion_duration_cued_D0 | best | 100 | 6400 | 0.250 | 0.25 | 0.000 | 0.499 | 512 | [[0, 96, 32, 0], [0, 96, 32, 0], [0, 96, 32, 0], [0, 96, 32, 0]] |
| motion_duration_cued | motion_duration_cued_D0 | terminal | 313 | 20032 | 0.250 | 0.25 | 0.000 | 0.498 | 512 | [[0, 0, 0, 128], [0, 0, 0, 128], [0, 0, 0, 128], [0, 0, 0, 128]] |
| orientation_cued | orientation_cued_D0 | best | 400 | 25600 | 0.492 | 0.50 | -0.016 | 0.469 | 512 | [[55, 201], [59, 197]] |
| orientation_cued | orientation_cued_D0 | terminal | 1563 | 100032 | 0.488 | 0.50 | -0.023 | 0.491 | 512 | [[149, 107], [155, 101]] |
| spatial_binding | spatial_binding_D0 | best | 313 | 20032 | 0.514 | 0.50 | 0.027 | 0.501 | 512 | [[39, 217], [32, 224]] |
| spatial_binding | spatial_binding_D0 | terminal | 313 | 20032 | 0.514 | 0.50 | 0.027 | 0.501 | 512 | [[39, 217], [32, 224]] |

## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)

**image_recognition** (completed, 2191404 params, 124 s): 100:0.500, 200:0.510, 300:0.496, 313:0.497
**krauzlis_cued_motion** (completed, 2191404 params, 1692 s): 100:0.500, 200:0.500, 300:0.500, 313:0.500
**motion_duration_cued** (completed, 2191404 params, 424 s): 100:0.250, 200:0.250, 300:0.250, 313:0.250
**orientation_cued** (completed, 2191404 params, 1206 s): 200:0.500, 400:0.508, 600:0.500, 800:0.504, 1000:0.500, 1200:0.500, 1400:0.500, 1563:0.496
**spatial_binding** (completed, 2191404 params, 283 s): 100:0.500, 200:0.500, 300:0.508, 313:0.516