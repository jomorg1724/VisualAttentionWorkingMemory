# Frozen VDA analysis — October 9

The user requested the previous analysis package for VDA4/9/16 and clarified
the central experiment: **100% cue bottom-right; change top-left; inhibit
bottom-right attention only during the change frames**. The selected model is
expanded-suite best16000, after all16000 updates and11264 final-test trials.
Final VDA balanced accuracy averaged across theta conditions:4=.71223,
9=.69099,16=.69412. Best/latest independently reloaded with86 finite aligned
Adam states; production supervisor exited0. Doom remains unstarted.

## Prospective experiment

One finite7200-second local inference allocation, no training/optimizer/cloud
or automatic extension. All original checkpoints, stimuli and learned sources
remain unchanged. Exact native-renderer equality, zero-angle/catch equality,
zero-bias and global-softmax-shift tests passed before production.

- **Primary:** VDA4, bottom-right100% cue,18-degree top-left or bottom-right
  changes plus no-change catches.384 matched scenes, balanced three colors and
  both rotation signs. Sham, bottom-right inhibition−2/−4/−8, top-left
  stimulation+4, and opposite-direction controls. Bias acts only frames5–6.
 8064 presentations. Four benefit/four reciprocal-cost tests at dose4,
  exact paired binary tests with Holm correction across8.
- **Native check:**512 fresh trials for each of9 trained VDA theta conditions;
  4608 presentations. Fresh203101 namespace.
- **Psychometrics:** absolute rotations2/4/8/12/18/26/40/65 degrees plus catches,
  native validity levels, cued and diagonal-opposite changes,96 scenes per cell.
  Native-impossible validity/event combinations omitted here.18144
  presentations; fresh203201 namespace. Actual change magnitude, not the
  training distribution's maximum theta, is the psychometric horizontal axis.
- **Exploratory interventions:** three sizes,75%/100% cue validity,18/40degree
  changes and catches, both diagonal cue arrangements,48 matched scenes per
  geometry.35 arms: dose, cue/sample/blank/change timing, both sites, a blank
  area and global-shift controls.100800 presentations; namespace203301.
  Mirrored geometries share scenes and are clustered together in uncertainty
  calculations. The primary test uses independent203403 scenes.
- **Attention atlas:**90groups ×7 timesteps,48-scene condition means and three
  real individual examples per group.100% cue,18-degree changes/catches,
  both cue corners, sham and four change-time±4 interventions. Individual
  overlays are paired with their own maps; the condition mean is labeled
  separately. All colors/signs are pooled in the means.

Total131616 model presentations; initial unit/control checks are separate.
`protocol.json` is written before these observations are collected.

## Interpretation

An uncued change with a100% cue is an intentional cue-violation test, outside
the native100%-cue training distribution. It remains a positive VDA label:
any physical change is reportable. False alarms are no-change reports only.
We report cued and uncued detection, catch false alarms, paired balanced
accuracy, d-prime/criterion and attention allocation separately.

Manipulations add to the pre-softmax spatial priority logits for tokens whose
centers lie in the selected cell. They are not literal electrical stimulation
or neuronal silencing. Cell/token footprints differ by VDA size. Softmax
redistributes probability and recurrent effects may persist after an early
intervention. The model has one CLS query: its query-averaged incoming attention
is exactly its one recorded row, reshaped10x10 and nearest-projected100x100.

The maps preserve one fixed logarithmic color scale across conditions/times.
Group means may blur Gabor orientations and cue colors; this is disclosed.
One trained checkpoint supports a computational causal test, not independent
training-seed replication or equivalence to a biological experiment.

## Outputs and validation

`experiment.py` writes trial-level outcomes, paired pixel hashes, every-timestep
attention mass and atlas arrays. `report.py` writes tables, PNG/PDF figures,
three comparison GIFs and `run01/gallery/index.html`. `audit.py` checks exact
counts, pairing, timing, invariance controls, checkpoint/source hashes and every
exported image/map/overlay sprite against saved arrays. Completion requires the
worker's exit0 and `completion_verified.json`, not merely launcher existence.
