# More training, motion allocation, and a residual: design notes

Design only. No run, profile, model change, or compute allowance is started here. Prefer two staged, two-arm comparisons over an immediate four-arm factorial. This keeps each conclusion interpretable and avoids paying for a residual experiment if ordinary continuation already resolves the practical problem.

## Experiment 1: unchanged architecture, two fixed allocations

Both arms start from the exact selected pre-update-attention8400 checkpoint. Target4000 additional updates × batch8 =32,000 fresh episodes per arm, subject to a separately authorized finite budget and measured throughput. Do not reset learned weights, per-parameter Adam moments/steps, task-local sampler state or RNG. Preserve the actual optimizer grouping: sensory and ordinary motion/orientation heads currently use3e-5; attention, memory/input/output, comparator and binding head use3e-4. No automatic loss balancing, new cues, targets, delays, labels, or learning-rate changes.

|80-update shuffled cycle|Each of8 single/binding delay cells|Each of2 motion delay cells|Total added exposure|
|---|---:|---:|---:|
|Existing90/10 control|9 updates;3600 episodes at endpoint|4 updates;1600 episodes at endpoint|32,000 episodes|
|Motion-focused50/50|5 updates;2000 episodes at endpoint|20 updates;8000 episodes at endpoint|32,000 episodes|

Load identical parent task-local streams in both arms; each cell receives the same subsequent per-cell example prefix. The schedule changes when those prefixes are visited, not their underlying examples. Deterministically shuffle each declared cycle and save its identity/offset with checkpoints. This is an explicit schedule migration, not a claim that the arms follow the same global stochastic sequence.

Planned looks at +800/+1600/+2400/+3200/+4000 updates are complete80-cycles. Plot every cell against both total added updates and actual fresh per-cell episodes. Focused+800 and control+4000 have an exact match of1600 new episodes per motion cell. Their different total training/nonmotion history still prevents treating that comparison as an isolated per-example causal effect.

Only unchanged90/10 continuation versus its paired untouched8400 baseline directly tests whether additional training under the existing recipe helps. Focused versus control at equal total updates tests allocation. It does not establish “more time” alone, nor can a finite unsuccessful horizon establish that more training would never help.

The already completed three-parameter logit-offset audit supplies evidence that decision bias contributes. Repeat that same fixed validation-only calibration as a cheap secondary saved-output analysis at selected endpoints if desired, with identical treatment across arms. Keep raw native argmax the primary outcome. Calibration must not become a hidden training objective or a replacement for safeguarding native performance.

## Selection and preservation of existing abilities

Evaluate untouched8400 once on the exact fresh validation/test items used for each new arm. The earlier79.30% singleD24 and99.22% bindingD24 numbers are context, not the numerical denominators for a new paired test. Keep all10 standard cells and the existing four held-out-center binding cells in final reporting. Repeated delays/models share base episodes; binding resamples four-case blocks.

Use256 validation episodes per standard cell at each planned look if the budget permits. Provisional selection: among checkpoints whose validation BA has not dropped more than2 percentage points in any standard cell relative to paired8400, maximize mean motion AUC; break ties by the worst primary-cell BA change, then the earlier checkpoint. If none qualifies, report the best minimum-cell-change checkpoint descriptively and mark that no checkpoint met the preservation screen. These noisy validation point estimates select a candidate; they are not statistical proof of noninferiority and never stop healthy training early.

For final acceptance, predeclare a practical **2-percentage-point BA noninferiority margin per cell**, including singleD24, every binding delay and held-out-center condition, and both motion delays. Require all paired one-sided lower confidence bounds to exceed−2pp before claiming preservation. Control the familywise error across14 cells, for example through simultaneous paired bootstrap bounds or conservative Bonferroni-adjusted one-sided bounds. A useful motion improvement can be defined separately as at least5pp BA on one motion delay, with a positive paired95% interval and preservation of the other cells. Report AUC and confusion matrices beside BA so class-bias improvements do not conceal ranking losses.

512 episodes per cell is an efficient screening endpoint, not a promise of resolving2pp noninferiority. With moderately discordant paired decisions, even a zero point difference can have a lower bound below−2pp. Mark that outcome **inconclusive**, not preserved or equivalent. A later confirmatory evaluation may need a few thousand independent bases per critical family, with its size fixed from validation discordance and a new budget; it is not automatically authorized. No equivalence claim follows from nonsignificance. Historical test reuse and one training seed also limit generalization beyond these fitted models.

## Experiment 2: one residual architecture change, same teaching

If an architecture experiment is still warranted, compare residual absent/present from the **same** checkpoint and under the **same** fixed allocation chosen from Experiment1 validation. If Experiment1 finds no acceptable continuation, retain8400 as the common parent. Warm-start all compatible parameters and Adam histories; only genuinely new residual parameters get fresh optimizer state. Both arms receive equal additional updates and shared per-cell prefixes. Keep all tasks, objectives and safeguards above.

The residual specification must identify the exact accumulator tensor and destination layer, preserve frame causality, and explain which original routes already access that information. The current model already sends the final sensory field to a pooled sensory branch and sends current field/previous memory through the comparator. A new deeper residual therefore tests its particular integration/optimization route, not simply whether the model was allowed to see sensory information. Prefer one documented destination and an initially zero residual projection with a nonzero input path, so the parent behavior is initially preserved and the projection can learn immediately. Do not cross multiple skip locations or normalization choices in this experiment.

A staged design estimates the residual's effect under the chosen schedule; it cannot estimate an allocation×residual interaction. If that interaction becomes the actual scientific question, then a2×2 from one common parent is required:90/10 versus50/50 crossed with residual absent/present, equal32,000 added episodes per cell. It doubles arm count from two to four and should not be launched merely because the factorial is available. Stage selection and sequential training also mean the two experiments are not independent replications.

Scope recommendation: finish and inspect each finite experiment rather than committing in advance to a new architecture or an endless horizon. No mandatory accuracy gate, automatic extension, or unrequested cloud job follows from this design.
