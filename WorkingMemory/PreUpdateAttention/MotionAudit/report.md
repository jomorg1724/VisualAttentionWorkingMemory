# Saved-artifact motion audit

Motion was already substantially impaired in the spatial parent. Attention adds a clear D0 decision bias: it never predicts **down**, even though its down-versus-other probability ranking has AUC .738. A validation-only correction of three effective class offsets rescues **12.89 percentage points** of held-out D0 accuracy. This supports an output-bias contribution, while the remaining discrimination deficit and weak D24 transfer remain unresolved.

No model inference, GPU work, main-weight updates, or new training episodes were used. All four current models share the exact same 512 held-out motion base episodes, repeated at D0/D24. Labels, paired IDs and full metadata match. Class order below is right/up/left/down.

| Frozen model | D0 BA% / AUC | D24 BA% / AUC | D0 predicted counts | D24 predicted counts |
|---|---:|---:|---|---|
| Spatial parent4400 | 46.09 / .798 | 25.00 / .756 | 231 / 12 / 104 / 165 | 0 / 512 / 0 / 0 |
| Continuation8400 | 50.78 / .817 | 31.64 / .730 | 37 / 215 / 218 / 42 | 0 / 435 / 55 / 22 |
| Additive feedback8400 | 45.31 / .808 | 25.00 / .734 | 19 / 153 / 307 / 33 | 0 / 512 / 0 / 0 |
| Pre-update attention8400 | 39.65 / .774 | 35.35 / .701 | 13 / 348 / 151 / 0 | 123 / 279 / 110 / 0 |

Balanced labels make accuracy equal BA. AUC is macro one-versus-rest AUC of saved probabilities; it measures within-class ranking, not the cross-class comparison required by argmax. Attention's D0 per-class recalls are 3.13%, 98.44%, 57.03%, 0%. Its lower AUC than continuation means decision bias alone should not be assumed to explain the entire deficit. D24 right-class AUC is .489 for attention, despite other classes retaining .716–.810 ranking.

## Minimal calibration experiment

For attention and continuation separately, fit zero-sum offsets b to log(saved probabilities), minimizing mean cross-entropy + .01/2 × sum(b²), using only the selected8400 checkpoint's existing **128 D0 validation episodes**. Four zero-sum offsets have three effective free parameters. Fixed regularization, zero initialization and L-BFGS-B; no hyperparameter search or checkpoint reselection. Both fits converged. D24 uses the exact D0 offsets without further fitting.

| Model and test | Original BA% | Corrected BA% | Paired change pp [95% CI] |
|---|---:|---:|---:|
| Attention D0 | 39.65 | 52.54 | +12.89 [8.59, 16.99] |
| Continuation D0 | 50.78 | 56.64 | +5.86 [2.15, 9.38] |
| Attention D24 | 35.35 | 37.70 | +2.34 [−1.76, 7.04] |
| Continuation D24 | 31.64 | 31.05 | −0.59 [−3.32, 2.15] |

Attention D0 NLL improves 1.239→1.097; continuation 1.070→1.061. Corrected attention predicts every class, with down recall38.28%. Calibration closes much of the D0 accuracy gap but attention remains descriptively below equally calibrated continuation. This is a small analysis-only rescue on an already studied test set, not a fresh confirmatory benchmark or a deployed model change. Bootstrap intervals condition on the fitted offsets; they do not include calibration-sample or training-seed uncertainty. ECE alone is not an optimization target: continuation's ECE increases despite improved NLL/accuracy. Renormalized offsets can change probability-based OVR AUC through their shared denominator.

## Where the loss began and what the trajectories show

The **earlier, separately paired** spatial experiment compared Retention14800 and spatial4400 on test seed35973001: D0 BA79.10→46.68%, D24 BA77.54→25.00%; D24 AUC .935→.762. Thus the large historical motion decline predates attention. This earlier matched comparison is not paired with the current seed46973001 experiment and does not isolate architecture from training or task differences.

| Arm | Validation D0 BA% at5400 /6400 /7400 /8400 | Validation D24 BA% |
|---|---|---|
| Attention | 48.44 /55.47 /27.34 /37.50 | 25.00 /22.66 /26.56 /36.72 |
| Continuation | 54.69 /39.06 /50.78 /46.09 | 25.00 /25.00 /30.47 /28.13 |
| Feedback | 50.78 /31.25 /40.63 /44.53 | 25.00 /25.00 /25.00 /25.00 |

Attention's primary selection AUC rises .949→.993 while motion D0 fluctuates sharply. Actual source excludes motion from selection: equal mean AUC over eight single/binding cells. Actual CSV counts and the 80-update cycle agree: each arm has4000 added updates, 32,000 episodes; motion gets200 updates/1600 episodes at each delay, exactly10% total. Each primary cell gets3600 episodes. This explains what training prioritized, **not that allocation caused the loss**; no allocation comparison was conducted here.

All logged pre-clipping gradient norms are finite. Clipping fractions decline across successive1000-update blocks: attention70.6→46.5%, continuation73.3→53.0%, feedback72.2→51.7%. Large finite maxima occur in all arms (up to about300); they do not uniquely implicate attention instability. At the last recorded motion diagnostic, attention has nonzero early rate/adaptation gradients and3.98% inactive spatial units. These selected batches establish gradient flow, not useful retention or a causal explanation. Full recorded diagnostics are retained in findings.json.

## Saved metadata and the next diagnostic

Attention D0 is worse than continuation both when the duration winner equals the final direction (42.64% vs58.14%, n129) and when it does not (38.64% vs48.30%, n383). Its deficit is therefore not confined to a final-direction shortcut. At seven switches, accuracy is25.49% vs37.25% (n51); at two switches70% vs80% (n10). Most episodes have winner-count margins1/8 (n314) or2/8 (n153); attention D0 accuracy is38.22%/39.22%, versus continuation45.22%/56.86%. Sparse larger-margin strata and class mixtures prevent attributing these associations to switch count or margin alone. Full strata, denominators and confusions are saved.

Previously saved additive-branch blank interruption changed motionD24 BA by0 and AUC by−.000066 [−.000463,.000326]. It gives no evidence that ongoing additive current during blanks is necessary for that model's motion output; resumed feedback, encoding and training effects remain possible.

The concurrent frozen phase interventions can test whether changing attention's evidence-period routing rescues motion. Interpret any rescue alongside the demonstrated output bias. If those interventions remain ambiguous, one narrow next option is a fresh-split linear duration readout from the frozen final representation used by the existing comparator/head, compared with the deployed head. Do not infer representation erasure from failed argmax or weak probes. No such feature probe was run here.

## Reproduction and limits

Run `C:/Python310/python.exe -X utf8 -B WorkingMemory/PreUpdateAttention/MotionAudit/audit.py` from the repository. The script disables CUDA visibility before scientific imports, sets CPU threads1, imports no Torch, and writes findings.json. Saved file hashes pin all inputs. A first attempt encountered a Windows translation of a remote prediction path before producing results; correcting only that path resolved it. Final computation took about3seconds; no model or data were regenerated.

Intervals use1000 paired class-stratified base-episode resamples. The same512 bases recur across models/delays; they are not independent replications. Current test/calibration findings are exploratory, conditional on selected models and their existing validation selection. Cross-platform attention training versus local controls is disclosed in the original report. No main checkpoint, source renderer, training protocol, or cloud resource was changed.
