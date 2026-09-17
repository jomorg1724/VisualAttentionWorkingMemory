# Frozen pre-update attention mechanism diagnostic

The strongest result is specific to the blank interval: excluding memory tokens during the24 inserted blanks reduces delayed-orientation accuracy from79.30% to50.00%, while excluding them during sample, query or probe does not produce a comparable loss. This establishes dependence on the learned routing during blanks. The intervention both removes recirculated memory values and renormalizes attention toward the visual blank input, so it does not uniquely separate active memory refresh from rejecting irrelevant sensory drive.

Motion has at least two contributing problems. Acute routing changes during moving frames improve accuracy modestly (roughly3–6percentage points), while the independent validation-only class-offset calibration improves immediate motion by12.89points. These effects come from different interventions and must not be added together. Neither result explains the entire historical motion decline, which began before this attention model.

Frozen attention8400 and ordinary continuation8400. Primary prefixes: 512 orientation and 512 motion episodes, reusing existing heldout evidence; orientationD0 uses 128. No model parameters, task images, cues, labels, losses or sampling rules changed.

| Condition | Model / acute intervention | BA% | AUC | Change from attention baseline pp [95%CI] |
|---|---|---:|---:|---:|
| single_D0 | attention/baseline | 100.00 | 1.000 | — |
| single_D0 | attention/exclude_sample | 100.00 | 1.000 | +0.00 [+0.00,+0.00] |
| single_D0 | attention/exclude_query | 100.00 | 1.000 | +0.00 [+0.00,+0.00] |
| single_D0 | attention/exclude_probe | 100.00 | 1.000 | +0.00 [+0.00,+0.00] |
| single_D0 | continuation/baseline | 100.00 | 1.000 | +0.00 [+0.00,+0.00] |
| single_D24 | attention/baseline | 79.30 | 0.895 | — |
| single_D24 | attention/exclude_sample | 80.27 | 0.886 | +0.98 [-0.98,+3.12] |
| single_D24 | attention/exclude_blanks | 50.00 | 0.573 | -29.30 [-33.01,-25.78] |
| single_D24 | attention/exclude_query | 81.25 | 0.889 | +1.95 [-0.98,+4.49] |
| single_D24 | attention/exclude_probe | 80.08 | 0.896 | +0.78 [+0.00,+1.76] |
| single_D24 | continuation/baseline | 58.40 | 0.672 | -20.90 [-26.37,-15.62] |
| motion_D0 | attention/baseline | 39.65 | 0.774 | — |
| motion_D0 | attention/exclude_moving | 42.77 | 0.793 | +3.12 [+0.78,+5.47] |
| motion_D0 | attention/bypass_moving | 43.55 | 0.790 | +3.91 [+0.39,+7.23] |
| motion_D0 | continuation/baseline | 50.78 | 0.817 | +11.13 [+7.62,+14.84] |
| motion_D24 | attention/baseline | 35.35 | 0.701 | — |
| motion_D24 | attention/exclude_moving | 38.48 | 0.723 | +3.12 [+0.20,+6.05] |
| motion_D24 | attention/bypass_moving | 41.21 | 0.744 | +5.86 [+1.76,+9.96] |
| motion_D24 | continuation/baseline | 31.64 | 0.730 | -3.71 [-6.45,-0.78] |

## Timing and intervention meaning

Orientation indices: instruction0; sample1–2; D24 inserted blanks3–26; identity query27; probe/report28. D0 query3 and probe4. Memory-source exclusion sets memory-key logits to negative infinity then renormalizes the same visual keys/values. Previous-memory queries and recurrent E/I state updates remain. Motion instruction0; initial dots reference1; transition-bearing frames2–9; D24 blanks10–33; report34. The motion bypass sends H directly through the inherited memory_input normalization and input convolution on frames2–9 only. The ordinary trained model accepts no intervention metadata.

The sensory field sequence is cached once per frozen model, then reused unchanged across interventions. This is exact because neither tested model feeds memory into the sensory encoder. Existing sensory and comparator decision routes remain unchanged. CPU checks confirm ordinary/wrapped baseline equality, empty-phase no-op equality, source-mass exclusion and unchanged probe-only comparator contributions. Branch logits decompose the final decision into sensory, postupdate-memory and old-memory/current-field comparator projections, adding classifier bias once.

## Scope of inference

- Acute interventions are out of distribution and do not isolate how benefits were learned.
- No rescue does not establish irreversible loss of information.
- Memory-source exclusion preserves previous-memory queries and native E/I recurrence, so it does not eliminate all memory influence.
- Probe-only memory-input intervention leaves old-memory/current-probe comparator unchanged by construction.
- The native query carries an identity cue and is a separate phase from inserted blanks and the visual probe.

Stage/head attention mass and value RMS are descriptive; high mass or activity is not evidence that orientation/duration content was preserved. Full confusion matrices, source summaries and component margins are in [results.json](results.json). The separate [saved motion audit](../MotionAudit/report.md) examines historical task allocation, class biases and a validation-only calibration; it is not a new main-model training run.

## Learned head behavior and motion logit contributions

The two heads were allowed to attend to both sources. In baseline delayed orientation, the second head allocates38.4% of attention mass to memory during sample presentation,77.9% during blanks,77.2% during the identity query and40.5% during the probe. The first head remains around12% memory mass across these phases. These are descriptive averages, consistent with learned source specialization; they do not by themselves demonstrate preservation of feature content. Value-contribution RMS is measured per head before the learned output projection.

For immediate motion, the attention model's sensory branch contributes an average+1.050logits to up minus down, compared with+0.421in the continuation control. Its memory branch contributes+0.572, comparator−0.247 and final classifier bias+0.050. Thus the class preference is not simply a large learned classifier-bias parameter: it is distributed across projections of realized sensory and memory features.

After24blanks, the sensory branch is nearly constant across episodes (centered per-class SD0.00013–0.00023), but it favors down over up by0.238logits. The up preference at this delay instead comes from the memory branch(+0.743), comparator(+0.258) and classifier bias(+0.050). Therefore a simple claim that the blank sensory branch causes the delayed up bias would be wrong. This is an exact decomposition of the current decision, not a unique account of what training produced the offsets.

[Per-class branch means and between-episode SD](branch_statistics.json). All original weights remain frozen and their checkpoint hashes are unchanged. The single GPU diagnostic worker exited after212.8seconds. No further run or feature-probe extraction was launched.
