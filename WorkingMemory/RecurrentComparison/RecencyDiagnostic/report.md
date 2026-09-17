# Winning evidence later helps the E/I model more

The frozen E/I model benefited when the winning direction occurred later, despite identical direction totals and identical final two visual evidence frames. The LSTM showed no clear corresponding benefit. This supports greater favorable sensitivity to late winning evidence in E/I on these controlled sequences.

| Selected step5000 model | Early BA | Late BA | Late − early, percentage points (paired95% CI) |
|---|---:|---:|---:|
| LSTM |85.74% (439/512)|84.18% (431/512)|−1.56 [−4.88,+1.56]|
| E/I adaptive |72.27% (370/512)|78.52% (402/512)|+6.25 [+2.73,+9.58]|

The E/I-minus-LSTM difference in those ordering effects was **+7.81 percentage points**, CI[+3.90,+11.53]. Both models received exactly the same movies. This is a paired interaction, not a subtraction of separate confidence intervals.

True-class probability changed by−0.00994 in LSTM (CI[−0.03184,+0.01200]) and+0.04666 in E/I (CI[+0.02917,+0.06295]); the interaction was+0.05660 (CI[+0.03618,+0.07578]). The true-winner logit minus the strongest competing model logit changed by−0.146 in LSTM (CI[−0.338,+0.037]) and+0.410 in E/I (CI[+0.284,+0.530]). These logit scales are model-specific; they are not duration margins or directly comparable cross-model confidence units.

Exactly512 matched render groups were evaluated:128 per winning direction, with256 ending in the winner and256 ending in another direction. There were1024 movies,2048 model presentations and22,528 logical frame presentations, but only512 paired units for uncertainty. The eight-transition length and visible instruction/cue format were trained conditions. For every pair we matched the entire direction-count vector, unique winner, count margin, first direction, last two directions and total switch count. Every winning-evidence center moved later, averaging1.010 transition positions (range0.25–2.00); “early” and “late” describe that relative movement, not necessarily opposite halves of the movie.

All4^8 direction schedules were enumerated before inference. The available canonical matching bank contained909 final-not-winner and722 final-winner groups. A fixed random draw selected64 of each and rotated these128 templates over four direction identities, giving512 distinct ordered schedule pairs and1024 distinct individual schedules. Each group had a distinct fresh render seed. The full stream state was restored before rendering each variant; physically continuous dot movies were generated rather than shuffled. Exact tensor checks confirmed identical instruction, last two evidence rasters and final report within each pair, and full-movie hashes matched across model evaluations.

Descriptively, E/I's late advantage was+3.52 points when the final direction was not the winner and+8.98 points when it was (256 groups each). Its largest count-margin stratum effect occurred at a one-transition winner advantage:57.38%→68.44% (244 groups). LSTM in that stratum was79.10%→77.05%. These secondary strata were not separately powered or multiplicity-adjusted; count-margin/switch-count tables with denominators are in `summary.json`.

Intervals use2000 bootstrap draws of complete paired render groups, stratified by winner and final-winner status. They condition on the selected ordering bank. Directional rotations of128 templates are not512 independent order templates; these intervals do not include uncertainty over other possible schedule banks, training seeds or checkpoints.

The controlled ordering distribution differs from the original random held-out distribution. Moving winning evidence also changes distractor placement and local run structure, even with global switches matched. Thus this result does not uniquely establish leak, adaptation, saturation or a particular circuit cause, nor does it decompose the original10.74-point LSTM–E/I gap. The LSTM remains more accurate under both orderings. A useful next development question is how to reduce E/I's temporal-order sensitivity while preserving its learned sensory and memory performance; no new training or lesion was performed or is launched here.

Execution completed in71.95seconds inside the separate600-second allowance, with zero optimizer steps. Both models were frozen and evaluated sequentially in fp32 on the local RTX3070 Laptop, Torch1.13.1+cu117, two CPU threads, TF32 disabled. The first production microbatch was also the accounted inference profile. Peak allocated tensors were115.54MB for LSTM and114.36MB for E/I, excluding CUDA context/desktop usage. The exact child terminated successfully; the deleted cloud pod remained deleted. See `exit.json` and `completion_receipt.json` for measured execution and final artifact identities.

Reproduction details and limitations are in `README.md`; checkpoint/source identities are in `config.json`, schedules in `pairs.json`, and all logits/metadata/movie hashes in `predictions.jsonl`. One independent read-only review checked actual source and interpretation; its response is preserved in `decision_review.md`.
