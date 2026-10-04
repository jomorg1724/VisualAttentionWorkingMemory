# What is lacking in terminal6760?

**Failure is already present without a retention delay. Spatial cue location affects outputs, but the correct cue–evidence relationship is mostly not expressed in the decision. Longer delay adds stereotyped responses; it is not the sole failure.**

Frozen checkpoint SHA256 `1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841`; terminal step6760 verified against predecessor report. Exploratory n=32 each, one checkpoint. New native test seeds; zero training, zero probe fits.

## Native matched-evidence delay comparison

| Task | D0 BA / AUC | D24 BA / AUC | D24 prediction counts |
|---|---|---|---|
| Signed spatial orientation |50.0% / .523|50.0% / .535|[0,32]|
| Retrocued spatial binding |46.875% / .480|50.0% / .555|[16,16]|
| Cued motion duration |25.0% / .544|25.0% / .490|[0,32,0,0]|

Exact nonblank rasters and labels are identical between D0 and D24; only native ignore blanks differ. On independent native two-frame draws, orientation is84.375% BA/AUC1.0 and motion direction is100%/1.0. Thus this is not a universal inability to discriminate orientation or motion, but those anchors have different geometry and requirements and do not isolate four-patch sensory encoding.

## Matched cue and evidence counterfactuals at D0

The cue always retains a valid report interpretation; labels are recomputed. No cue blanking was used. Retargeting deliberately chooses a location with a different correct answer where possible, so these are controlled, native-valid scenes, **not a natural-frequency sample or a valid-versus-invalid cue benefit**.

|Paired edit|Correct labels changed|Choices changed|Mean absolute probability change per class|Both pair members correct among label-changed pairs|
|---|---:|---:|---:|---:|
|Orientation cue location|32/32|13/32|.06572|6/32|
|Orientation cue sign|29/32|2/32|.01725|0/29|
|Orientation rotation signs reversed, cue fixed|29/32|2/32|.01341|2/29|
|Binding retrocue location|32/32|15/32|.02386|8/32|
|Binding exchanged pair changed, cue fixed|32/32|6/32|.01084|4/32|
|Motion cue location|31/32|23/32|.02469|2/31|
|Motion temporal reversal (OOD)|32/32|1/32|.00678|0/32|

- **Orientation:** location sensitivity is substantially larger than sign or rotation-evidence sensitivity. Flipping the instruction sign should reverse the answer on29 trials, but only2 choices change and no label-changing pair is correct in both conditions. The mean new-versus-old correct-class logit-margin shift is−.00093; only14/29 shift in the task-appropriate direction. This points to missing use of the sign–rotation conjunction, not simply failure to notice a cue.
- **Binding:** both retrocue and exchanged-pair edits produce nonzero logit changes, but they are weakly task-aligned. The mean correct-class margin shift after retargeting is−.01046 (15/32 positive), and after evidence change +.00583 (16/32 positive). This does not establish competent comparison of location-specific remembered orientations.
- **Motion duration:** retargeting strongly changes choices, but it does not recover the appropriate count winner: BA24.583%, with predictions still confined to up/left. Only2/31 answer-changing cue pairs are both correct. Reversing reference+eight motion frames changes one choice while all32 count-winner labels reverse; probabilities change only.00678 on average. **This reversal is OOD** because backward dot replacement is not the native stochastic process; it supports weak observed sequence-direction dependence under this perturbation, not a causal proof that temporal evidence is absent.
- **Spatial response bias is directly visible:** in native motionD0, all8 trials at location0 and all8 at location3 predict left; all8 at location1 and all8 at location2 predict up, irrespective of the balanced correct direction. BindingD24 likewise predicts changed for all trials at locations0/3 and unchanged for all at1/2 (8 each). These within-draw perfect location–choice mappings are a more specific failure signature than aggregate chance performance; they do not establish the model's internal mechanism.
- **Delay:** atD24 orientation always says aligned and motion always says up. Binding remains variable across query locations but constant within each location in these draws. D0 is already at chance, so fixing retention alone is not supported as a sufficient remedy.

## What this supports—and does not

The highest-priority missing behavioral capability is **using the cue's meaning to select and combine the correct evidence**, rather than responding to cue location while weakly using sign, orientation comparison, or motion history. Acquisition, spatial sensory resolution, credit assignment, relational comparison and readout/criterion remain competing explanations. No fitting or internal perturbation separated them. Do not infer complete cue blindness, absent internal information, a uniquely faulty memory module, or a proven architectural remedy.

Small-sample uncertainty is large: native binary50% accuracy has a95% Wilson interval33.63–66.37%; native motion25% has13.25–42.11%. All paired accuracy-difference bootstrap intervals include zero. The defensible result is the measured pattern of paired response/logit dependence, not a claim of significant accuracy improvement. Repeated conditions share base episodes and are not independent extra samples. 15 conditions use160 unique base episodes (96 across the three focused tasks,64 sensory anchors), not480 independent trials.

Frame timings are renderer metadata, not guessed: orientation cue[0,1,2], sample[1,2], probe3+D; binding sample[1,2], retrocue3+D, probe4+D; motion cue[0..9], reference1, moving[2..9], report10+D. Stack3 exposes old frames during the first two nominal blanks. No task-padding, model changes, or metadata-to-model inputs.

## Verification and artifacts

- All15 conditions completed with32 observations each. Unmodified recurrent wrapper matches direct model exactly (max absolute logit error0 for every condition's first2 trials).
- `eval()`, all parameters `requires_grad=False`, inference mode. Every model state tensor is bitwise unchanged; checkpoint SHA and five relevant source hashes unchanged. Predecessor supervisor exit0 and worker/supervisor absence were checked before starting. One local MPS worker; CPU intra-op2/inter-op1 and BLAS thread environment2.
- `diagnostic.py`: reproducible generator/interventions and bounded inference. `renderer_inspection.json`: metadata and exact-renderer/paired-delay checks. `trial_results.json`: every label, logit and metadata record. `summary.json`: metrics, confusion, paired effects and immutable-source verification. `strata_and_margin_uncertainty.json`: target/sign strata and paired margin intervals. `budget.json`, `completion.json`, `report_verification.json`: timing/exit receipts. `summarize.py`: this CPU-only report.
- Full cue-validity psychometrics, spatial allocation, inhibition, microstimulation, reaction times, recognition/Krauzlis/ring diagnostics and a full-suite reevaluation are **not measured** here.
