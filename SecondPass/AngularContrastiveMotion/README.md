# Angular contrastive motion encoder

The completed fresh CNN run learned a direction-similarity representation that
achieved 100% balanced accuracy and AUC 1.000 in all six synthetic change-test cells.
It trained 25,024 updates on 391,000 unique pairs/782,000 replay presentations in
3 hours, 28 minutes, 33 seconds, within its eight-hour local cap. Validation selected checkpoint 19000;
latest 25024 is also retained. The selected model alone received the final test.
See the [detailed technical report](TECHNICAL_REPORT.md) and [final results](FINAL_REPORT.md).

Three ordered 100×100 RGB frames become nine input channels. A shared residual CNN
with 32/64/128/256 channels produces a 256×13×13 field, followed by global spatial
mean, Linear 256→128/GELU/Linear 128→128 and unit normalization. All 4,738,528 parameters
in 64 tensors train from scratch. Architecture code is reused; trained weights,
VAE/predictive objectives and classifier heads are not. Source: [model.py](model.py),
[dataset.py](dataset.py), [trainer](train.py).

The supervised pair objective is `0.5*(d-Delta/pi)^2`, where `Delta` is the simulator's
shortest circular direction difference and `d=sqrt(||za-zb||²+1e-8)-1e-4`.
Same-direction clips have independent dot layouts/counts/speeds. The target mixture
is approximately one-third same direction, one-third 26°/28°, and one-third 45°–180°.
At a fixed embedding separation, the outward spring force grows linearly with
angular target; it is not exactly angle-proportional at every separation. The
technical report derives the actual smoothed gradient and collapse limitation.

One MPS worker used FP32, two CPU threads, Adam 1e-4/no clipping and 32 pairs per
update, accumulated in microbatches of 4 pairs. Each 1,000-pair pool received two
shuffled epochs before replacement. A validation-fitted scalar embedding-distance
threshold supplies change/no-change decisions; no FFN was trained.

The final angular set contains 384 independent generated pairs. The operational
set contains 3,072 **presentations in 1,536 paired nuisance contexts**, not 3,072 fully
independent trials: adjacent opposite-label examples share nuisance variables and
the complete before clip. Test threshold 0.08698047697544098 was fixed on validation
for the selected model. Fresh test examples use the same synthetic renderer and
trained speed/angle conditions. Persistent coherent full-field periodic dots have
no cue or distractor; these results do not establish native Krauzlis performance,
attention or repeat-seed generalization.

Evidence: [run report](LocalRuntime/run/report.json),
[selected validation](LocalRuntime/run/validation_019000.json),
[configuration](LocalRuntime/run/config.json), [budget](LocalRuntime/run/budget.json),
[first persisted update](LocalRuntime/production_verified.json), and
[completion verification](LocalRuntime/completion_verified.json).
The [saved supervisor](LocalRuntime/run/local_supervisor_result.json) and
[guard](LocalRuntime/run/guard_result.json) report normal completion. No automatic
follow-up or new compute allowance follows from this completed experiment.
