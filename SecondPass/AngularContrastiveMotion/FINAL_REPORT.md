# Angular contrastive motion: final result

The fresh 4,738,528-parameter/64-tensor CNN completed 25,024 updates, 391,000 unique
training pairs and 782,000 pair presentations on October 4, 2026 at 12:06:29 PDT.
Runtime was 3 hours, 28 minutes, 33 seconds within an eight-hour local cap. Stop reason was
`planned_complete`. Validation selected checkpoint 19000; latest 25024 was retained.
The completed run used no pretrained weights or learned classification head.

The selected model's fixed validation threshold,
**0.08698047697544098**, separated every change/no-change test presentation correctly
in all six speed/angle cells. This final evaluation belongs to selected 19000;
terminal 25024 was validated but not separately final-tested.

| Speed, pixels/frame | Change, degrees | Presentations | Paired nuisance contexts | BA | AUC |
|---:|---:|---:|---:|---:|---:|
| 0.375 | 26 | 512 | 256 | 100% | 1.000 |
| 0.375 | 28 | 512 | 256 | 100% | 1.000 |
| 1.0 | 26 | 512 | 256 | 100% | 1.000 |
| 1.0 | 28 | 512 | 256 | 100% | 1.000 |
| 2.0 | 26 | 512 | 256 | 100% | 1.000 |
| 2.0 | 28 | 512 | 256 | 100% | 1.000 |
| **Total/mean** | | **3,072** | **1,536** | **100%** | **1.000** |

Each cell has 256 presentations of each label. Adjacent opposite-label examples
share initial dots, speed/base direction, change sign and the complete before
clip; their first after frame is also identical. The 3,072 presentations therefore
represent 1,536 paired nuisance contexts, not 3,072 fully independent trials.
No confidence interval or independent-trial uncertainty claim is made.

| Independent angular-test metric | Result |
|---|---:|
| Generated pairs | 384 |
| Angular spring loss | 0.0013643201 |
| Collapsed-distance baseline loss | 0.0799425021 |
| Distance/normalized-angle correlation | 0.9878348708 |
| Mean same-direction distance | 0.0058306991 |

Mean loss over the final 100 training updates was 0.0012886712. The angular baseline
assigns zero distance to every pair; it is not a separately trained model.
The selected validation angular loss was 0.0012566851, with mean operational AUC/BA 1.0.
Selection maximized validation mean AUC then minimized angular loss. The threshold
was fitted exclusively on validation and carried unchanged into the test.

This result demonstrates useful motion-direction similarity under **explicit
simulator-angle supervision** on the simplified dot setting. Dots are persistent,
coherent, full-field and periodic, with one constant direction per three-frame
clip. All tested speeds and change magnitudes occurred in training. There is no
native cue/distractor/Krauzlis result, held-out-renderer test or repeat-seed
replication. The earlier frozen predictive-encoder FFN experiment differs in
objective and training policy, so its failure does not isolate a causal reason
for this run's success.

Read the [technical report](TECHNICAL_REPORT.md) for architecture, exact force,
sampling and evaluation dependence. Evidence:
[final report JSON](LocalRuntime/run/report.json),
[selected validation 19000](LocalRuntime/run/validation_019000.json),
[terminal validation 25024](LocalRuntime/run/validation_025024.json),
[progress](LocalRuntime/run/progress.jsonl),
[configuration](LocalRuntime/run/config.json), [budget](LocalRuntime/run/budget.json),
[production proof](LocalRuntime/production_verified.json),
[completion verification](LocalRuntime/completion_verified.json),
[supervisor completion](LocalRuntime/run/local_supervisor_result.json), and
[normal guard exit](LocalRuntime/run/guard_result.json).
The completion receipt records best 19000/latest 25024 and 64 Adam states each.
No further run, checkpoint deletion or compute renewal was performed for this report.
