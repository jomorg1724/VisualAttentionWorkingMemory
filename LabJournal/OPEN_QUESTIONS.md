# Open questions after the contrastive result

[Index](README.md) · [Latest report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md) · [History](RESEARCH_HISTORY.md)

| Question | What is known | What remains unmeasured |
|---|---|---|
| Does this encoder understand native Krauzlis motion? | It solves a six-frame persistent-dot comparison at 0.375/1/2 px per frame and 26°/28° changes | Finite dot lifetimes, apertures, native stimulus durations, cues and competing stimuli |
| Can motion supervision supply a useful component of the original recurrent task? | A fresh angularly supervised CNN has an operational distance readout | Transfer with real cue selection and a learned sequence decision; no such training has started |
| Why did prediction help reconstruction but not the frozen FFN? | Next-frame loss improved strongly; concatenated predictive means stayed at chance | Direction accessibility, nuisance sensitivity and the effects of objective, pooling, initialization and readout optimization |
| Is contrastive supervision the isolated reason for success? | The latest method differs in supervision, architecture/pooling, pair nuisance distribution and decision rule | A matched ablation separating those changes; no causal isolation is claimed |
| How precise is the learned angular geometry? | One held-out angular test has correlation 0.987835 and loss 0.00136432 | Psychometric precision at smaller changes, other speeds/noise and repeated seeds |
| Is the comparison result robust? | All 3,072 held-out presentations were correct under a validation-selected threshold | Independent-context and seed uncertainty; the test has 1,536 paired contexts, not 3,072 independent samples |
| Can spatial binding and working memory use this motion code? | Earlier memory/attention models had mixed successes and failures | A justified new integration architecture and its trained performance |

These are research directions, not a queue or training authorization. No new run, architecture sweep, cloud rental or diagnostic is initiated by this document. Earlier hypotheses and their dated revisions remain in the [archived open-questions page](archive/OPEN_QUESTIONS_through_20261004.md) and the individual experiment records.
