# Correspondence Predictive Memory — October 5, 2026

User requested combining local correspondence, transport prediction and
appearance-invariant transformation learning in a neuroscience-inspired model
as the next moving-dot experiment. Implementation and detailed equations:
`SecondPass/CorrespondencePredictiveMemory/README.md`.

432,924 fresh learned parameters,43 tensors. Fixed local normalized pixel
correlations supply441 displacement candidates at16 spatial sites. Learned
population refinement, causal image transport, width64 recurrent attention,
spatial ConvGRU and CLS memory, nonlinear change decision. No true motion-angle
or flow target. Independent dot layouts with shared dynamics provide synthetic
positive pairs for VICReg-inspired transformation regularization. This pairing
and the fixed visual correspondence operation are explicit inductive priors.

Scoped six-frame continuous-dot task remains speeds1.5/4/8 and angles26/28.
Fresh194xxx streams; both appearance views participate in choice/prediction
training. Target25,024 updates,1000 dynamics pairs x2 epochs, batch32 pairs,
Adam3e-4 FP32. One prospective14,400-second local allocation from first profile,
no extension/cloud/extra arm. Validate384; selected final3072; separate768-trial
transfer test at unseen speeds0.75/3/6 and angles18/42. Best/latest only.

CPU behavior and real two-update production-lifecycle fixtures passed. Fixed
pixel-correspondence diagnostic achieved97.92% BA/AUC0.99590 on384 independent
trials, threshold chosen on192 validation trials. This is not a trained-model
result. Earlier96.35%/0.98831 diagnostic receipt is retained; samples changed when
the transfer namespace was added, with no test-directed model tuning.

At5:55PM PDT queued process15136 after OutcomeContrastiveRViT/run01, which was
still training22354/25024. Latest predecessor validation20000 BA50.26%, AUC0.50434.
Queue budget has not started at enqueue. It waits for verified normal completion,
owned worker/supervisor exit, released shared GPU lock and unchanged source.
Actual run receipts are in `SecondPass/CorrespondencePredictiveMemory/LocalRuntime/run01`.

## User-directed immediate handoff,6:01PM PDT

User requested killing the old run and starting this variant immediately. Stopped
waiting queue15136, then oldworker16140. Old supervisor and guard exited; shared
GPU lock released. Preserved oldbest5000/latest24750, both independently CPU
verified with36 finite aligned Adam states. Code124 in the old supervisor is the
expected user cancellation, not a spontaneous failure. No final old-model test
was performed. New supervisor14972 started with the unchanged frozen source;
profile/production progress must be read from runtime receipts.

Actual fresh production verified6:02PM PDT: worker37408, savedstep64, all43 finite aligned Adam states independently CPU-reloaded; nativeprofile pins25024 updates. Budget began6:01:32PM PDT and ends10:01:32PM PDT. Runtime root_startup_verification.json.
