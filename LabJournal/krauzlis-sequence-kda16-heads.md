**Completed2026-10-03UTC:**4216updates/134912fresh movies; selected2108 and
terminal4216 both50%BAall3nativeconditions, meanfreshtestAUC0.476345/0.447496.
45artifactsverified and bothcheckpoint27AdamstatesCPUreloaded; poddeleted
afterretrieval. Tasknotacquired. Userrequestedreplacingitscloudslot with
[the random-frame-gradient candidate](krauzlis-random-frame-gradients.md).

# Krauzlis: sixteen memory heads in one KDA layer

**Fastervalidation snapshot:** CPU-only frozencheckpoint400 completed100validationtrials/condition; B12/B20/B28 BA50%each,meanAUC0.509996. Weights/checkpointunchanged; noGPUworker,optimizer/streams/selectionchange. Thisisvalidationonly, notfinaltestperformance. [Snapshot](../SecondPass/SequenceKDA16/CloudRuntime/early_validation/latest.json).

**PRODUCTION VERIFIED.** NativeA40profilepinsfull4216/134912exposure; productioncheckpoint3/96episodes CPUreloadedwithall27Adamstatesadvanced/allparameterschanged, freshwholemodel/streams/RNG and no profiletransfer. Profilepeak12,662,468,096bytes. Newpodpa0ko8f2qirisy,NEW8h/$5cap20:33:14UTC→04:33:14UTC,independentguard/mirroractive. Validation/finals pending. LocalCNN-GRUcontinues,RViTuntrained. [Evidence](../SecondPass/SequenceKDA16/RUN_STATUS.md).

**Immediate launch supersedes waiting queue.** Usercancelledthree-layerpod5hnvb87npqpqb4 at2615updates andrequestednextqueuedmodelstart. Allcancelledartifactsverifiedandoldpoddeletedfirst. NewA40podpa0ko8f2qirisy,own8h/$5cap20:33:14UTC→04:33:14UTC; nativeprofile/pin underway,allfreshweights. [Currentstatus](../SecondPass/SequenceKDA16/RUN_STATUS.md). Originalqueuedesignbelowishistorical.

## Question and decision

After the original two-head single-layer model finished at50% balanced accuracy
in all three native conditions, the user requested one KDA with sixteen heads.
The ongoing three-layer depth comparison and delayed-frame CNN–GRU continue;
this new model is queued after verified retrieval and deletion of the cloud run,
with its own explicitly approved eight-hour/$5 allowance.

## Architecture

[Implementation](../SecondPass/SequenceKDA16/model.py) retains100x100RGB inputs,
chronological10x10pixel patches, shared128-wide patch projection, spatial/time
positions and a terminalCLS-only128→256→2 decoder. There is exactly one KDA.
Its sixteen64-dimensional key/value heads have separate learned projections;
internal q/k/v/decay width1,024 projects back to128. Eight initial decay horizons
are32frames and eight128frames, with all projections and gates trainable.
737,170parameters/27tensors and65,536FP32 recurrent-state values/trial versus
158,340parameters and8,192state values in the original model.

This changes head count and capacity together. It is not a parameter-matched
head-count intervention, and it does not add a second processing stage: the
single-layer terminalCLS query still has no earlier layer to make it depend on
the observed movie. Results will test this larger single-layer package, without
assuming that more heads must solve cue-dependent motion selection.

## Training and queue

All model/Adam/RNG/task-stream state starts fresh; native teaching and the
original ordered token protocol are unchanged. Adam1e-4/no clipping, effective32
micro4, full BPTT and FP32 including backend intermediates. Target4,216updates;
actual exposure is fixed before production from disposable native A40 profiling.
The larger head state/intermediates require measured memory and timing feasibility.
No profiling state enters production. Validation selection uses three-condition
mean AUC, then BA, then earlier checkpoint; final selected/terminal tests are
paired on200fresh trials per condition, including target/foil/catch reporting.

The detached parent queue63567/PPID1 waits for exact pod5hnvb87npqpqb4's completed
artifact retrieval, both verified final checkpoint reloads and confirmed deletion,
then checks no active provider pods before a one-shot new rental. Queue waiting
spends none of the new cap. Rental creation starts28,800seconds/$5 including
setup/profile/evaluation/retrieval. The known first-SSH-timeout startup issue is
fixed by retrying probes within the existing240second readiness window.
Independent guard, mirror, verified retrieval and ephemeral-pod deletion remain
in the frozen runtime. Failed predecessor cleanup blocks new rental. A separate bounded local wake
assertion supports automatic launch/retrieval and ends on completion/failure or
the new absolute deadline; it performs no provider operations.

## Evidence and present limits

Two focused model checks verify short operator output/state/gradient parity,
early-frame gradients, ordered tokens, all16head gradients and all27parameters
advancing. Two worker/source-bundle checks verify fresh construction/empty Adam,
stream/RNG identity, native seeded schedule and source-only backend closure.
Three parent checks verify exact cleanup gating, failed-predecessor refusal and
retry of a slow firstSSH probe. No exhaustive nativeCPUcampaign was run.

**Queued, not trained.** No new GPU has been rented and no production optimizer
progress exists. [Run status and immutable receipts](../SecondPass/SequenceKDA16/RUN_STATUS.md)
track the transition from waiting to actual training. Poor live scores are not a
stop condition. Preserve all earlier models; report measured exposure and
hardware alongside final per-condition results when they exist.
