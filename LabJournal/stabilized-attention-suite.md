# Stabilized attention: joint task suite

## Frozen suite analysis requested and running — October8

The user requests psychometrics across all13tasks, attention manipulation in Krauzlis and sign-cued orientation, and per-timestep spatial projections for all tasks. The follow-up explicitly asks for both target stimulation and distractor inhibition to improve target responses. Analysis/run01 uses frozen validation-selected best54808, no optimizer, new trial namespaces and one finite14400second local cap. Prospective125696 inference presentations include33568psychometric,83168paired intervention and8960atlas presentations. Native renderers match bitwise at original settings; a zero intervention produces identical logits. One actual CLS attention query over100patches is projected honestly; no full NxN matrix is fabricated. Photo exemplars are paired with their own maps; condition averages are separate. Runtime protocol and live receipts are under SecondPass/StabilizedAttentionSuite/Analysis/run01. No learning or biological-equivalence claim follows from a map manipulation. Final results require completion and independent audit.

October 7, 2026. User requested the smallest-change adaptation of the successful stabilized Krauzlis model to the entire task suite. One fresh model, unchanged shared sensory/attention/memory/local-comparison backbone, 13 task-selected output layers, 381,264 parameters. All 13 tasks and 35 native conditions, existing photo splits and difficulty ranges. This is joint multi-task learning, not a held-out-task transfer claim.

Adam3e-4, full-BPTT FP32 batch32/micro4, gradient clipping100 and CE+.05 pixel prediction remain unchanged. Equal-task and within-task-condition balanced fresh sampling follows the established suite scheduler; the old single-task 1,000-movie/two-epoch replay is not retained. Fresh199xxx streams and all fresh weights; old checkpoints preserved.

One eight-hour local allocation starts with native35-condition profiling, including all evaluation. Requested17160 updates, prospectively reduced to whole13-update cycles if necessary. Full validation130/every520/terminal; Krauzlis monitor every130; final11192 trials. Best selected by equal-task AUC then chance-normalized BA, excluding empty-list recognition from selection. No automatic extension or cloud.

CPU checks passed exact original Krauzlis forward equivalence, native stream replay, isolated output heads and13-task backward/Adam updates with80 finite states. GPU activation and measured target require LocalRuntime/run01 runtime receipts. See [implementation and protocol](../SecondPass/StabilizedAttentionSuite/README.md).

## Actual activation

Production independently verified: StabilizedAttentionSuite worker41092/supervisor43836/guard13444; saved13/live40, all80 finite shape-aligned Adam states CPU-reloaded. Shared tensors have13updates and each task head1; all13tasks updated. Frozen source hashes and original Krauzlis best/latest hashes match. Profile pinned13702updates/438464fresh presentations/1054updates per task. First full validation130, then every520 and terminal; Krauzlis monitoring every130. Deadline4:19:41PM PDT October7, no automatic extension. No held-out learning result yet. Receipt SecondPass/StabilizedAttentionSuite/LocalRuntime/run01/independent_production_verified.json.

All35 profile conditions completed on the local GPU before fresh production. The planned test counts are unchanged. This launch is not evidence of successful learning.

## Historical comparison anchor

The retained catalog identifies validation28539 from the fresh-origin SpatialReadout continuation: all seven sensory tasks, ring, cued orientation and binding100%; recognition73.09%, motion duration26.17% (four-class chance25%), Krauzlis50%. This is a historical validation snapshot, not a matched final test or proof the later target completed. See EXPERIMENT_CATALOG.md and RESEARCH_HISTORY.md. The new per-task report should preserve this distinction and compare training exposure explicitly.

## Explicit continuation authorization — October7

User approved doubling full-suite exposure. Resume StabilizedAttentionSuite run01/latest13702 with exact model/Adam/scheduler/native-stream/allRNG state; add13702 to cumulative27404 (2108updates/task,876928presentations). Architecture/loss/Adam3e-4/clipping100/batch32micro4 unchanged. Separate LocalRuntime/continuation01; original best13000/latest13702/results preserved. Original best stays validation-eligible. One new finite28800second LOCALcap at launch, no cloud/automatic extension. First full validation13832, global520cadence/terminal; Krauzlis monitoring global130cadence; final11192 fresh test namespace199104. CPU restoration and native-next-draw equality passed, both original checkpoints have80finite aligned Adamstates. Actual activation requires new runtime receipts. See SecondPass/StabilizedAttentionSuite/Continuation/README.md.


Continuation independently verified: worker34492/supervisor17136/guard34524; saved13715/live13729. All80 parameter tensors advanced from latest13702; all80 finite aligned Adam states, each head1055updates after13newupdates. Exact restore and original/checkpoint/source hashes verified. Target27404 cumulative, first full validation13832. Deadline1:42:10AM PDT October8, no automatic extension. Evidence StabilizedAttentionSuite/LocalRuntime/continuation01/independent_production_verified.json.



## Pause and authorized extension — October8

Paused cleanly at24457 on user request; best24440 meanAUC0.8795554553. Both saved checkpoints have80finite aligned Adamstates, zero unsaved updates; original worker/supervisor/guard exited. No continuation01 final test. User now explicitly requests resume and double the planned training length: new cumulative54808,30351 further updates from latest24457. New continuation02 preserves all previous artifacts. Same model/optimizer/native sampling and validation criterion; exact CPU restore/next-native-draw checks passed. One finite57600second LOCALcap, no automatic extension/cloud. First validation24587, global520/terminal; Krauzlis global130. Final11192 still-unused199104test stream. See Extension/README.md and continuation02/protocol.json; activation requires receipts.

## Latest authorized extension and actual activation — October8

User explicitly resumed the paused suite and doubled the prior planned cumulative27404 target to54808. New LocalRuntime/continuation02 resumes continuation01/latest24457 exactly;30351 further updates,1753856 cumulative presentations,4216updates/task at endpoint. Prior paused latest24457/best24440(AUC0.8795554553), source and results preserved. Extension adapter changes only initial step/target/local cap/parent verification; model, optimizer, loss, clipping100, native streams and batch unchanged. CPU restore/next native draw and CUDA state restore exact. Worker59836/supervisor6052/guard23572 independently live-verified; saved24470/live24545,80finite aligned Adamstates. Saved13-update scheduler sequence exactly continues paused partial cycle;74parameter tensors advanced (three heads not scheduled in this partial13-update interval). No optimizer reset. First full validation24587; global520/terminal, Krauzlis global130. Final11192 unused199104stream. New finite57600s LOCALcap through10:49:45PM PDT October8, no automatic extension/cloud. Evidence SecondPass/StabilizedAttentionSuite/LocalRuntime/continuation02/independent_production_verified.json. Earlier active descriptions below are historical.


## Latest result — full-suite extension completed, October8

StabilizedAttentionSuite continuation02 completed54808 cumulative updates/1753856presentations/4216updates per task at7:03PM PDT October8. Best/latest54808 both independently CPU verified with80finite aligned Adamstates; all35 final test cells/11192 trials complete, namespace199104. Final equal-task meanAUC0.981764; BA cued orientation0.805664, contour0.890625, Krauzlis0.898225, recognition0.898872, motion duration0.972656, spatial frequency0.996094; remaining7tasks1.0. Validation selected54808, meanAUC0.979576. Worker59836/supervisor6052/guard23572 exited normally; GPU lock free. Frozen source and original checkpoints preserved. No additional training/automatic extension queued. Evidence SecondPass/StabilizedAttentionSuite/LocalRuntime/continuation02/independent_completion_verified.json and final_test.json. Earlier active descriptions below are historical.



## Frozen full-suite analysis completed — October 8

The selected suite checkpoint54808 was analyzed without training in SecondPass/StabilizedAttentionSuite/Analysis/run01. All125696 presentations completed:33568 psychometrics,83168 paired interventions (39Krauzlis arms/41signed-orientation arms),8960 atlas trials across35nativeconditions. Independent checks passed identical paired pixels, global-shift and zero-delay controls, exact exported sprite pixels, recognition frame hashes, normalized maps, unchanged model tensors/checkpoints/frozen source, worker exit0 and freeGPUlock. The atlas contains77groups/1376group timesteps,13GIFs and12PNG/PDF figures. Interactive report: Analysis/run01/gallery/index.html; receipts completion_verified.json and browser_verified.json.

Both requested benefit directions were tested. None of8primary one-sided benefit tests established a balanced-accuracy gain afterHolm correction. Near-threshold Krauzlis target+4/+8 gave+0.35/+0.87pp (intervals cross0); distractor-4/-8 changed no choices. Orientation distractor inhibition was nearzero; target+4/+8 reducedBA16.2/20.7pp. Strong costs occur in both tasks: Krauzlis trained-angleBA85.6%→33.5% withtarget-8,24.7% withdistractor+8. SelectedKrauzlis postchangecontrolmap96.7%target/.06%foil suggests limited suppression headroom, not proof of the mechanism. Orientation15degree psychometricBA50.8% withoutblankdelay vs87.5% with4blanks motivates controlled timing follow-up. These are computational priority-logit interventions, not biological microstimulation equivalence or training-seed replication. OneCLS query over100patches; no inventedNxNmatrix. No newtraining/cloud/automaticextension queued.

