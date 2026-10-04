# Local delayed-frame CNN–GRU training

**2026-10-03 01:14:47UTC: CANCELLED and saved at the user's request.**
Terminal2,575updates /82,400fresh trials saved with Adam state; worker55539 and
supervisor54824 exited normally afterSIGTERM, guard/owner launchd jobs removed.
No worker failure or finaltest claim. Artifacts remain at the original runtime;
fresh StructuredMotionRViT replaces this local accelerator worker without
inheriting its state. [Cancellation receipt](LocalRuntime/user_cancellation_verified.json).
Earlier running notes below are historical.

**Training running and independently verified,2026-10-02.** AppleM4Max36GiB/MPS, one local accelerator worker, twoCPUthreads. Source-isolated runtime: `/Users/jonathanmorgan/VAWMRuntime/delayed_frame_gru_local01`; launchd owner`org.vawm.delayed-frame-gru-local01` supervisor54824/PPID1, trainingworker55539, independentguard54822. Cloudthree-layerKDApod`5hnvb87npqpqb4` continues with its own unchanged deadline and mirror.

Exactly the requested architecture: independent current/previous residualCNN branches, all24convolutions stride1/full100x100/no pooling, learned spatial readouts256 each, orderedconcat512→standardGRU256→binaryMLP. All21,293,770parameters train from fresh construction, with no inherited model/Adam/RNG/stream/profile progress. NativeB12/B20/B28,26/28degree changes,cues,rendering,labels and57/29/14target/foil/catch proportions unchanged. [Design/model](README.md).

MPS memory motivated microbatch1 selected before profiling, retaining effectivebatch32,Adam1e-4,betas.9/.999,eps1e-8,weightdecay0,no clipping,strictFP32 and fulltemporalgradients. Profiling discarded3nativewarmupupdates and measured6steadyupdates, plus20evaltrials/condition. All profile state was discarded before production. Prospectivepin **2,631updates /84,192episodes**, reduced from4,216 to fit measured local costs. Each nativecondition receives877updates/28,064episodes. This has lower exposure and different platform/microbatch than the cloudKDA comparisons; no matched-exposure superiority claim is warranted.

Cap begins **2026-10-02T18:57:51.068054+00:00** and hardends **2026-10-03T02:57:51.068054+00:00 /October2 7:57:51 PM PDT**. Scientificdeadline10minutes earlier; evaluation/reporting included. No automatic extension or restart. The independent guard/caffeinate prevents idle-systemsleep during this job and targets only verified matching worker/supervisor processes at the absolutecap.

Checkpoint3/96productionepisodes independently CPU-loaded and digest-verified. All86namedAdamstates at3; every parameter tensor changed, native streams advanced32episodes/condition, initial model equalled the seeded constructor with empty Adam/streams, MPSRNG persisted. SHA256`8d085ef62165dfd99606b662836fecb86fc167ec69de2065a077e4d220c358f6`. [Parentverification](LocalRuntime/production_verified.json). Startup snapshot reached4updates/128episodes; this is not a current-step or acquisition result.

Validation at1,315/2,631 uses100trials/condition, selects mean3conditionAUC thenBA/earlierties. Fresh paired selected/terminal finals use200trials/condition and retain target/foil/catch confusion. No held-out accuracy result is available yet. Nativeprofile/production optimizer evidence replaces any repeated broad CPU campaign.

Runtime records: [live status](/Users/jonathanmorgan/VAWMRuntime/delayed_frame_gru_local01/run/live_status.json), [allocation](/Users/jonathanmorgan/VAWMRuntime/delayed_frame_gru_local01/run/allocation.json), [guard](/Users/jonathanmorgan/VAWMRuntime/delayed_frame_gru_local01/run/guard_status.json), [config](/Users/jonathanmorgan/VAWMRuntime/delayed_frame_gru_local01/run/config.json). The inherited allocation prose has a stale `micro4` label; actual immutable profile/production configs and executed training use **micro1**. A separate execution-metadata receipt records this correction without rewriting saved allocation or live sources.
