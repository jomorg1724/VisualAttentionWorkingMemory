# Krauzlis-only whole-model fresh local run

Current user-authorized experiment: ONE newly constructed terminal spatial-transformer model, exclusively native `krauzlis_cued_motion` B12/B20/B28. The architecture is unchanged `../model.py`: CNN, three KDAs, spatial ConvGRU, terminal transformer, dense readout and task heads. All learned parameters are trainable. Other task heads are inactive and expected not to update; the trunk is not frozen.

No pretrained/checkpoint input, initialization migration, profile-state transfer, anchors, curriculum, clipping, auxiliary loss or stimulus changes. One Adam at 1e-4; fp32 full BPTT; effective batch32/micro4; one Apple MPS optimizer worker, at most two CPU threads. Native queues retain 57 target/29 foil/14 catch cases per100 draws, labels, cues and photometry.

## Execution and artifacts

Canonical live runtime: `/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run`.
Source-only hash-verified runtime copy: sibling `repo/`; no `.pt` inputs copied. Original repository and prior artifacts are unchanged except this isolated adapter and journal additions.

Durable owner: launchd `gui/501/org.vawm.krauzlis-wholemodel-fresh01`, one-shot `RunAtLoad=true`, `KeepAlive=false`, deliberately using Interactive scheduling to match the measured production path. A CPU-only probe imported the actual runtime and verified child SIGKILL at its two-second absolute deadline before any accelerator work. Probe receipts are in sibling `probe/`. No tracked Hermes background job is used or needs ownership handoff. This survives chat closure, not machine shutdown; the absolute cap never renews.

`budget.json` fixes eight hours from immediately before the first profile process, covering profile, training, validation, final tests and reports. Supervisor kills its worker process group at that absolute deadline. Update-boundary finalization reserve precedes it. There is no restart/resume CLI or automatic extension.

The disposable profile executes two complete optimizer updates for each native condition and20 evaluation episodes/cell through the production launcher. The plan uses maximum measured cost per condition, the exact fresh seeded condition schedule, 1.25 training/1.35 evaluation margins, two100/cell validation looks, two200/cell final tests and900 seconds overhead. It targets10000 updates but pins a lower feasible count before production if necessary. See `allocation.json` for exact exposures and any reduction. Profile optimizer/model/streams are discarded; production reconstructs everything and verifies initial tensors independently.

## Validation and evidence

Fresh seed118192763; scheduler118392763. New train/validation/test namespaces118492763/118592763/118692763. Validation selection is mean three-cell AUC, then mean BA, with earlier ties; final test draws are separate and paired across terminal/selected roles. Native evaluator extended only to record target/foil/catch subgroup confusion, hit rates, specificity and false-positive rates. Undefined subgroup metrics are null, not zero.

`test_worker.py` was first run failing for the missing adapter, then passed in the actual Torch2.8 CPU runtime (3.52s final prelaunch run). It checks no checkpoint loading during initialization, every initial tensor against direct construction, empty Adam/streams/counters, native pixel/label/metadata equivalence, all-condition updates, inactive heads, namespace separation and evaluator coverage/subgroup fields.

`constructor_equality.json` binds checkpointzero to every direct-constructor tensor. `initial.pt`, early checkpoints1/2/3, periodic100-update checkpoints and validation/terminal checkpoints persist full model, named Adam, scheduler, train stream, CPU/Python/NumPy/MPS RNG, counters, optimizer time and selection history. Each is fsynced, hash-verified and read back before its pointer advances. `persisted_progress_verification.json` checks used module changes and Adam progression; `first_cycle_timing.json` checks production against matched profile conditions. `progress.jsonl` fsyncs every completed optimizer update and finite-gradient norm. Final outputs are `report.json`, `REPORT.md`, `test_terminal.json`, `test_selected.json` and full-state checkpoint pointers.

The reused final report's heading/legacy limitations text mentions the broader original ConvGRU suite, but its protocol, architecture, actual exposure and all evaluated cells identify this Krauzlis-only terminal-transformer run. Do not treat profile scores or training loss as held-out acquisition, and do not infer cloud state from this local run; no cloud API was called.
