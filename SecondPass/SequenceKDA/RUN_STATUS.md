# Single layer sequence KDA cloud training

**Completed 2026-10-02 at 17:55:29 UTC / 10:55:29 AM PDT.** All **4,216 updates / 134,912 episodes** completed with no reported failure. The validation-selected checkpoint is update2,108; terminal is4,216. Fresh paired tests contain200 episodes per condition for each checkpoint. Both predict change on every test trial: **50% balanced accuracy** in B12/B20/B28, 57% raw accuracy from the target base rate, 100% target hits, 100% foil false alarms and 100% catch false positives. This run did not acquire the task at the pinned exposure; it does not establish that longer training or every single-layer KDA design must fail.

| Condition | Selected BA | Selected AUC | Terminal BA | Terminal AUC |
|---|---:|---:|---:|---:|
| B12 | 50.0% | 0.4170 | 50.0% | 0.4207 |
| B20 | 50.0% | 0.5688 | 50.0% | 0.5342 |
| B28 | 50.0% | 0.5144 | 50.0% | 0.5524 |

Selected mean test AUC is0.500102; terminal0.502422. Both are near chance. Total optimizer time was5,757.306seconds (95.96minutes).

All45 manifest files were retrieved and verified. Selected and terminal checkpoints were CPU-reloaded, each with27 active Adam states. Pod`8gamd8ems1pa0n` was stopped and deleted after verified retrieval at17:56:01UTC. Evidence: [report](CloudRuntime/artifacts/report.json), [selected test](CloudRuntime/artifacts/test_selected.json), [terminal test](CloudRuntime/artifacts/test_terminal.json), [retrieval](CloudRuntime/retrieval_verified.json), [cleanup](CloudRuntime/cleanup_verified.json). No further training is running or newly launched.

The following launch record is historical.

**Training launched and independently verified on 2026-10-02.** Pod `8gamd8ems1pa0n`, one A40 at $0.49 per GPU hour, configured Palladio account `jonathan@palladio.ai`. Architecture is exactly one KDA layer over chronological native RGB patches with final CLS classification. All 158,340 parameters are trainable and freshly initialized.

Native GPU profiling prospectively pinned **4,216 updates / 134,912 episodes**, reduced from the requested target of 10,000 updates to fit conservative measured costs. B12 and B20 receive 1,405 updates / 44,960 episodes each; B28 receives 1,406 / 44,992. Batch32/micro4, one Adam1e-4, no clipping, strictly FP32 and full temporal gradients. Native 26/28 degree changes, cues, labels and 57/29/14 target/foil/catch proportions are unchanged. Profile weights, optimizer, streams and counters were discarded before production.

The eight-hour/$5 cap began **2026-10-02 16:15:49.999944 UTC**. Scientific work cutoff is **2026-10-03 00:05:49.999944 UTC**; independent hard stop is **00:15:49.999944 UTC / October 2 5:15:49 PM PDT**. There is no automatic extension or increase in the pinned exposure. The $1 storage reserve is included in the envelope, using [published storage pricing](https://docs.runpod.io/pods/pricing).

Independent startup evidence is persisted: checkpoint3 contains 96 production episodes, all 27 parameter tensors changed, all named Adam states advanced to3, native streams advanced and the persisted initial model matched the seeded direct constructor with empty Adam/streams. The checkpoint was downloaded, SHA-256 verified and CPU-reloaded. SHA-256: `8bd23465f908fb498f7243a3260065f5623e5c521eced0aa71bf6bbb51446bef`. A live snapshot already showed update11/352episodes; this is a startup snapshot, not the current step or an acquisition result.

Validation is planned at2,108 and4,216,100trials/condition; selection uses three-condition mean AUC then BA, earlier ties. Fresh paired selected/terminal finals use200trials/condition and keep target/foil/catch rates visible. No held-out result is available at this launch snapshot.

The authenticated pod-local guard and private provider-backed status are verified. Detached mirror PID88271 polls every15seconds and has already retrieved the allocation and an early checkpoint. It verifies final artifacts and both final checkpoint reloads before exact-pod cleanup. The absolute guard stops even if the laptop is disconnected; incomplete retrieval retains the stopped disk and never triggers a GPU restart.

Source and launch evidence are in [CloudRuntime](CloudRuntime/): `production_verified.json`, `downloaded_checkpoint_verified.json`, `guard_verified.json`, `budget.json`, `mirror_started.json` and mirrored `artifacts/allocation.json`. Scientific archive SHA-256: `0ac8815627019975cdc30968c136b3174cd123f8662266255195f216bfac3800`.

The user's request to avoid exhaustive checks controls subsequent work. Repeated remote CPU suites were removed; launch used the already verified native CPU optimizer evidence plus the required native GPU feasibility profile. Healthy slowdown is recorded as a warning and does not trigger cancellation. If the finite cap limits actual exposure, completed final evaluations are retrieved with actual and pinned exposure explicitly distinguished.
