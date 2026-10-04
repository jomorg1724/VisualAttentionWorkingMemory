# Krauzlis +/-90° — whole-model-fresh cloud variant

**Status: not launched.** Preparation and CPU verification only. No GPU profile,
cloud provisioning, API write, training launch, restart or retrieval performed.
The existing local original-angle run and every original source remain untouched.

## Scientific contract

Only `krauzlis_cued_motion`, all B12/B20/B28 conditions. Whole direct constructor
`SpatialConsolidation(task_classes())`: CNN, all three KDA modules, final spatial
ConvGRU, terminal 49-token spatial transformer, dense readout and every head
freshly initialized. Other task heads remain unused and are explicitly audited
as unchanged, with no Adam state. No inherited weights, profile state, optimizer,
RNG, streams or training resume. Fresh seed namespaces are in `worker.py`.

`stimuli.py` is a private byte-preserving copy of the native renderer except one
expression: consume the native `choice([26,28])`, then set magnitude to90.
Native random +/- sign, labels and cues remain unchanged. Two cue frames, five
fixation-only frames, reference frame, B12/20/28 baseline transitions, eight
postevent transitions, report;16 dots/patch, speed.375 pixels/transition,
16° spread, lifetime10, target/foil/catch57/29/14 per100. Catch movies are exactly
identical with paired seeds. Pre-event movies match exactly; after an actual
change, boundary-reset counts can differ, so later RNG draw consumption and later
trials need not remain paired. This is an explicit magnitude adaptation, not the
original Krauzlis26/28° stimulus. Native metadata is otherwise preserved;
experiment protocol/provenance identify the variant.

One Adam1e-4, fp32, TF32 disabled, full BPTT, no clipping, effective batch32/micro4.
Independent train/validation/final-test namespaces and task-condition streams.
Validation selection: mean of three condition AUCs, then mean BA, earlier ties.
Two validation looks100/cell; selected and terminal final tests200/cell, paired
fresh final draws. BA/AUC/confusion per condition; target hit, foil false alarm,
catch false positive and subgroup confusion retained. No pooled substitute.

Prospective useful target:10,000 updates, capped by one exact-path disposable
single-A40 profile (two full updates/condition and evaluation timing). Planner
accounts for setup elapsed, all evaluations, checkpoints/reporting,25% training
and35% evaluation margins plus900s overhead and separate600s retrieval reserve.
It refuses fewer than1,000 updates rather than silently converting acquisition
into a pilot. This threshold is a prespecified exposure floor, NOT evidence of
convergence. Exposure is pinned once before production, reported per condition,
and rechecked against actual remaining time. All profile state is discarded.

## Launch and spending contract

Parent alone owns provisioning and paid execution. New single NVIDIA A40 only;
old pods are deleted and must never be queried for credentials or restarted.
Hard deadline is creation-time+28,800s (eight hours), including setup, profile,
training, final tests and retrieval. Scientific deadline is600s earlier. Live
quote at creation must satisfy `8 * compute_hourly_usd + storage_reserve_usd <=5`;
reserve must cover running disks and bounded retained-disk retrieval/cleanup.
No automatic extension, restart, resumed optimization or extra review gate.

Runtime: `/Users/jonathanmorgan/VAWMRuntime/cloud_krauzlis90_fresh01`.
Remote package root: `/workspace/vawm_krauzlis90_fresh01`.
`pod_guard.py` reuses the prior verified status/stop transport but changes the
completion contract: after full manifest/hash/coverage verification AND actual
worker, supervisor and owner exit, publish status and stop compute immediately,
retaining persistent disk. Retrieval acknowledgement is NOT a stop prerequisite.
Publication failure cannot keep a completed GPU billing. Explicit setup/worker
failure stops promptly; unclaimed setup stops at15min; absolute8h wins regardless.
Provider lifecycle must be read back by the parent before claiming compute off.

If parent is online, retrieve live promptly and verify every published hash.
If offline, disk remains billable after compute stop; later short zero-GPU-only
retrieval requires separate authorization. Never resume training for retrieval.
Reserve/cleanup horizon must be explicit before rental; unlimited storage cannot
be guaranteed within$5. Delete retained storage only after verified retrieval.

### Concrete pre-rental blocker

The prior deploy wrapper obtains `VAWM_STOP_API_KEY` from deleted pods. It is NOT
reused. The local `~/.runpod/config.toml` backend exists and a sanitized read-only
`GET /v2/pods` probe returned HTTP200. It has no local scope metadata, and no
separate `VAWM_STOP_API_KEY` exists in this session. Its suitability as a
scope-limited pod-local stop/status credential is therefore **not verified**.
No credential contents were printed, no API writes were made, and no key was
copied to the package.
Parent must resolve and read-only verify suitable independently available guard
authority before rental; do not copy a broad control-plane credential as a silent
fallback. Also obtain a live A40/storage quote and a new private never-deployed
status record before rental. This preparation is not permission to spend while
fixing authentication. The supplied launcher does not provision pods.

## CPU reproduction

```sh
cd /Users/jonathanmorgan/Desktop/VisualAttentionWorkingMemory
PY=/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python
$PY -m pytest SecondPass/SpatialReadout/SpatialConsolidation/Krauzlis90Fresh -q
$PY -m SecondPass.SpatialReadout.SpatialConsolidation.Krauzlis90Fresh.bundle \
  --output /Users/jonathanmorgan/VAWMRuntime/cloud_krauzlis90_fresh01/package
cd /Users/jonathanmorgan/VAWMRuntime/cloud_krauzlis90_fresh01
$PY launch.py check
```

Do not rebuild into an existing identical deployment directory; immutable bundle
builder intentionally refuses overwrite. CPU tests use only temporary disposable
checkpoints, never shipped in the archive. Runtime tests mock provider transport;
no live shutdown claim follows from them.

## Parent execution instructions (not executed)

1. Resolve the blocker above, live quote/storage retention reserve, read-only
   credential authority and status-record identity; enforce the guard from pod
   creation, not after SSH setup. Embed supplied stdlib `pod_guard.py` as the new
   pod entrypoint, with guard-only secret env `VAWM_STOP_API_KEY`, `RUNPOD_POD_ID`
   supplied by RunPod, `VAWM_HARD_DEADLINE`, `VAWM_STATUS_TEMPLATE_ID`, and SSH
   public key. Guard pops stop secret before starting `/start.sh`. No old-pod auth.
2. Parent creates only the authorized new A40 with persistent `/workspace` disk,
   captures creation-time budget, verifies actual rate and guard authenticated
   read/status publication. No auto-restart policy. The guard's15min unclaimed
   setup timeout is a backstop, not permission to leave a known failed setup idle.
3. Upload ONLY the archive in `package/bundle_receipt.json`, plus runtime
   `pod_guard.py` and `remote_owner.py`. Verify full SHA256 before extracting with
   `tar --no-same-owner`; verify all deployment-manifest hashes. Load
   `/etc/rp_environment` if present and install package requirements in the
   official Torch2.8 CUDA12.8 runtime. Run scoped CPU tests in that interpreter.
4. Write immutable `assets/budget.json` with `cap_started`, `hard_deadline`,
   `deadline=hard_deadline-600`, `wall_cap_seconds=28800`,
   `retrieval_reserve_seconds=600`, `max_usd=5`. Write `assets/pricing.json` with
   `gpu="NVIDIA A40"`, `gpu_count=1`, `compute_hourly_usd`,
   `storage_reserve_usd`, `verified_live_rates=true`, `observed` (epoch of live
   quote, at most15min before creation). Both are parent-observed evidence.
5. With guard alive/authenticated/publishing, run the one-shot owner detached:
   `cd /workspace/vawm_krauzlis90_fresh01; env -u VAWM_STOP_API_KEY nohup python3 -u remote_owner.py >owner.log 2>&1 </dev/null &`.
   Owner automatically executes prepare → bounded profile → measured pin →
   guarded production → finalize. No extra approval queue. Duplicate ownership
   is rejected. Read back persisted progress and first-cycle timing, not merely
   PIDs, before saying training started.
6. On completion, guard verifies full scientific manifest and exited process
   owners before status+stop. Read exact pod lifecycle and private status. Retrieve
   and hash-check promptly if online; otherwise report stopped compute/retained
   storage and seek separately bounded retrieval-only authorization.
