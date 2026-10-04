# Krauzlis fresh acquisition: local completion and cloud attempt02

## Local native-angle run — completed, not acquired

Saved report: `/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run/report.json`.
The report records `planned_complete_cycles`, no failure, complete final coverage,
**4595 updates / 147040 episodes**, and validation-selected checkpoint **2297**.
All weights were fresh. This native-angle local run is not the +/-90 cloud variant.

| Final held-out role | Checkpoint | B12 AUC | B20 AUC | B28 AUC | Mean AUC | Mean BA |
|---|---:|---:|---:|---:|---:|---:|
| Validation-selected |2297|0.547175|0.535190|0.520170|0.534178|0.500000|
| Terminal |4595|0.536108|0.538097|0.602713|0.558973|0.500000|

Sources: `test_selected.json` and `test_terminal.json` in that run directory.
Each role evaluates 200 episodes per condition: 114 target, 58 foil, 28 catch.
Every prediction in both roles is positive: target hit rate 1.0, foil/catch false
positive rate 1.0, specificity 0.0. These scores do **not** establish acquisition
under this exposure; they do not establish that the architecture cannot learn.
Terminal's higher test AUC does not replace the validation-selected winner.

### Completed frozen local diagnostic of selected2297

See [the diagnostic journal](krauzlis-frozen-diagnostic.md) and
[full report](../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/FrozenDiagnostic/REPORT.md).
425/100/175 independent train/validation/test groups, paired cue variants, no deployed
updates or cloud access. Fresh native head: BA .500/AUC .622, all positive; paired
AUC .527 and zero action flips. ConvGRU cue decoding survives the gap (BA1.000)
and remains accessible at report (BA .663), while final-label decoding is near
chance (.513). Actual dot-direction errors are coarse (memory approximately41–46°).
Matched temporal-access probes did not reliably rescue the label. This rejects a
blanket claim of complete cue erasure, not all possible retention/binding failures.
Checkpoint/tensor hashes and native parity passed; all work remained inside the
single nonrenewable1800s diagnostic cap.

## Cloud attempt02 — executable prepared, not rented by this researcher

User explicitly selected a **new eight-hour / $5** attempt, not an extension of
attempt01's expired cap. Runtime:
`/Users/jonathanmorgan/VAWMRuntime/cloud_krauzlis90_fresh02`.
Parent owns deployment. No cloud calls or GPU work were performed in preparation.

The same CNN + three KDAs + ConvGRU + terminal spatial transformer + dense heads
will be freshly constructed, with fresh Adam, RNG and native streams. No local,
failed-cloud, or disposable-profile checkpoint is an input. Reused recorded seeds
make this a retry of the same seeded +/-90 experiment, not an independent-seed
replication. Task remains B12/B20/B28, batch32/micro4, Adam1e-4, unclipped fp32/full
BPTT; target10000, minimum1000, prospective profile-bound allocation.

Reproduced the old failure with the real CPU Session: checkpoint3 was durable but
the inherited hook still published verifier step1 (`assert 1 == 3`). The scoped
fix calls the native persisted verifier at the adapter's step3 startup save.
Actual native CPU profile6 -> fresh production3 -> persisted checkpoint and named
Adam -> launcher `production_ready=true` now passes. The launcher also regenerates
stale verification against actual checkpoint bytes rather than treating receipt
lag as failed optimization; genuine unverifiable state remains bounded/fail-closed.

Clean archive suite: **8 passed in 46.88s**. Runtime/guard/native handshake suite:
**18 passed in 132.93s**. CPU tests are infrastructure evidence, not model results
or A40 throughput. The 60-second profile-pin-to-supervisor handoff reserve remains
in place and has passing acceptance/rejection tests. Early checkpoint3 plus cloud
cadence1000 remain unchanged. Old attempt sources/logs were not modified.

Guard stops on absolute deadline, setup/worker failure, or hash-verified finalization
with owners exited; completion retains disk instead of waiting for laptop retrieval.
A40 must be <=$0.49/hour, no parallel active pod; $1 storage reserve and 24h retained
storage horizon. Parent must retrieve and clean retained storage inside that horizon;
no automatic deletion or stopped-pod restart is authorized by this launcher.
