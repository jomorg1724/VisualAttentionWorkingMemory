# Upstream motion comparison — frozen selected2297

## Outcome

**Native pixels support motion comparison; these fixed neural probes do not provide a rescue. The unique neural bottleneck remains unlocalized.**

300 fresh independent native episodes (258 events,42 catches), each also extracted with a matched cue-swapped variant. Reused425 training/100 validation groups; all fits and thresholds frozen before fresh test generation. Validation reuse remains exploratory.

- Full-history image observer: changed-side BA **0.965 [0.940,0.985]**, any-event BA **0.880 [0.821,0.932]**, actual target-report BA **0.877 [0.837,0.915]**, AUC0.952. Detects233/258 events with6/42 catch false positives. Cue decoded300/300.
- Image observer restricted to the last two transitions of each phase: side BA **0.760 [0.699,0.810]**; target-report BA0.730. Its side advantage over matched-time CNN25 phases is+0.264 [0.188,0.335]. Full-history access alone does not explain the whole image/probe gap. Geometry, event timing, speed/mass and handcrafted computation remain substantial observer privileges.
- Direct pre/post side probes: CNN25/KDA25/KDA7 BA0.496/0.496/0.489; memory0.477. Final-memory0.449; phase-minus-final gain+0.027 [−0.054,+0.112]. No reliable temporal-access rescue under the matched2144-feature quadratic assays.
- Full-memory phase direction errors remain36–41°, larger than native26/28° events. Direct signed-change estimates largely collapse toward zero (changed-patch error26.9–27.4°, zero-prediction baseline27.16°). Full-history pixel changed-patch error7.41°.
- A weak positive is retained: subtracting prior full-dimensional memory angle estimates gives side BA0.566 [0.511,0.619], but changed-patch error54.4°. One shuffled side control reaches0.562; no multiplicity correction. Do not infer neural erasure or a unique causal encoding/retention/comparison lesion.
- Native trained head remains all-positive, BA0.500/AUC0.475. No task, architecture, deployed training or cloud changes.

## Verification and constraints

Selected2297 checkpoint and runtime source hashes unchanged. Exact original/instrumented renderer pixels, labels, metadata and RNG; exact hooked/unhooked logits at B12/B20/B28. Fresh noncue hashes disjoint from all previous splits.48 scaler reconstructions and ridge equations passed; every new and reused direction prediction replayed with error0. Pixel-method parity to3.55e-15 degrees. Final-only earlier-state removal/NaN invariance passed. Two contract tests passed. CPU numerical threads≤2, one scientific worker, one immutable1200-second cap including implementation, extraction, fitting, verification and reporting.

Initial launch failed before fitting because optional `threadpoolctl` was absent; removed that dependency, retained pre-import numerical environment limits and PyTorch caps, preserved the failed log and original deadline. Premature process-manager null-exit notices were rejected using OS/log/receipt verification; no duplicate scientific run. Final elapsed and artifact hashes: `final_receipt.json`.

Full evidence: [REPORT.md](../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/UpstreamMotionDiagnostic/REPORT.md) and [DETAILS.md](../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/UpstreamMotionDiagnostic/DETAILS.md). Predictions, fits, fresh features/angles, pixel flow summaries, example rasters, grouped bootstrap indices, source/checkpoint identities, selections, audit scripts and receipts live beside the report. Prior artifacts preserved.
