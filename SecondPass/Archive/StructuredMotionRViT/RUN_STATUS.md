# Run status

**COMPLETED — 2026-10-03 06:13 UTC.** All660updates/20,000presentations/2,000unique. Fresh selected meanBA50.37%, terminal50.67%; no further local training of this model. [Final report](FINAL_REPORT.md). Earlier live notes below are historical.

**2026-10-03 01:30 UTC: fresh local TRAINING, optimizer progress independently verified.**

Author-derived Simoncelli–Heeger motion CNN at five scales, causal nine-frame windows, fresh appearance/fusion CNN and RViT readout. All learned parameters initialize fresh; fixed analytic coefficients are buffers. No inherited task or segmentation weights.

Production checkpoint3/96presentations was independently CPU loaded: every one of92learned parameter tensors changed, all92Adamstates advanced, published analytic buffers unchanged; initialdirectconstructor/emptyAdam/freshstreams verified. [Parent receipt](LocalRuntime/production_verified.json).

Profile pinned **660 updates / 20,000 presentations / 2,000 unique movies**. Two native1000-movie pools, each reused10shuffled epochs,33updates/epoch including tails; no new teaching. Effectivebatch32/micro1/evalmicro1, Adam1e-4/zero decay/no clipping, FP32/full BPTT, two CPUthreads. Fixed analytic frontend onCPU; all learned CNN/RViT/readout parameters onMPS. ExactCPU motion features verified in the mixed-backend native29-frame backward check. Profile state discarded before fresh production.

Validation100/250/500/660 with100trials/cell; meanAUC/thenBA/earlier tie selection, fresh paired200/cell selected/terminal finals. No production validation or held-out result was available at this launch snapshot.

Runtime `/Users/jonathanmorgan/VAWMRuntime/structured_motion_rvit_local01`. Supervisor20405/PPID1, productionworker22149, independentguard20403, one local accelerator worker. Newfinite8h cap conservatively retains firstcompatibilityattempt01:15:49UTC; hardend09:15:49UTC /October3 2:15:49AM PDT, scientificcutoff09:05:49UTC. Setup/profile/eval/reporting included; no automaticextension/restart.

PreviouslocalCNN-GRU cancelled and saved2575updates/82400trials; no inherited state or finaltest claim. Its worker/supervisor exited and its launchd jobs were removed. Cloud RViTreplay and16headKDA retain their unchanged caps. [Architecture/proposalPDF](TechnicalDocument/architecture_proposal.pdf).

Earlier implementation checks: [threeCPUchecks](verification/model_cpu.json) passed4.21s; [Appleexecutioncheck](verification/mps_compatibility.json). These establish implementation readiness, not acquisition.
