# Latent-state dynamics atlas

Status: **paused at the user's request before any extraction or analysis**. See [queue receipt](queue_receipt.json). The user redirected this work to [ordinary attention-map visualization](../AttentionMaps/README.md). The broader PCA/tSNE/probe implementation below remains preserved but is not running. Its pending 1800-second allowance was transferred to the narrower visualization; it is not a second allowance.

Primary model: the frozen selected pre-update attention checkpoint 8400. It is pinned before this analysis and independent of the later Stage1 test results.

The six cells are the existing single-orientation, two-location orientation binding, and eight-transition motion-duration tasks, each at D0 and D24. Target 256 training, 64 validation and 128 held-out base episodes per family; profile may pin a smaller finite pilot. D0/D24 share identical episode evidence and labels. All frames from one episode stay in the same split. Actual rendered angles and direction schedules are recorded only as analysis metadata.

We save per-frame sensory field H, rates R, adaptation A, the 128-dimensional combined decision representation, source-attention masses, rate-channel averages and rate-site averages. H/R/A are each pooled from 64×13×13 to 64×3×3 and stored losslessly as fp32 compressed arrays. This preserves coarse location while discarding within-bin detail. The deployed model retains its full state. Adaptation access is diagnostic only.

Outputs include train-fitted shared PCA and exploratory six-cluster coordinates, complementary descriptive tSNE, frame trajectories, held-out linear readouts of axial orientation, signed change/swap labels after the probe, motion direction, winner and centered duration fractions. Evidence regression also measures residual information after conditioning on final direction at fixed elapsed time. Cross-time readouts test whether a learned code remains accessible to the same decoder. Channel tuning and spatial maps support exploratory segmentation without claiming biological cell types.

UMAP is not installed in the active training environment. PCA/tSNE meet the requested exploratory visualization purpose without package changes. Cue images and phase are coupled in these tasks, so semantic cue selectivity cannot be identified separately here. Before-probe change labels must not be read as available future information.

The finite analysis cap is 1800 seconds from extraction/profile, including fitting and rendering. The persistent supervisor enforces it. Main weights remain frozen and checkpoint hashes are checked. A single wrapper/pixel/RNG equivalence check validates instrumentation; no repeated launch gates.

When complete, open [the standalone interactive atlas](index.html), [report](report.md), [numerical results](results.json) and [research rationale](RESEARCH.md). Compressed feature files remain available for reanalysis without rerunning the network. A final comparison to a later Stage1 checkpoint is deferred from this pilot.
