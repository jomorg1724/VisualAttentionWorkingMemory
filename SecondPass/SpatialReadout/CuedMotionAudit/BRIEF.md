# Cued-motion audit — frozen analysis, not a training change

## Question and scope
The user clarified that the two targets are `motion_duration_cued` and `krauzlis_cued_motion`, not orientation. Check whether native input pixels support the report labels, and localize the current model's failures where possible. Do not promise a uniquely identified cause from a limited diagnostic.

The active A40 continuation remains unchanged. No cloud profiling, extra worker, optimizer update, new architecture, teaching change, or cap extension is authorized by this analysis.

## Fixed model
Use the downloaded current checkpoint `checkpoint_035039.pt`, cumulative step35039, from the whole-model-fresh ConvGRU lineage, subsequently continued with explicit user authorization. Local path: `/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_continuation_01/artifacts/checkpoint_035039.pt`. Verified16764891bytes; SHA256 `93762f7968a29092b8acce7ea9632937a23965160822fe98bda9b3e5844531e7`. It is a current periodic checkpoint, not the validation-selected28539 winner. Do not mix predecessor warm-start models into this diagnosis.

## Minimal discriminating tests
1. CPU-only native renderer/label audit and nonlearned pixel observer. Decode cue from pixels; estimate motion from raster transitions rather than metadata. Known geometry/event timing are diagnostic privileges and must be disclosed. Threshold selection, if needed, uses separate calibration trials, not final diagnostic scoring trials. Success establishes usable information in rendered examples, not that this network will learn the rule. Failure of this observer does not establish impossibility.
2. Frozen neural behavior on fresh diagnostic seeds. Compare identical duration evidence atD0/D24, exact cue-retarget pairs with recomputed labels, and per-timestep deployed readout scores. Krauzlis cue swaps preserve the entire movie and interchange target/foil semantics, leaving catch negative. Report signed score/margin changes and both-members-correct as well as argmax changes. Earlier readout/OOD manipulations are diagnostic, not trained task performance.
3. Read-only deployment/source and saved-validation checks. Preserve source dependencies, report gradients/optimizer observations only when actually measured, and distinguish saved validation from the new frozen diagnostic.

## Resources and stopping rule
Pixel audit: one CPU thread, initial12-minute scope. Neural diagnostic: one local MPS worker (CPU fallback only if necessary), one CPU thread while pixel audit runs, new nonrenewable1200-second cap beginning before first accelerator operation and covering extraction/report. Pin sample sizes after a bounded first batch, keep its work in the budget, and leave partial evidence if the cap binds. No new paid compute. No broad feature-probe sweep or retraining.

## Interpretation boundaries
Simple two-frame motion success does not establish decoding of smaller peripheral dot patches, tiny subpixel angular changes, cue-conditioned selection, or counting over eight changing directions. D0 still contains temporal accumulation. Native frame stacking retains two earlier images, so the first report/blank step is not immediately sensory-free. All metadata are scoring/diagnostic records, never main-model inputs. Keep random replacement noise and exact native cue/label rules unchanged.

Outputs: pixel and model scripts, raw scored records, focused findings, parent synthesis/report and journal update. All results must be backed by actual execution.
