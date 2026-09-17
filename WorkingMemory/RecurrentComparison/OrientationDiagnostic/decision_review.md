# Independent read-only review

Reviewer: `/root/temporal_neuroscience_design`, review identifier `orientation_frozen_v1_source_review`, completed before first GPU launch on2026-09-13.

The reviewer inspected actual run.py and supervise.py. No substantive issue was found: actual-angle instrumentation preserves images/labels/RNG; capture indices match the renderer; all delay variants stay within their base split; scalers/fits/thresholds use train/validation before held-out access; independent probe encoding resets sensory traces; comparator reads r and adaptation remains diagnostic-only. Sample-trained transfer retains its original scaler. Postprobe fits predict the task label and do not reconstruct the sample angle.

Resolved protocol refinements before launch: run the small D0/D24 operational comparators irrespective of angular MAE; add a validation-selected circular-angle-distance comparator without new extraction; permit the same fixed MLP64 fallback on isolated-probe angular decoding when ridge validation MAE exceeds3.75degrees. These are bounded offline fits, not architecture or main-model training changes.

Interpretation: weak probes do not establish information erasure. D0 is the comparator positive control; failure there limits D24 failure interpretation. Strong [r,a] recovery alone does not establish firing-rate readout access. Any postprobe label rescue addresses operational use, not an uncontaminated sample reconstruction.

## Final interpretation review

The same reviewer confirmed the completed report: preprobe firing-rate recovery of about 4.25 degrees and the label-only comparator at 73.44% support long-delay comparison/use as a practical bottleneck. The 87.30% circular result uses stronger angle supervision and explicit distance logic; it is analysis-only and is not a universal improvement. Sample-to-late transfer failure can include offset/scale calibration, not uniquely a rotated code. The postprobe linear result does not establish erasure. No further tests were requested.
