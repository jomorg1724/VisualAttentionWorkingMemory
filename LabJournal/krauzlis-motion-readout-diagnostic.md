# Motion-specific readout diagnostic — frozen selected2297

**Analysis-decoder training only.** The original native task, stimuli, curriculum and deployed model were not modified. These readouts estimate physical motion events and changed side; this is not successful acquisition of the complete cue-conditioned task. No cloud or deployed-model optimizer was used.

Raw-pixel adequacy gate: **PASS**, fixed before fitting and before independent test generation. Train600 and validation150 are exact hash-verified rerenders of the prior adequacy train/validation movies; test300 is newly generated, disjoint from every prior saved movie.

Validation pixels: side BA **0.984 [0.961, 1.000]**, event BA **0.988 [0.976, 1.000]**, event AUC **0.996 [0.987, 1.000]**. All seven prespecified criteria were evaluated without consulting test results.

Fresh test pixels: side BA **0.988 [0.974, 1.000]**, event BA **0.893 [0.830, 0.946]**. Frozen early-CNN maps with the same displacement/comparison family: side BA **0.531 [0.469, 0.592]**, event BA **0.500 [0.500, 0.500]**.

**The pixel control succeeded; the identical motion-comparison family did not transfer to frozen early-CNN maps.** The neural event classifier reports an event on every test movie, including every catch, and side decoding is not reliably above chance. This narrows the failure to the tested substrate/readout pairing rather than repeating the old raw-pixel probe failure. It does **not** prove that the CNN erased motion information: brightness/feature constancy, stride, nonlinear feature transformations, channel weighting and spatial support are not equivalent across substrates. No random encoder was included, so trained-weight effects cannot be separated from architectural effects.

**Next decision:** retain the frozen deployed observer and unchanged teaching. Close this bounded comparison; do not install this analysis decoder or infer an encoder lesion. If the neural readout underperforms despite the pixel control, the unresolved question is whether the frozen representation preserves usable motion under a representation-appropriate, independently calibrated comparison—not whether an external-history decoder already remedies the deployed task. Any further diagnostic requires a separately justified design and authorization.

Full evidence/report:`SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/MotionReadoutDiagnostic/REPORT.md`.

No deployed updates, stimuli/curriculum changes or cloud. Fresh small analysis decoders only. The matched neural comparison remains subject to spatial/feature-constancy limitations; no encoder-erasure or deployed-remedy inference.
