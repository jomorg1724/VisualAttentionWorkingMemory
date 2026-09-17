# Independent read-only review before production

Reviewer: `/root/temporal_neuroscience_design`. Date: 2026-09-12.

Inspected the research design and actual `model.py`, `train.py`, `sweep.py`, `evaluate.py`, and the completed `focused_check.json`. No training, additional tests, or runtime edits were performed by this reviewer. The user already authorized this comparison; this note is not an additional permission gate.

**Finding: no substantive implementation or protocol objection to proceeding at the equal exposure fixed from the planned profiles.** The proposed 40,000 fresh episodes per arm remains a profiled target, not a claim of sufficient acquisition. The new absolute allowance is 14,400 seconds for both arms combined, including profiles and evaluation; no renewal is implied.

## Computation and gradient flow

- Both arms reuse the same selected WM step-6860 parent and identically initialized new spatial interface/output tensors. Motion-energy outputs and the fused sensory field are computed at every observed frame and feed the new recurrent computation.
- The LSTM has independent gate normalization, with retention/input biases applied afterward. Its cell state is not normalized. Its nominal initial retention timescales span 4–128 frames. The small residual output is nonzero, allowing initial gradients into the new branch.
- The E/I implementation applies fixed presynaptic signs to columns of `F.linear`'s weight matrix, including after optimization. Magnitudes use inverse-softplus initialization with incoming E/I totals of 0.6 each. Rate and adaptation updates both use the old state; nonnegative rates/adaptation and bounded leak/adaptation parameters are preserved. No state or recurrent-current normalization changes this interpretation.
- Explicit state and non-reentrant activation checkpointing preserve full temporal gradients. The existing focused check records nonzero early/late representation and encoder gradients for both models, parent compatibility, common initialization, and recurrent sign preservation. These are implementation checks, not acquisition evidence.
- Both arms have 512 additional state scalars, but unequal parameter counts: 603,144 new LSTM parameters versus 306,184 new E/I parameters. Equal persistent-state size does not imply equal compute or representational capacity.

## Optimization and exposure

- New parameters use learning rate 3e-4 and transferred parameters 3e-5. Generic weight decay excludes normalization parameters, biases, raw constrained recurrence and timescale/adaptation variables. Pre-clip gradients and clipping frequency are logged; clipping is not described as a forward-stability guarantee.
- The small E/I raw-recurrence gradient in the CPU check motivated a shared Adam epsilon of 1e-10 before GPU profiling. The planned profile/training diagnostics record effective recurrent-weight updates and per-entry gradient scales for both arms. This is a disclosed numerical choice, not proof that optimization is solved.
- The 40-update cycle contains two updates for each sensory anchor and nine for each of motion L2, motion L8, minimal one-item orientation recall, and delayed one-item orientation recall. At the full proposed exposure this means 2,000 episodes per anchor and 9,000 per new-task cell in each arm. Actual fixed totals must come from the run receipt.
- Task-local seeds, cell order, initialization, sampling schedule and held-out examples are shared across arms. Checkpoint selection is the actual arithmetic mean of six cell AUCs. Per-cell BA/confusion and acquisition curves remain necessary to interpret that selection.
- The main scope contains motion and orientation only. There is no attention, content addressing, item-slot mechanism, new sensory generator, or extra model arm. The reset evaluation removes added recurrent history while leaving opponent traces intact.

## Interpretation limits to carry into the report

E/I mean-drive balance and bounded isolated leaks do not guarantee full-loop stability. Its initial rate timescales of 2–8 frames with weak adaptation are more contractive than the LSTM's initial long-retention coordinates; these disclosed initial conditions and their learned trajectories matter when interpreting different acquisition rates.

The previous broad run did not reliably acquire the minimal new rules. Therefore this focused comparison should be read as a new acquisition experiment, not a calibrated capacity test or an explanation of the earlier failure. A healthy authorized run should continue through its fixed budget despite early chance scores. Neither a hard-to-optimize E/I model nor a finite chance endpoint disproves a neuroscience mechanism.

Improvement over the earlier broad run could reflect the focused task exposure and shared new interface as well as recurrent memory. A reset intervention is informative about a trained model's history use but is not a separately trained parameter-matched control. No such extra control or further review is required to launch the comparison already authorized.
