# One independent read-only review

Reviewer: `/root/temporal_neuroscience_design`; protocol, actual source and final branch interpretation were reviewed within the same diagnostic decision.

Protocol review: split whole paired templates and their rotations before nuisance repeats; index5 is the transition4 state; the sensory128 baseline contains opponent traces; distinguish r from[r,a]; reuse the modest MLP64 for count regression as well as duration classification. Implemented: canonical individual schedule banks are split first, then pairs are formed within each split, which additionally prevents any individual template overlap. Duration probes read sensory+r only. The same MLP64 architecture serves prefix/final[r,a] count regression and duration classification.

Actual-source response: “No requested runtime edits or interruption. Canonical individual split is stronger than paired-only and correct; t5 prefix, train scaler, locked val selection, available sensory+r duration route and128-template grouped bootstrap all hold.” Reporting qualifications: last-two matching does not fix transitions5–6; first-four count recovery can reflect later correlated evidence. Raw-score AUC differs from earlier softmax-AUC reports. These limits are explicit in the report. Routine CPU target-dtype recovery did not change scientific scope, selections, exposure or the original absolute deadline.

Final response: “Prioritize readout fitting, defer slow-E. Final r already carries strong linearly accessible counts and operational sensory+r ridge rescues+9.47pp; MLP offers no observed advantage. One wording/implementation nuance: probe concatenates384 dimensions; deployed readout projects r→128 then adds sensory, so rescue may include bypassing that learned fusion bottleneck, not mere recalibration. A targeted existing memory_output+motion-readout fit with frozen recurrent/sensory dynamics is justified, followed by fresh standard task evaluation before broad benefit claims. No further diagnostic requested.”

Resolution: recommend one focused existing-readout fitting follow-up. No slow-state branch is forced; no further diagnostic or training is launched by this report itself.
