# LocalDiagnostic

Analysis-only comparison of a pooled and spatial-priority output trained on
identical cached terminal fields from the frozen motion-only step-12200 model.

- [`PROTOCOL.md`](PROTOCOL.md): preregistered splits, probes and metrics.
- `python -m WorkingMemory.SpatialPriorityReadout.LocalDiagnostic.check_model`
  performs CPU construction checks.
- `python -m WorkingMemory.SpatialPriorityReadout.LocalDiagnostic.run
  --mode smoke|production --out <directory> --deadline-unix <seconds>`
  performs bounded extraction/fitting.
- [`report.md`](report.md) and `runs/` contain final evidence after execution.

The active cloud SpatialPriorityReadout process is neither queried nor
modified by this local script.
