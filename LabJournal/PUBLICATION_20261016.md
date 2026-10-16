# October 16 code and documentation publication

This publication records the pixel and predictive VAE experiments, weighted-sum
models, recurrent contrastive models, local motion comparison, stabilized shared
attention, the task suite and VDA extensions, static and motion memory-guided
search, popout search, Doom training implementations, and their documentation.

The current local experiment is the 19-task static-memory 2x continuation.
`SecondPass/MemoryGuidedSearch/LocalRuntime/static_memory2x05` resumed update
173,247 toward 193,800. Static memory-guided search receives two batches in each
20-batch shuffled cycle; each other task receives one. The optimizer, partially
consumed task bag, condition queues, native streams and random-number states
were restored. See `SecondPass/MemoryGuidedSearch/STATIC_MEMORY_2X.md` for the
experiment history and `AGENTS.md` for operational status.

At the last verified status before publication, update 175,136 was training.
Monitoring at 174,767 gave equal-task mean AUC 0.9443616221. The selected best
was update 169,870 with mean AUC 0.9463159310. These are the 71-condition
monitoring results; full final evaluation for this continuation is pending.

Per the user's publication instructions, model weights, datasets, raw trial
records, videos, animation exports and bundles containing those artifacts stay
local. Existing tracked historical artifacts are not rewritten or deleted.
Configuration/catalog definitions, source code, tests, prose, and explanatory
figures are retained. Local checkpoint and data paths referenced by the code
must be supplied separately to reproduce checkpoint-dependent analyses or
resume training on another computer. No LFS objects are introduced here.

Training continues during publication; this commit is a source/documentation
snapshot and does not pause or restart the worker.

Publication checks: 184 staged Python files compile; the memory-guided search
contracts and weighted-scheduler contracts pass (13 tests). Credential-pattern
scanning found no candidates in the selected files. The staged file list
contains no weight, dataset-array, raw CSV/JSONL, video, animation or archive
formats. Source files retain their original bytes because active checkpoints
pin their SHA256 hashes; historical whitespace and line endings are preserved.
