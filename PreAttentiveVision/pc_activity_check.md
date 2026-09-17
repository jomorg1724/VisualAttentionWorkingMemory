# PC activity check after the seven-task run

On 2026-09-12, at the user's request, inspected live Windows process names/command lines, NVIDIA GPU utilization and process listings, Windows GPU engine counters, and a five-second CPU utilization sample.

- The benchmark completion receipt reports success; its budget has `active: null`.
- No project training supervisor or model worker was present in the live process inventory. Python processes observed before the new CPU analysis were Hermes gateway, service and kernel processes, not this project's trainers. None appeared as an active CUDA training process in the NVIDIA listing.
- NVIDIA snapshots: approximately 27% GPU utilization, 2859–3199 MiB allocated across desktop applications, 60–61 C, 20–21 W.
- Windows engine counters attributed visible activity to Edge video decoding and 3D rendering, and Desktop Window Manager 3D rendering. GPU counter snapshots and NVIDIA utilization use different sampling/aggregation, so their percentages should not be summed as a complete decomposition.
- Codex's main process was the largest CPU consumer in the five-second sample, about 6.3% of total logical-processor capacity; Edge and Hermes were substantially lower during that sample. Lifetime accumulated CPU time was not used as a measure of current load.

No unrelated application was terminated. These observations exclude a visible stray project trainer at the time checked; they do not identify every intermittent fan burst or measure fan speed. The ensuing component-combination investigation uses saved scores on CPU only, with no GPU model inference or training.
