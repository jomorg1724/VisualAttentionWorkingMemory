# Explicit Doom continuation — October 10

The user requested resuming Doom on a new pod after the previous pod and volume
were deleted. The verified October 9 archive restores the original source,
parent suite checkpoint, latest114931 and validation-selected best105761.

The new pod is `dgqbm6fw23icyw`, an RTX4090 in EU-RO-1 at $0.89/hour, using
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` from the official
`runpod-torch-v280` template read. Network volume `dscudkmrjb` provides20GB at
$1.40/month while retained. It mounts at `/workspace` and preserves checkpoints
after compute stops. The fixed eight-hour allocation starts at pod creation,
October10 08:46:04PDT, and ends at16:46:04PDT, including setup and evaluation.
The existing one-million decision ceiling remains unchanged.

`prepare_resume.py` verifies source/scenario/checkpoint hashes, restores the
model and all three optimizers, checks replay sampling and CUDA RNG, and runs
one disposable CUDA learning update. It prepares a separate training directory
by changing only allocation/provenance metadata in a copy of latest114931.
All model, optimizer, replay, evaluation-counter and random states are preserved.
The resumed worker uses the unchanged original trainer's `--resume` path.
Original checkpoints and source remain immutable; old best remains eligible.

`cloud_control.py` provides an independent timed guard and detached supervisor.
The guard requests a clean episode-boundary save before the deadline and stops
the pod, using its provider-issued pod-scoped credential. It also stops compute
after the supervised worker exits. No account API credential is uploaded.
There is no automatic restart or allocation extension.

Deployment is described in `deployment.json`. Only
`independent_production_verified.json`, plus the matching live process and
status, establishes that resumed production is running. `training/resume_prepared.json`
records exact restoration; `training/exit.json` and `stop_receipt.json` record
the eventual stopping outcome. Write `training/stop_requested.json` for an early
clean stop. Preserve and download checkpoints before deleting the volume.
