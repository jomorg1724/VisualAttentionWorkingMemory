# Explicit second eight-hour Doom continuation

The user requested a new pod and another run equal in length to the preceding
eight-hour allocation. This run resumes continuation01/latest287281 exactly,
including all model and target-network weights, critic/actor/temperature Adam
states, replay buffer, validation counter, and random states. The selected
best281213 (ten-episode validation mean270.473155) remains eligible.

One RTX4090 pod `kekhy7y09gklsb` in EU-RO-1 costs$0.89/hour. The new dedicated
20GB network volume `l62z8etmmo` preserves checkpoints after compute stops and
continues billing until deleted. The allocation starts October10 at21:36:55PDT
and ends October11 at05:36:55PDT, including setup and validation. The independent
guard requests an episode-boundary save before stopping compute. No automatic
restart or extension is authorized. Original recovered files remain intact.

The original architecture, discrete distributional SAC objective, skill5 task,
actionrepeat4, Adam3e-4, clip100, replay capacity20000, two eight-transition
segments per update, and one-million cumulative decision ceiling are unchanged.
The same PyTorch2.8/Python3.12 image and ViZDoom1.2.3 environment are restored.
Dependencies live in `/opt/doom-venv` on the pod-local disk to avoid slow
small-file installation on the network filesystem. All training state and
checkpoints remain under `/workspace` on the persistent volume.

`upload_verified.json` proves transfer integrity. `training/resume_prepared.json`
records exact restoration and a discarded disposable CUDA update.
`independent_production_verified.json` is required before claiming training:
it checks a subsequent saved checkpoint, finite aligned advanced Adam states,
updated weights, unchanged original hashes, and live worker/guard identities.

Remote root: `/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/continuation02`.
Write `training/stop_requested.json` to request an early clean stop. Preserve and
verify checkpoint downloads before deleting the network volume.
