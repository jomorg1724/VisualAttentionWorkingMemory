## October 11: kill-only experiment recovered and deleted

The separate [kill-only experiment](../../DeadlyCorridorKills/README.md) was
stopped at105216 decisions at the user's request. All results and checkpoints
were downloaded and verified, then its pod and dedicated volume were deleted.
No Doom compute or cloud storage remains. See its run01/cleanup_verified.json.

## October 11: separate fresh kill-only experiment active (historical)

User explicitly requested fresh initialization and reward only for enemy kills.
See [DeadlyCorridorKills](../../DeadlyCorridorKills/README.md). This original
native-reward experiment and its recovered weights are preserved. New pod
yq0g7zledvfwuk stops compute at17:31PDT October11; dedicated volume7gz6ogdywq
persists. Independent fresh-initialization and actual learning verification passed.

## October 11: Doom recovered and all cloud resources removed

Continuation02 stopped at474136 decisions/108056 learning updates. Best468005
validation average286.950343; latest100 training average261.259792. All103 files
and both checkpoints recovered and verified locally. Both pods and the dedicated
volume deleted; provider confirms zero pods/volumes. See
`continuation02/cleanup_verified.json` and `continuation02/recovered`.
No active Doom run. Entries below are historical.

## October 10 second continuation active

User explicitly requested another eight-hour run. Doom resumes latest287281 on
RTX4090 pod `kekhy7y09gklsb`, using network volume `l62z8etmmo`. Exact restoration
and actual saved learning progress independently verified. Compute stops at
**October11 05:36:55PDT**; storage persists. See
[continuation02](continuation02/README.md) and its production receipt. Prior
continuation01 was recovered and both old pods and volume deleted; entries below
are historical.

## October 10 continuation active

The original run01 pod and volume were deleted after verified local backup. Doom is now resumed from114931 on new pod `dgqbm6fw23icyw`, with exact learning-state restoration. See [continuation01 protocol](continuation01/README.md), `continuation01/deployment.json`, and `continuation01/independent_production_verified.json`. Eight-hour compute cap ends October10 16:46:04PDT. Historical run01 details follow.

# Deadly Corridor cloud run01

Explicitly authorized October 9, 2026. This is the first production Doom run,
separate from the completed 16-task suite and frozen VDA analysis.

## Allocation

- Pod `jcfmgww5ubnvax`, RTX 4090 24 GB, Secure Cloud EU-RO-1, $0.89/hour.
- Created October 9 at 17:09:53 PDT. Fixed allocation ends October 10 at
  **01:09:53 PDT** (08:09:53 UTC), including setup. No automatic extension.
- Decision ceiling: 1,000,000 agent decisions, each repeating an action for
  four game tics. The time cap can end training before this ceiling.
- Independent 20 GB network volume `hb9ruanesc`, mounted at `/workspace`,
  estimated $1.40/month while retained. Compute ceiling $7.12 plus disk/storage.
- Official live template `runpod-torch-v280` supplied image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; observed digest
  `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
- Strictly verified SSH: root at 213.173.102.177, port 35328. The local existing
  SSH key authenticates; no private key or account API credential was uploaded.

## Scientific protocol

Prepared model and objective are retained: the successful suite best54808
initializes the trainable visual/recurrent encoder. The same 64-dimensional
post-GELU belief representation feeds a 54-action actor and twin 32-quantile
critics. Discrete SAC, Adam 3e-4, gradient clipping 100, native rewards scaled
by .01 for learning, skill 5, action repeat 4, RGB-only observations.

Whole episodes supply pixel replay. The current and target encoders reconstruct
memory from the episode start before each two-segment, eight-transition update.
There are 2,000 initial exploratory decisions. Actor, critic, entropy-temperature
optimizers, target networks, replay and all RNG states are saved at episode
boundaries. Validation uses 10 fixed episodes approximately every 5,000 decisions;
30 fresh test episodes follow only if the decision ceiling is completed.

Cloud runtime uses Python 3.12.3 / PyTorch 2.8.0+cu128, compared with local
preparation's Python 3.10 / PyTorch 1.13.1. Exact cross-platform numeric identity
is not claimed. Four contract tests and a full production-sized disposable CUDA
update passed. ViZDoom is pinned to 1.2.3. WAD hashes match exactly; configuration
hashes differ solely in CRLF versus LF line endings, verified after normalization.

The first production attempt stopped before any learning updates because a log
record repeated the `steps` keyword. The sole runner fix names the episode-local
quantity `episode_steps`, retaining cumulative `steps`. The failed attempt is
archived remotely under `run01/failed_logging_attempt`; the allocation deadline
was not extended. Source hashes and the change are recorded separately.

## Checkpoints and controls

Remote run directory:
`/workspace/vawm/SecondPass/DeadlyCorridor/CloudRuntime/run01/training`.
`latest.pt` contains resume state; validation-selected `best.pt` contains weights
and its selection result. An expired allocation cannot automatically resume.

The independent cloud guard requests a clean episode-boundary stop three minutes
before the allocation deadline, then terminates the worker process group if
needed and requests RunPod pod shutdown. It also requests shutdown after normal
worker completion. The guard uses the provider-issued **pod-scoped** key in a
root-readable container file; credentials are excluded from code, logs, bundles
and checkpoints. Its identity was verified through an authenticated read of
this pod. The control API request at the future deadline cannot be pre-validated
by actually stopping a currently running experiment.

To pause early, write `stop_requested.json` in the remote training directory.
Check `exit.json`, `supervisor_exit.json`, `guard.json`, and live provider state
before reporting that compute has stopped. Network storage remains independently
billed after compute stops. Do not delete the network volume before retrieving
and verifying checkpoints.

Evidence: `run01/deployment.json`, `preflight_verified.json`,
`independent_production_verified.json`, `source_manifest.json`, `logging_fix.json`.
