# Deadly Corridor: separate RL experiment

Prepared October 8; **cloud production started and independently verified October
9, 2026** on RunPod RTX 4090 pod `jcfmgww5ubnvax`. First verified checkpoint:
2,354 decisions, 37 updates, 35 episodes; all 67 Adam states finite and aligned,
54 encoder tensors advanced. See [cloud run protocol](CloudRuntime/README.md).
Fixed allocation ends October 10 at 01:09:53 PDT; cloud-side shutdown guard armed.
This experiment is not a task-suite member and does not overwrite the successful
suite model. No held-out game-performance result is available at launch.

## Architecture

100×100 RGB -> the existing correspondence / appearance encoder -> the same
shared spatial attention, spatial GRU, CLS GRU and comparison trace -> the
existing Linear(799,64)+GELU readout -> **one shared 64D representation h**.
This is exactly the tensor the 13 belief heads currently read.

Three new heads consume h: a categorical actor and two action-value critics.
Each is Linear(64,128), ReLU, Linear(128,output). The actor has 54 logits; each
critic has 54×32 quantile outputs. No game variables, labels, health, depth map,
automap, previous reward or privileged positions enter the encoder or heads.

The online encoder starts from suite checkpoint 54,808 and is trainable through
the critic objective. The actor head uses the same h with its encoder gradient
detached. This common shared-encoder convention avoids competing actor and
critic updates to the trunk; it does not freeze the trunk. Target encoder and
target critics track their online counterparts with Polyak coefficient .005.
Suite classification heads and the pixel auxiliary loss are not RL objectives.

`encoder.py` implements the original computation incrementally: five observed
gray frames, sixteen local population/reliability frames, motion trace, spatial
memory, CLS memory, previous flow and comparison memory. It does not reset
memory at every game step. Prefix equivalence is checked beyond both temporal
window lengths against the original forward computation and trained weights.

## Learning rule

The base algorithm is **discrete Soft Actor-Critic**, with fixed-quantile
distributional critics. Discrete SAC and quantile regression are established
methods; this particular recurrent combination is an implementation adaptation,
not a claim to reproduce a published Deadly Corridor benchmark result.

For action a, Q_i(h,a) is the mean of N=32 predicted return quantiles z_i(h,a,j).
For each next action separately, choose the whole target-critic distribution
whose mean is smaller. Do not take an elementwise minimum of quantiles.
The distributional target retains the categorical next-action mixture:

    y[a,j] = .01*r + (1-done)*gamma^(elapsed_tics/4)
                         * (z_target(h_next,a,j) - alpha*log pi(a|h_next))
    w[a,j] = pi(a|h_next) / N

The reward r is the raw accumulated native scenario reward over the repeated
action. Gamma=.99 per four-tic decision; reward scaling .01 is fixed and raw
returns are always logged. A terminal transition has every target atom equal
to .01*r. No reset image can bootstrap a preceding episode.

Each chosen-action critic is fitted by weighted quantile Huber regression:

    L_Q = mean_i sum_(a,j) w[a,j] * |tau_i - I(y[a,j]-z_i < 0)|
                               * Huber_1(y[a,j]-z_i)
    tau_i = (i+.5)/N

Both critics contribute to the encoder gradient. The actor objective is

    L_pi = E_h sum_a pi(a|h) * [alpha*log pi(a|h) - min(Q1(h,a),Q2(h,a))]

The learned temperature targets entropy .5*log(54), starting at alpha=.1.
Its log-alpha loss is log(alpha)*(observed_entropy-target_entropy); low entropy
therefore increases alpha. Adam3e-4, FP32 and gradient clipping100 are used.

Replay stores **pixels and complete episodes**, not stale learned features.
Two segments of up to eight transitions are sampled per update. The current
and target encoders separately replay the entire real episode prefix under
no_grad, then unroll the training segment. This makes recurrent state match the
respective current weights without inventing a reset 32 frames before sampling.
Gradients are truncated at the segment boundary. Full-prefix reconstruction
costs more than approximate burn-in; production throughput still needs profiling.

Collection uses fixed weights within each episode. Learner updates occur between
episodes (up to32 updates, approximately one per four collected decisions after
2,000 warmup decisions). This avoids changing weights beneath a live recurrent
state. Replay capacity is20,000 transitions (~600MB RGB storage plus overhead).

## Environment contract

- Installed ViZDoom1.2.3, native `deadly_corridor.wad`, skill5, native reward and
  death penalty100, finite2100-tic scenario horizon. Configuration and WAD hashes
  are recorded at launch.
- **Skip three intermediate frames = hold the action for four tics**, then
  observe one RGB frame. The motion windows therefore operate in observed-frame
  units, a deliberate temporal-domain shift from the suite.
- The whole320×240 view is area-resized to100×100 and quantized once touint8 for
  identical online and replay pixels; aspect ratio is changed, nothing is cropped.
- HUD, weapon sprite, crosshair and sound disabled. This is a disclosed rendering
  configuration; native rewards/difficulty and scenario geometry are preserved.
-54 categorical action combinations: three strafe choices × three forward/back
  choices × three turn choices × attack/noattack. Opposing buttons never co-occur.
- Death, success and the scenario's finite horizon terminate the episodic return.
  A compute-cap interruption is not a synthetic terminal training transition.

## Launch and evaluation

Prepared command from the repository root (not executed):

```powershell
C:/Python310/python.exe -m SecondPass.DeadlyCorridor.launch doom --run SecondPass/DeadlyCorridor/LocalRuntime/run01 --steps 1000000 --wall-seconds 28800
```

The decision target is a configurable upper bound, not a throughput estimate.
The independent supervisor enforces one finite local allocation with no cloud
or automatic restart. Collection stops at episode boundaries, so the decision
count can exceed the target by at most one episode. A deadline exit is incomplete,
not a finished million-step result.

Training/validation/test use distinct episode-seed ranges. Ten fixed validation
episodes approximately every5,000 decisions select the highest mean raw-return
checkpoint;30 separate final-test episodes evaluate the selected model only
after planned completion. Report raw return, episode length and death rate;
do not label surviving until timeout as successful corridor completion.

Latest saves include all optimizer/target/temperature states, episode replay,
RNGs and counters at an episode boundary. Direct `train --resume` honors the
saved deadline without extending it. No claim is made to serialize the live
Doom engine mid-episode. Original suite files remain read-only.

## Sources and checks

- [Discrete SAC](https://arxiv.org/abs/1910.07207): categorical actor and exact
  summation over discrete actions.
- [Distributional RL with quantile regression](https://arxiv.org/abs/1710.10044):
  fixed quantiles and quantile Huber objective.
- [Distributional SAC](https://arxiv.org/abs/2004.14547): distributional
  maximum-entropy actor-critic context; the published continuous-action model
  is not copied as if it were a discrete implementation.
- [Official ViZDoom API](https://vizdoom.farama.org/main/api/python/doom_game/):
  action-repeat and accumulated reward semantics.
- [Official scenario configuration](https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/deadly_corridor.cfg).

`test_contract.py` verifies streaming/full-prefix equivalence, episode reset,
distributional action mixtures, terminal masking, quantile gradients, replay
alignment,54 valid actions, actual engine four-tic stepping, identical head input
tensors, finite encoder/actor/twin-critic updates, and unchanged parent hashes.
These are preparation checks, not evidence of a trained policy.
