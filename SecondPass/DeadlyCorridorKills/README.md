# Deadly Corridor: fresh kill-only experiment

## October 11 user stop and recovery

Stopped cleanly at105216 decisions,20677 learning updates and2476 episodes. Best
checkpoint20397 validation mean is0.7kills/game; latest evaluation101106 is0.0.
Final100training games average0.90kills. All89 archived files match their cloud
hashes, and latest/best checkpoints reload successfully with replay and optimizer
state preserved. Results are under CloudRuntime/run01/recovered.

The pod and dedicated network volume were deleted; fresh provider lists show
zero pods and volumes. See CloudRuntime/run01/cleanup_verified.json. No Doom
training is active. The local suite continues independently.

Authorized October 11, 2026. The recurrent visual architecture and distributional discrete SAC are retained. The encoder, actor and critics start randomly. Target networks copy the fresh online networks; Adam and replay start empty. No suite or Doom checkpoint is loaded.

Reward is `max(0, KILLCOUNT_after - KILLCOUNT_before)`, including the terminal action. Each kill contributes one unit directly to replay and the Bellman target (reward_scale=1). Native scenario rewards are discarded. Death, damage, movement, survival and pickups contribute zero. Skill 5, 2100-tic horizon, 54 legal button combinations, action repeat 4, and RGB-only 100x100 inputs are retained. Kill counts are used only for reward and reporting.

Adam 3e-4, clip 100, twin 32-quantile critics, learned SAC entropy temperature, discount .99 per four tics, Polyak .005, 20,000-transition episode replay and two eight-transition segments per update remain as implemented in the original run. Unit rewards replace the old native-reward scaling of .01. There are 2,000 exploratory decisions before updates. Evaluation averages kills over 10 fixed games every approximately 5,000 decisions and selects the best mean kills. Per-game outcomes are retained.

The ceiling is 1,000,000 decisions under one eight-hour compute allocation starting at pod creation, including setup; no automatic extension. Persistent network storage preserves checkpoints after compute stops. The original Doom experiments and active local suite remain intact.

The [ViZDoom API](https://vizdoom.farama.org/main/api/python/doomGame/) exposes the [KILLCOUNT variable](https://vizdoom.farama.org/1.2.4/api/python/enums/). Real-engine checks verify positive kills, zero movement/death reward and reset accounting. Initial weights and empty optimizer/replay state are saved as initial.pt for independent verification. Run records are under CloudRuntime/run01.
