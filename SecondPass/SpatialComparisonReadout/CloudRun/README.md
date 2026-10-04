# CUDA single-candidate execution

[BRIEF.md](BRIEF.md) is the versioned cloud contract. Older local two-arm sources and artifacts are preserved and are not relaunched.

- `worker.py`: narrow CUDA/full-checkpoint adapters around the native harness; candidate-only disposable profile and production supervisor.
- `test_cloud.py`: focused trusted-load, source/migration/initial-logit/native-stream replay, real CPU gradient/Adam/weight-update and immutable-cap regressions. Actual CUDA profile additionally verifies saved CUDA RNG replay and post-update state.
- `bundle.py`: minimal import-closure/source-hash bundle with the actual500 BSDS500 photos, exact identity manifest, terminal6760 checkpoint, reused baseline and historical timing summaries. No credentials, git history or unrelated runs.

Remote layout: `/workspace/vawm/repo`, `/workspace/vawm/assets`, `/workspace/vawm/run`. Official PyTorch2.8.0+cu128 runtime, NumPy2.1.2; explicitly trusted checkpoint loads use `weights_only=False` because legacy RNG state contains NumPy objects. No frozen native source edits are needed. Local stock Torch2.2.2/NumPy2 is incompatible for native sampling; the focused real CPU checks run in the deployment runtime (local NumPy1.26 compatibility path is separate, not a source change).

Supervisor entrypoint (once only; run detached with stdin closed, log redirected, new session):

```sh
cd /workspace/vawm/repo
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -m SecondPass.SpatialComparisonReadout.CloudRun.worker supervise \
  /workspace/vawm/run --created <original-pod-creation-epoch>
```

Never derive a new cap at launch or resume. `activation.json` exclusively records original creation/deadline; `profile/profile.json` is disposable, not production. The supervisor pins `allocation.json` and `config.json`, restores production independently from terminal6760, and launches exactly one candidate. There is no automatic restart. Exact cloud resume state is saved; a new resume would require an explicit same-deadline action.

Evidence: `progress.jsonl` for persisted completed updates; `latest_checkpoint.json` for full-state receipt; `persisted_progress_verification.json` for inherited and comparator weights/Adam/native-stream advancement; `first_cycle_timing.json` for matched production/profile cost. `production_supervisor.json` records durable worker/supervisor identities and compute deadline.

Final outputs: `REPORT.md`, `report.json`, `test_terminal.json`, `test_selected.json`, selected/terminal full checkpoints, and `cloud_completion.json` with status, requested/pinned/actual updates and SHA256 artifact manifest. All35 cells, empty recognition specificity and Krauzlis target/foil/catch remain explicit. Parent owns artifact retrieval and pod deletion. A live process or a completed profile is not a production launch claim.
