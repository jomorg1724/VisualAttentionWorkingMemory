# Unified task-suite assembly — 2026-09-22

## Question and authorized scope

The user asked to assemble all tasks discussed in the current KDA-model conversation, with training across the full suite **later, from fresh weights**. This is implementation and CPU validation, not a new learning experiment. No training or cloud run was launched.

## What changed

[SecondPass/TaskSuite](../SecondPass/TaskSuite/README.md) supplies a machine-readable catalog and a thin unified stream adapter over the existing renderers:

- Seven original corrected sensory tasks: cardinal motion direction, signed orientation, contrast, spatial frequency, chromatic increment, contour grouping and natural-image spectral detail.
- Ring-cued orientation direction, the current KDA model's initial acquisition task.
- Five spatial tasks: signed location-cued orientation, cued motion duration, Krauzlis target/foil motion change, retrospective spatial binding, and exact-image set recognition.

There are 13 tasks and 35 primary task/condition cells. Sensory tasks retain exactly two presented frames and their original difficulty mixtures. Sequence tasks retain their original cues, labels, timing and primary delay/load/baseline conditions. Empty recognition lists remain always-negative controls, ineligible for BA/AUC selection.

The task/condition streams use new split seeds and independently checkpointable native RNG states. This is a versioned sample-order change, not a change to stimuli, and not a replay of historical draws. Delay cells are not paired presentations. No auxiliary angle labels, new cue vocabulary, old checkpoint migration or training curriculum was added.

## Verification, not performance

Actual command results:

- `/tmp/vawm-task-suite-venv/bin/python -m pytest SecondPass/TaskSuite/test_suite.py -q`: **8 passed** after the independent review's dataset-resume correction.
- `/tmp/vawm-task-suite-venv/bin/python -m SecondPass.TaskSuite.verify --model-smoke`: **35/35 cells passed, zero blocked, zero training updates**. Four episodes per cell were rendered, replayed and checked for native equivalence. See [verification.json](../SecondPass/TaskSuite/checks/verification.json).
- Fresh random KDA with all 13 heads: finite CPU forward outputs for every task, using its longest primary sequence. All parameters remained trainable; state tensors were unchanged by inference. It has 2,750,324 parameters including the expanded heads. No weights were loaded or saved and no optimizer was constructed.
- Existing sensory `check()` and motion `test_motion()` checks also passed. Their pixel-observer scores establish renderer signal consistency, not neural-model acquisition.
- BSDS500 official data were downloaded and prepared in the ignored local data directory. Its 200/100/200 source splits are disjoint by manifest source ID and file hash; all 500 image files are present. Tests additionally draw photo tasks across train/validation/test and check exact image membership and repeat semantics.

Practical issues: the host's PyTorch 2.2.2 NumPy bridge was incompatible with global NumPy 2.0.2, so CPU checks used an isolated temporary environment with NumPy 1.26.4; global packages were not changed. The first official BSDS500 download hit the existing 300-second limit, then a bounded resumed transfer completed and the original preparation routine extracted the genuine dataset. No substitute images were used.

## Interpretation and next decision

The independent read-only inventory confirmed all 13 renderer mappings and
identified that native recognition-stream snapshots omit photo-manifest identity.
The suite wrapper now records that identity and rejects changed-manifest restores
before mutating stream progress; a regression test uses only a temporary manifest
copy. All 35 cells and fresh-model forward checks passed again. The review's
missing-data warning predates the successful download recorded above.

This shows interface compatibility, correct reuse of native task laws, stream replay, and executable generation across the suite. It does **not** show learning, retention capacity, general attention, or a successful multitask optimizer. The current trained KDA model still has only its previous orientation-family training history.

Before the later authorized training run, implement the reporting contract and choose the task/condition sampling allocation, loss weighting, effective batch, validation selection and finite compute budget. Initialize all model weights and optimizer state afresh, keep all learned weights trainable, and do not inherit the old split learning rates or five-task clipping recipe. Task grouping does not mandate a staged curriculum. Keep a plain-model comparison in the research design; this assembly does not authorize another arm.

The older seven-task sensory and five-task spatial protocols remain sources of task definitions, not current launch authorization. Earlier trained models, checkpoint lineage and results are preserved.
