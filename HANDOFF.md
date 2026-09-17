# Handoff — reset to first principles

Prepared 2026-09-16 after the user judged the AV-context v2 arm, and with it
the whole from-scratch line, a failure. This document replaces the 2026-09-14
handoff (still in git history). It is a handoff, not authorization to train.

Repository: `C:/Users/jomor/Documents/VisualAttentionWorkingMemory`, pushed to
[github.com/jomorg1724/VisualAttentionWorkingMemory](https://github.com/jomorg1724/VisualAttentionWorkingMemory)
(private). Windows, PowerShell, Python 3.10, torch 1.13.1+cu117, laptop RTX
3070 (8 GB). Checkpoints, datasets and result bundles are on disk but not in
git; every receipt records their SHA256. **Nothing is running locally or in
the cloud, and no RunPod pod exists on the account.**

Read next: [current status](LabJournal/CURRENT_STATUS.md),
[chronology](LabJournal/CHRONOLOGY.md),
[experiment 24](LabJournal/experiments/24-av-context-v2.md), then this file's
audit list.

## 1. Where the project stands

The user's assessment, which the evidence supports: the lineage was a house of
cards. Each component was validated on the previous component's terms, the
whole chain was never shown to be learnable end to end, and when the new
five-task battery demanded genuinely new computation (cue-directed selection,
signed change, multi-patch duration integration) nothing in the stack could
acquire it.

What has been tried on the current five-task battery, all stopped:

| Arm | Init | Exposure | Outcome |
|---|---|---|---|
| Biased five-task (exp. 19) | warm start from attention 8400 | 140,920 episodes | binding 100%, recognition 97/83/72%, orientation ≈50%, motion 25%, Krauzlis 50% |
| Motion-only continuation (exp. 20) | warm start from cloud 10000 | +17,600 motion episodes | 25% at every delay |
| Prospective query (exp. 21) | warm start | 41,200 episodes | small orientation/binding gains, motion at chance |
| Spatial priority readout, scratch (exp. 22) | scratch | ≥200,000 | all tasks at chance |
| Dual attention, scratch (exp. 23) | scratch | ≈218,000 | all tasks at chance |
| AV-context v1, scratch | scratch | ≈256,000 | all tasks at chance |
| AV-context v2, five changes (exp. 24) | scratch | 424,840 (cloud) | all tasks at chance; training CE never left chance on 4/5 tasks |

The clean split in that table is not architecture. It is warm start versus
scratch. Every warm-started arm solved binding and recognition quickly; every
scratch arm solved nothing, including binding and recognition. The question
the v2 experiment was designed to ask (does cue-directed routing get used) was
never actually posed to the network because it never left the constant-output
regime. See the loss table in experiment 24.

## 2. Where we went wrong

1. **Frozen and slowed weights during development.** The PAV encoder was
   trained on two-frame change tasks, then frozen (`requires_grad_(False)`,
   `encoder.eval()`) while the temporal accumulators were selected on top of
   it. When everything was later unfrozen, all inherited parameters trained
   at 3e-5 while new parts trained at 3e-4, and that split was copied
   verbatim into every later recipe, including the scratch arms, where it
   trained 73% of a random network ten times too slowly. Component
   comparisons made under frozen or crippled predecessors selected
   components that fit a fixed representation, not components that could
   learn one. The user's rule going forward: never freeze weights during
   development, and never inherit a learning-rate split.
2. **Hard-coded computations that were never tested against learned ones.**
   The opponent accumulator uses a fixed Gabor quadrature bank, fixed
   fast/slow retention (0.25/0.75), a fixed opponent energy equation, fixed
   E/I sign masks with softplus magnitudes and bounded time constants.
   These were adopted as neuroscience-motivated priors and then carried as
   load-bearing structure. None was ever ablated against a plain learned
   alternative on the same tasks.
3. **Warm starts that hid learnability.** Every "success" came from a chain
   of continuations. Binding at 100% looked like evidence that the
   architecture works; it was evidence that a pretrained two-frame change
   detector plus a working recurrent memory can be fine-tuned into a swap
   detector. The first time the stack had to learn from scratch it could not
   even do that.
4. **A weak training signal for a long credit-assignment path.** One scalar
   label per episode of 5–45 frames, batch 8, five microbatches of different
   tasks averaged into one Adam step, gradient clipping at 1, full BPTT
   through encoder → traces → fusion → attention → E/I recurrence → readout.
   The scratch runs show the signature of this: gradient norm decaying from
   0.6 to 0.26 while outputs go constant.
5. **Too much scaffolding per experiment, too little diagnosis.** Each run
   came with pinned hashes, receipts, lifecycle monitors and cloud
   provisioning, and each ended at chance. The engineering was sound; the
   science had no ladder of small learnable steps under it.
6. **Never establishing that the tasks are learnable at all.** No standard
   model was ever trained on the five-task battery. We do not know whether
   `motion_duration_cued` or `krauzlis_cued_motion` can be learned by anything
   at this resolution and signal level. Section 4 gives specific reasons to
   doubt it.

## 3. Keep the tasks as the benchmark

The five tasks stay:
[PROTOCOL.md](WorkingMemory/SpatialTaskBattery/PROTOCOL.md),
[stimuli.py](WorkingMemory/SpatialTaskBattery/stimuli.py),
[SOURCES.md](WorkingMemory/SpatialTaskBattery/SOURCES.md),
[rendered previews](WorkingMemory/SpatialTaskBattery/previews/index.html).
Scoring: [summarize_spatial](WorkingMemory/UnbiasedAttention/protocol.py)
(within-task condition means, chance-normalised BA, mean OVR AUC; empty
recognition lists excluded from ranking). Fixed seeds: train 61973001,
scheduler 62973001, validation 63973001, test 64973001.

Keeping them as benchmarks does not mean keeping their parameters
unexamined. Section 4 lists what to verify before any of them is used to
judge an architecture again.

## 4. Audit the environments and the training logic first

> Done 2026-09-16/17: see also [experiment 26](LabJournal/experiments/26-plain-baseline-rung1.md) and [experiment 27](LabJournal/experiments/27-accumulator-conv-stack.md): environments sound; the specified recipe collapses on every task; a plain CNN+GRU learns the orientation family by curriculum; a gated spatial accumulator in the conv stack (ConvGRU/KDA) reaches ceiling at every delay where the plain model reaches 0.85-0.88. Audit detail: see [experiment 25](LabJournal/experiments/25-battery-audit.md) and `WorkingMemory/BatteryAudit/README.md`. Every task is recoverable from pixels by a fixed observer at D0 (BA 0.95-1.00); streams pass all checks; the v2 recipe clips every update to 0.36 from step one. The Krauzlis integer-rasterisation suspicion below is wrong (bilinear sub-pixel rendering).

Do this before any architecture work. Each item is concrete and cheap.

### 4.1 Motion duration (`_motion` in stimuli.py)

- Dots are rendered by `np.rint` to integer pixels after steps of 0.8, 1.2
  or 1.6 px. At 0.8 px a dot moves 0 or 1 px per frame, irregularly.
- 16 of 32 dots per patch are replaced at every one of the 8 transitions
  (`reset[rng.choice(32,16,replace=False)]=True`), so a dot survives two
  frames on average. The per-frame direction signal is ~16 dots × ~1 px
  against 16 fresh random dots.
- The label is the majority direction over 8 frames with a margin of only
  `max(1, length//8) = 1` frame (`_motion_schedule`), so schedules like
  counts [3,2,2,1] are legal.
- Four such patches, one cued by a thin ring (160 non-gray pixels in the
  cue frame; 0.07 contrast after 13×13 average pooling).
- The old single-field motion task that reached ~79% used 1–3 px steps on a
  single patch. This task is far harder and its difficulty was never
  calibrated against anything.

Test: train a plain strong model (3D conv or per-frame CNN + GRU, batch 64,
lr 1e-3, no fixed priors) on motion D0 only. If it cannot exceed chance in
100k episodes, the task is mis-specified for this resolution, not the model.
Then sweep step size, replacement rate and margin to find where it becomes
learnable, and record the human-plausible operating point.

### 4.2 Krauzlis (`_krauzlis`)

Step 0.375 px per frame, 16 dots per patch, radius 8 px, direction SD 16°,
event = a small mean-direction change. Same learnability test as 4.1; the
sub-pixel step plus integer rasterisation is the first suspect. Check the
57/29/14 event proportions are what the summariser assumes for balanced
accuracy.

### 4.3 Orientation (`_orientation`)

The ±glyph is 10×10 px placed 23 px above the target centre, which for the
top row lands at rows 4–14, near the border. Verify the glyph survives the
encoder's downsampling and that its sign is unambiguous at 13×13. Verify
`rotation_multiset_before_target_assignment` really makes the aligned,
opposite and unchanged locations equally likely under both labels (the
foil-control claim in the metadata).

### 4.4 Binding, recognition, cue timing

- Binding: confirm that "one swap per trial" plus a retrocue cannot be
  solved from the global change signal (the metadata claims this; test it
  with a model that sees no cue).
- Recognition: confirm split disjointness and that positive probes are
  byte-identical to a study image; confirm the empty-list cells never enter
  ranking.
- For every task, print the frame index at which the cue appears, the
  evidence frames, blanks and report frame for D0 and D24, and confirm the
  worker feeds all of them (`frame_count` versus the images tensor).

### 4.5 Label pairing and streams

`SpatialBatteryStream._case` draws labels through a `pending` queue so
classes are balanced within a stream. Verify per-task label counts over 10k
draws, verify `state_dict`/`load_state_dict` round-trips exactly (the resume
path depends on it), and verify validation/test streams with the fixed seeds
regenerate identical trials across processes.

### 4.6 Training logic

- Each update averages five per-task losses (`loss/5` each) and clips the
  summed gradient at 1. Log per-task gradient norms separately once; if one
  task's gradient dominates the clip, the others are effectively frozen.
- Batch 8 per task is tiny for a scratch run with one scalar label per
  episode. Establish the learnable batch size on the plain baseline first.
- `activation_checkpoint=True` with `preserve_rng_state`: confirm the
  recomputed forward is bit-identical to the direct forward (the readout
  dropout was deleted, so it should be).
- The E/I update uses previous rates in the adaptation term and bounded
  time constants (τ_r 1–32, τ_a 4–128). Check that a 24-frame blank at the
  initial time constants does not annihilate the state by construction
  (0.75²⁴ ≈ 0.001 for the slow trace is already known).
- Adam ε = 1e-10 and weight decay 1e-4 on matrices were never tuned.

## 5. Re-evaluate the architecture from first principles

Ask, for each piece, three questions: is it essential to solve the task
suite, is it hurting, and has it ever been shown to learn from scratch?

| Component | Status | What to do |
|---|---|---|
| Pretrained PAV encoder (convnext_se_residual) | never trained end to end on these tasks | Replace with a small plain CNN trained from scratch inside the baseline; compare only if the baseline learns |
| Fixed Gabor quadrature bank + fixed opponent energy | hard prior, never ablated | Compare against learned 3D/temporal convolutions; if the learned version wins or ties, drop the prior |
| Fixed fast/slow traces (0.25/0.75) | hard prior | Learn the retention or replace with a learned recurrent unit; the 0.75²⁴ decay makes them useless for D24 anyway |
| Average pooling to 13×13 | dilutes sparse signals; v2's max pooling did not rescue a scratch run | Keep resolution higher until the task is learnable, then compress |
| Pre-update joint attention with source bias and locality (λ=4) | helped delayed orientation in a warm start; head 1 was never recruited from scratch | Drop for the baseline; reintroduce only when a working representation exists and a cue-use failure is measured |
| Spatial E/I memory (Dale signs, softplus magnitudes, bounded τ) | worked when warm-started; never learned from scratch | Baseline with a plain ConvGRU or LSTM first; then test whether E/I constraints cost anything |
| Priority-map readout (spatial softmax × evidence) | uniform-priority fixed point demonstrated | Use a plain pooled readout for the baseline; treat priority as a hypothesis to test, not a default |
| Per-task heads with task-identity selection | fine | keep |

Suggested ladder, every rung with all weights trainable and a plain
comparison model beside it:

1. **Learnability of each task at D0 with a standard model** (per-frame CNN
   + GRU, batch 64+, lr 1e-3, 100k episodes). This is the environment audit's
   final word and the floor every later model must beat. Record it.
2. **Delay curve for that model** (D0/4/12/24). This measures what memory
   the task actually demands.
3. **One added inductive bias at a time** (spatial recurrence, E/I
   constraints, opponent temporal features, attention, priority readout),
   each compared to rung 2 on the same seeds, at least two seeds.
4. Only then a combined model, and only then any Guided Search framing.

The GS6 scaffold (no diffuser; activated long-term memory as synaptic
weights) remains the user's framing for the eventual system. It is not a
constraint on the baseline.

## 6. Rules for the next agent

- Never freeze parameters during development. Never inherit a learning-rate
  split. If a component is compared, it is compared with everything
  trainable.
- Every model must first be shown to learn from scratch on at least one
  task before it becomes a parent for anything.
- Change one factor at a time and keep a plain baseline in the table.
- Keep the receipts, pinned hashes and finite budgets; they are what made
  this failure diagnosable. But do not build cloud lifecycle machinery for an
  experiment before a laptop-scale version has left chance.
- The user wants study, learn, justify, build: read the primary source,
  explain the computation with equations and shapes, then implement.
- Finite compute. Do not provision cloud without an explicit instruction.
  Local GPU jobs run sequentially.
- Report per-condition results, never aggregates that hide a weak task.
  Distinguish measured performance from biological motivation.

## 7. Operational notes

- Run `python WorkingMemory/cloud_shutdown.py <cloud_provisioning.json>` to
  stop a pod with retrieval first; it refuses to delete on failed retrieval
  unless `--force`. Run launchers and monitors from PowerShell so `ssh` is
  Windows OpenSSH; Git's MSYS `ssh` rewrites backslashes in remote
  arguments and broke two retrievals on 2026-09-15.
- Git stores files byte-exact (`* -text`, `core.autocrlf=false`) because
  workers verify SHA256 of pinned sources. Do not change that.
- Detached local processes launched from a Claude session died overnight on
  2026-09-15; if a run must survive the session, launch it as a scheduled
  task or service and check it before assuming it is alive.
- Preserved checkpoints worth knowing about: attention 8400 (the last
  warm-started model that solves the old battery; SHA
  `e37602aa…1bc9`), cloud 10000 of the biased five-task run
  (`35281f26…2942`), and the v2 cloud arm's 49 checkpoints under
  `WorkingMemory/AttentionContextComparator/V2/runs/cloud_20260915_222845/retrieved_user_stop/`.
  None of these should be a parent for new work under the rules above; they
  are references for the old battery and for post-mortem analysis.

## 8. Useful code to reuse

- `WorkingMemory/SpatialTaskBattery/stimuli.py`: the battery generator and
  streams (audit first, section 4).
- `PreAttentiveVision/evaluate_multitask.py` and
  `WorkingMemory/UnbiasedAttention/protocol.py`: scoring.
- `WorkingMemory/AttentionContextComparator/V2/worker.py`: a bounded worker
  with resume, per-update diagnostics and hash-pinned sources; strip the
  v2-specific diagnostics and reuse the skeleton.
- `WorkingMemory/AttentionContextComparator/V2/check.py`: the pattern for a
  hard identity gate between two model versions.
- `WorkingMemory/AttentionContextComparator/V2/preflight_probe.py`: forward-
  only linear probing of intermediate fields.
- `WorkingMemory/AttentionContextComparator/V2/cloud/`: provisioning,
  watcher, lifecycle and pull scripts for RunPod, if cloud is ever
  authorised again.
