# Sequence-generator notes for the trained opponent model

Design only. No generators, models, training jobs or GPU diagnostics were implemented or run for this note. The latest user direction selects the successful opponent model; KDA and ConvGRU are held aside unless a later roadblock motivates revisiting them. This is a stress battery for that existing computation, not a proposal for a new working-memory architecture.

The integrated decision is in `WorkingMemory/TASK_BATTERY.md`: match/change is the primary binary retrospective probe, while cue-conditioned accumulated travel is a separate integration task. Higher/lower alternatives below explain the marginal-information constraint; they are not additional primary experiments. The integrated cue/threshold timeline and selected anchor lengths/delays supersede illustrative options in this supporting note.

The primary question is whether the pretrained opponent system can **learn the new sequence/cue tasks and retain the relevant information**. A frozen two-frame transfer test answers a different question. Neither a success nor a failure can, by itself, establish the necessity of a separately named working-memory module.

## Existing implementation facts that matter

Source inspected: `PreAttentiveVision/neuroscience_stimuli.py`, `natural_stimuli.py`, and `TemporalIntegration/accumulators.py`. The seven current heads have four classes for motion and two for each other domain. Current labels concern two-frame direction, signed orientation, higher contrast/frequency/chromatic coordinate, structured-contour interval, and higher natural-image spectral detail. Sequence extensions must version any changed label meaning explicitly; retaining two output units does not make a new label an old task.

The selected parent is `PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/opponent_seed30301/checkpoint_004032.pt`. Its learned spatial encoder, projections, opponent output adapters and readout are useful initial weights. Keep their provenance. For new-task learning, allow **all learned parameters** to train, as the current direction specifies. Keep the fixed quadrature filters and retentions separate from that decision.

There is a concrete implementation trap to address in a future version: the current classifier freezes encoder parameters in its constructor, forces encoder evaluation mode in `train()`, and wraps the encoder in `torch.no_grad()` in `_advance()`. Changing `requires_grad` alone will not unfreeze it. A future sequence learner must explicitly change those three constraints and include all learned parameters in its optimizer. It must preserve the old executable sources/checkpoints. A new optimizer for the new task is a warm start from trained weights, not a claim of bit-for-bit optimizer continuation; scratch training is not required for the primary experiment.

## Observable sequence contract

- Supply one current RGB100×100 image to the model at each update. Its only inference-time history is the three explicit `(fast, slow)` state pairs. The task identity may select the known task head, as in PAV; this convenience must be stated.
- Render selection cues, boundaries, queries and the report signal as actual images. Use simple, counterbalanced visual symbols outside the sensory aperture. Do not pass a latent phase, target index, timestep, direction count, cue interpretation, difficulty, label or future sequence length into the model.
- A cue visible alongside a sample is an easier, different condition from an initial cue that must be remembered and applied later. Fix one as the primary condition and name the other separately. Do not silently repeat an earlier cue at report time and claim its maintenance was tested.
- Score one terminal response per trial, after a visible report/query frame. All delay, distractor and cue frames advance state. A zero-loss frame is still an observed frame; it must not suspend the accumulator. Reset only at trial boundaries.
- Use a fixed simulation frame interval initially, reporting durations in update/frame units. If physical frame intervals later vary, duration information must be perceptually supplied; a hidden timestamp cannot determine the label.
- Independent distractors should be plausible stimuli from the same domain and independent of the target label. Their spatial extent, average intensity, duration and cue frequencies should not identify the answer. Compare blank retention and interference as different conditions, not interchangeable meanings of “delay.”

## Motion: total directional duration, not a displacement summary

Continue the current dot process over the whole evidence sequence: 256 domain dots, 128 surviving and 128 reborn each transition, lifetime two frames, the same periodic domain/circular aperture and surviving-dot coherence1. An extension must carry positions and ages across transitions. Concatenating independently generated two-frame pairs would introduce unlabelled resets and motion boundaries.

For evidence transitions `t=1…L`, let `d_t` be the cardinal direction that actually moves the surviving dots. With constant frame interval:

\[
T_d=\sum_{t=1}^{L}\mathbf1[d_t=d],\qquad y=\arg\max_{d\in\{R,U,L,D\}} T_d.
\]

Require a unique winner and report the top-two duration margin. Repeated visits to a direction contribute to its **total**, irrespective of how many segments they occupy. Each duration tick must contain a new coherent displacement; repeating a static image cannot count as visible motion in an unobserved latent direction. The initial dot frame plus `L` transitions requires `L+1` motion images. Blank/report transitions are outside the motion-evidence interval.

Use inherited1/2/3-pixel displacement, initially fixed within a trial. Balance displacement against winner and duration margin. A later speed variation should be independent of the label: duration is not displacement magnitude. Include sequences in which the total-duration winner has neither the longest single run nor the final direction.

An explicit matched pair of14-transition schedules demonstrates the desired distinction:

| Schedule | Direction sequence | Counts `(R,U,L,D)` | Winner |
|---|---|---|---|
| A | R U R U L R U L R U R L R D | `(6,4,3,1)` | R |
| B | R U D U R U L U R D U R U D | `(4,6,1,3)` | U |

Both have the same first directionR, final directionD, longest run1, total duration14 and nominal net directional displacement `(R−L,U−D)=(3,3)`. The latter is a population-motion summary, not a claim that a single short-lived dot traverses the whole sequence. Rotate/relabel and randomize these constructions so the counterexamples do not become a fixed bank or a memorized answer template. Supplement them with longer runs and examples whose longest run belongs to a losing direction.

Training needs varied schedules, not just these adversarial patterns. Held-out matched schedule families should separately control first/final direction, longest run, net vector, duration margin and temporal position. Do not infer that a last-window observer is excluded merely because it performs poorly on one chosen example.

## Orientation: define the cue-conditioned quantity explicitly

Recommended operationalization, for the integrated specification to accept or replace explicitly: a visible cue `c∈{−1,+1}` selects counterclockwise or clockwise accumulation. Let each observed orientation be axial, modulo180°. Use inherited change magnitudes4/10/22° and independently varied carrier phase. Define:

\[
\delta_t=\operatorname{wrap}_{(-90^\circ,90^\circ]}(\theta_t-\theta_{t-1}),\qquad
A_c=\sum_t\max(c\delta_t,0).
\]

All elementary changes are below90°, so their signed axial interpretation is unambiguous. At report, a visible query specifies a magnitude `q`; the binary answer is whether `A_c>q`, with exact ties excluded and a declared margin around the threshold. This is accumulated angular travel in the cued direction, not the signed net change. A fixed small set of visually encoded query values is adequate for the first version; query identity must not predict the answer.

Generate both answer classes within every cue/query/length cell. Use histories with identical initial/final orientations and different accumulated travel. Also include histories for which changing only the cue changes the correct response. If every path is monotonic, or every path has equal clockwise/counterclockwise travel, cue use is not tested appropriately. Never define the answer merely as the cue's direction.

An algebraic constraint matters: `A_+−A_-` is the unwrapped signed sum, and `A_++A_-` is total angular travel. It is impossible to hold both of those quantities fixed and vary `A_c` for a fixed cue. Therefore use separate counterfactual checks for endpoint shortcuts, cue independence and total-travel shortcuts; do not promise mutually inconsistent joint matching. Wrapped endpoints can match even when paths contain different numbers of rotations.

This label is a deliberate extension beyond PAV's two-frame signed-change label. If the intended question instead concerns a selected transition or selected item, define the selection event and its comparison directly rather than silently substituting final-minus-initial orientation.

## The other five sensory domains

For the first memory screen, use a visually marked target sample, intervening same-domain distractors, and a final probe/query. Keep stimulus difficulty and nuisance ranges visible in the report. The alternatives below are explicit task choices, not simultaneous requirements or an architecture sweep.

| Domain | Operational sequence extension | Preserve/control | Principal shortcut or interpretation limit |
|---|---|---|---|
| Contrast | Remember the marked sample's modulation amplitude; after interference decide whether the probe is higher/lower, or use the explicitly versioned match/change variant below. | Existing Gabor geometry, pedestal range and `.025/.06/.13` amplitude increments; matched mean, phase nuisance and gamut. | The current parameter is intensity modulation amplitude in `I=.5+c·g`, not Michelson contrast. Absolute probe amplitude and pedestal boundaries can partly predict higher/lower. |
| Spatial frequency | Remember the marked sample's log frequency and compare it with a delayed probe; use `.08/.18/.35` octave differences. | Shared base orientation/contrast distributions, independently varied phases, valid native raster frequencies. | Probe frequency itself can predict a directional comparison near distribution boundaries. Phase or exact pixel identity must not carry the answer. |
| Chromatic increment | Retain the marked sample's coordinate along the existing linear-RGB opponent axis and compare a later probe, with `.018/.045/.1` differences. | The declared axis and luminance weights, patch geometry, gamut and label-independent sensory noise. | These RGB coordinates are a display/model adaptation, not calibrated cone-isolating stimuli. Cue colors must not supply the tested color coordinate or reveal the label. |
| Contour grouping | Minimal categorical version: report whether the marked earlier image contained the aligned contour after later structured/scrambled distractors. A richer, separate version would compare remembered contour geometry with an always-structured probe. | Current32 elements, seven target-path elements, jitter2/8/16°, matched orientation multisets and independently sampled nuisance phase/clutter. | In the current two-frame task exactly one image is structured, so observing the final image can identify its interval. Break that complement rule across the new sequence. The minimal task needs only a remembered presence bit; it does not establish high-dimensional contour memory. |
| Natural-image detail | Retain the marked sample's spectral version of a BSDS crop; after distractors compare a delayed probe's relative high-frequency weight, or use match/change of spectral version. | Native100 crop, same target photo/crop/phase across the target comparison, current beta manipulation and mean/RMS/gamut policy; official photo splits. | A changing photo/crop can confound spectral memory with source texture. Repeated exact rasters permit an image-match solution; disclose that legitimate but narrower computation. Fresh crops are not new source photos. |

For contour geometry memory, both probe classes must contain contours. Resample nuisance phases and clutter so exact whole-raster matching is not the sole cue. That introduces invariance learning as well as memory; report it accordingly. Do not call success proof that the current association-field-like grouping computation was necessary.

**Scalar-comparison marginal limitation.** For strict higher/lower labels, both sample and probe marginals cannot each be exactly identical across labels while every positive trial has `probe−sample>0` and every negative trial has the opposite: their conditional expected differences would then have to be both equal and opposite in sign. Broad overlapping pedestals and counterbalancing reduce shortcuts but do not make that impossibility disappear. Include matched-probe histories with opposite answers, report the performance of the no-history condition, and stratify by pedestal/query value.

If exact one-frame marginal counterbalance is the priority, explicitly choose match/change: build both sensory variantsA/B for every trial; no-change uses A/A or B/B and change uses A/B or B/A, with the assignments equally sampled. Distractors and final cues must follow the same laws in both labels. Re-render label-independent sensory noise rather than using an identical-noise fingerprint. For natural images, compute any shared safety scaling from the same complete A/B candidate set for both labels; otherwise label-dependent normalization itself may leak change. This changes the task question and requires new-task learning—it is not a covert relabeling of the old reported result.

For natural integration sequences, shared photometric scaling must likewise depend on a predeclared candidate set independent of the selected future trajectory, rather than the maximum deviation over whichever future frames happen to be chosen. In natural binding-lure probes, transplant the uncued item's spectral value onto the queried target's base crop; do not substitute another photograph. Sharing the base crop across item identities is an alternative when isolating glyph/value binding. Different item-specific photos provide an additional content-based retrieval cue, which must be disclosed if retained.

## The memory-route audit and the fixed trace test

Inference state in the current opponent implementation consists only of `F_t^s,L_t^s` at each of three scales. Its nonlinear energies and output/readout fields are computed from current traces and are **not fed back** into trace updates. There is no learned retention gate, previous-frame cache, stored task-head output, explicit count accumulator, or history-conditioned encoder input.

An unfrozen per-frame encoder can nevertheless condition its output on a cue visible in the **same image**, potentially suppressing distractor writes. It cannot use an earlier remembered cue to alter future encoding through a nonexistent feedback route. This distinction prevents an overstrong claim that input selection is impossible.

For two different histories followed by exactly the same pixel suffix, with fixed weights/stateless per-frame encoding during evaluation:

\[
\Delta F_{t+D}=.25^D\Delta F_t,\qquad
\Delta L_{t+D}=.75^D\Delta L_t.
\]

Every later recurrent advance counts inD, including a query/cue/go image; it need not be a blank. The slow multiplier is approximately`.3164` at4 updates, `.0100` at16, `.0001005` at32 and`1.01×10⁻⁸` at64. This is a structural equation, not a measured accuracy curve. Shared nonzero suffix inputs can cause fp32 history differences to disappear through rounding; measure that rather than assuming a universal cutoff.

Construct matched histories with opposite correct answers and an identical cue/query/go ending. Report relative and absolute trace separation, exact equality/zero-difference rates in fp32, output margins and task accuracy separately. A small nonzero trace may be amplified by a learned readout; no additive state noise is present in the current model. Consequently geometric decay does not prove an information-theoretic impossibility or a universal need for another module. Likewise, energy extracted at report time may carry sequence statistics without an explicit duration counter; test the resulting temporal weighting with schedule counterfactuals.

Keep any frozen zero-shot sweep labeled transfer diagnostics. The primary evidence about learnability comes after the pretrained model is allowed to learn the new cues, targets and sequence distributions. Keep fixed retention/filter coefficients explicit; changing them later is a distinct intervention, not ordinary parameter unfreezing.

Audit the collector as well as the model: no inference access to previous raw images/features, latent sequence summaries or cached predictions; no automatic resets at distractor/cue boundaries; no retained state between independent trials; no hidden state inside data normalization; no evaluation optimizer updates. BPTT activation storage during training is not an additional inference memory route. If a readout receives saved earlier features, document that as an external memory, not this accumulator alone.

## Training and evaluation exposure accounting

Design recommendation, not launch authorization: begin with a small declared range of evidence lengths and delays, for example evidence8/16/32 updates and blank delays0/4/16, rather than assuming two-frame weights should extrapolate to64-frame sequences without learning. Use one fixed training mixture across the seven domains. The old contour-focused allocation solved a different learning bottleneck; it is not automatically appropriate for the new battery. Initially balance task-level trial exposure and report changes only if separately justified by new results.

For evaluation, use a central condition plus separate stress directions—longer evidence, longer blank retention, same-domain interference and matched shortcut counterfactuals—rather than an uncontrolled full Cartesian product. Distinguish interpolation on trained lengths/delays from held-out extrapolation, such as64 updates. In particular, blank64 and evidence64 test different things.

- An episode is one fresh generated history/cue/query/answer. Report `N_trials`, one terminal loss per trial, and `N_visual_updates=Σ_i T_i`; also separate evidence transitions, sample/probe frames, cue/report frames, blanks and distractors. Motion has one more initial image than evidence transitions.
- Report fresh base histories separately from repeated cue/delay/counterfactual presentations. The same underlying history at several delays is paired evaluation, not several independent base episodes. Natural-image crops/history families share source-photo dependence.
- Equal trial counts do not mean equal computation for variable length. Record effective batch size in trials, microbatch size, accumulated gradients, frames per update and per-domain totals. Pad only for storage: padded positions after trial termination must neither advance state nor produce loss. Bucketing by length is simpler than silently treating padding as real blank time.
- Full BPTT through the valid sequence is the primary learning contract. With an unfrozen encoder, long sequences have a much larger activation cost than the previous frozen two-frame run. Choose microbatches and a finite later budget from actual throughput/memory when training is authorized. If truncated BPTT is used, state its gradient horizon separately from forward retention; a failure may reflect that learning restriction.
- Use separate fixed train/validation/test source seeds, task-local streams and explicit sampler/checkpoint versions. Split related histories/counterfactuals together. For BSDS, retain official source-photo partitions and cluster uncertainty by source photo; report new draws from reused held-out pools honestly.
- Select checkpoints on predeclared validation conditions, keep the final tests locked, and show task-by-length/delay performance, confusion/AUC, margin/difficulty and uncertainty. Repeated checkpoint looks, frames and noisy rerenders are not training-seed replication. Do not replace the hard task with a pooled score dominated by easier conditions.

The first actionable outcome is a measured map of where the adapted opponent computation retains, integrates, or loses the task-relevant information under these controls. That can guide the next build decision. It is not a test arranged to force a predetermined working-memory hypothesis to win.
