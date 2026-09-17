# Temporal memory battery for the opponent model

2026-09-12. The user subsequently authorized implementation and training with "Lets proceed with the training". Researchers are implementing the battery. Run receipts and production metrics, rather than this status sentence, establish actual launch and completion.

Execution allocation: a new finite14400-second local wall-time cap starts with the first GPU profile and includes training, evaluation and reporting. Target40,000 complete training episodes if measured sequence costs fit, with approximately10% in an initial sensory/cue bridge and the remainder in the specified joint mixture. Preserve weights and compatible optimizer state across the bridge/joint transition. Fix the actual exposure after profiling; report it as exploratory, not a sufficient acquisition horizon. One GPU worker at a time, fp32, full episode gradients, no automatic cap extension. Earlier run remainders remain separate.

Launch receipt: `runs/wm_20260912_181219/`. Production launched after profiles with batch4 and9800 updates =39,200 fresh episodes, comprising3920 bridge and35,280 joint episodes. All408,728 learned parameters train; fixed temporal operators remain fixed. Planned evaluation checkpoints are980/3920/6860/9800 updates. The new cap began18:12:21 PDT, with hard deadline22:12:21 PDT. Estimated training plus evaluation is approximately two hours, not a completion guarantee. Run manifests/logs remain authoritative.

Latest user scope: use only the successful neuroscience-inspired opponent model. Hold KDA and ConvGRU aside unless a later result warrants revisiting them. Design sequence tasks spanning the seven established sensory domains; explicitly train all new cues, objects and rules. The aim is to measure what memory functions the current model supports and what an added working-memory computation would need to improve. A separate memory module is a hypothesis, not a predetermined conclusion.

## Model and training decision

Initialize from the selected opponent checkpoint at4032 updates in `PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/opponent_seed30301/`. Preserve that checkpoint and the completed experiment. Train all learned encoder, projection, temporal-output and decoder weights on the new battery. Starting from useful sensory weights does not require freezing them; a fresh random initialization is not necessary simply because the task has new inputs.

The two trace retentions (.25/.75), Gabor kernels, energy equations and pooling remain fixed in this first capacity measurement. This preserves the winning architecture while letting its learned representations adapt. No separate working-memory module or interleaved variant is added in this battery.

Implementation note: the current wrapper freezes parameters in its constructor, forces encoder.eval() in train(), and executes the encoder under torch.no_grad(). A separately versioned sequence learner must parameterize/remove all three restrictions and build a NEW optimizer over all learned weights. Merely changing requires_grad is insufficient. Preserve parameter tensors; do not call a new optimizer an exact continuation of the old one.

Use full episode backpropagation initially, without detaching state at cues, event changes, blanks or distractors. Use sequence-length buckets and a profiled batch size that fits fp32 GPU memory. Exact activation recomputation is allowed to reduce saved activation memory; it must preserve the computation and gradients. Do not choose hidden-state truncation merely to make an artificial memory failure. The current execution allocation is stated above; actual run manifests fix exposure and deadline.

## Shared interface

- Input: one RGB100x100 frame per update. The only persistent model state is its fast/slow feature traces at the three spatial scales. For a batch the frame is [B,3,100,100]; persistent trace tensors are [B,32,50,50], [B,32,25,25], [B,32,13,13], twice each. No sample cache, previous logits, decision accumulator or externally retained cue embedding.
- Visual cues: fixed, trained pictograms for integration/remember instructions, polarity or target class, six item identities and report onset. Render compact cues in reserved10x10 corner patches, masked identically across labels even when neutral. Existing dots fit their central aperture; other full-frame stimuli get the same corner masks. All seven family adaptations are trained with these masks. Do not test unseen cue alphabets as a memory test.
- The usual family head may select four direction logits for motion and two logits for the other tasks. It never selects the relevant item, cue sign, answer, delay or latent phase. Rule and item cues are pixels. A decoder is allowed to read current pixels; design final frames so that this is insufficient for the memory question.
- Reset once before the complete episode. Cue, sample, delay, distractor, query and report frames all update the state. The loss is evaluated at the instructed response time; that loss mask is never fed to the network. Do not freeze updates during blanks or queries.
- Durations are counts of equally spaced observed frames/transitions, not biological milliseconds. Every visible hold is an input presentation. Motion labels count observed displacements, not invisible segment time.
- Train and test are fresh procedural episodes. Keep BSDS train/validation/test source-photo separation. Repeated presentations, changed probes from one memory set and difficulty evaluations of one history are not independent training episodes or test samples.

## Four related protocols, not four competing architectures

### A. Sensory and cue bridge

Before interpreting memory limits, teach the exact new stimulus/cue/probe formats at minimal delay and load1. Examples include a visibly selected CW versus CCW orientation increment, remembered-value matching, a contour-geometry match with fresh clutter, and the two-cohort continuous motion renderer. These are new trained tasks, not a demand for zero-shot transfer from the previous pair decoder.

Include cue-visible and cue-absent-after-instruction cases. A visible cue establishes conditional sensory performance; removing it additionally requires rule retention. Retain ordinary short two-frame sensory examples as a modest training anchor. This is one necessary acquisition stage, not a broad preliminary validation campaign. Failure at this level remains a sensory/cue-learning issue.

### B. Selective integration over a stream

Timeline: instruction cue -> evidence stream -> optional retention interval -> report marker. First establish no added retention interval; subsequently vary it through protocolC.

**Motion: total duration winner.** Let d_t be the cardinal direction of the transition into frame t. With fixed frame interval:

\[
T_k=\sum_{t=2}^{L+1}\mathbf1[d_t=k],\qquad y=\arg\max_{k\in\{R,U,L,D\}}T_k.
\]

There must be a unique winner. A direction can recur in several separate blocks; count TOTAL duration, not the longest single block. Example: R3,U5,R3,D2,L2 has totals R6,U5,D2,L2, so R wins even though U has the longest uninterrupted run and L occurs last. The numbers count transitions, not block endpoint frames. Net displacement is a different statistic and is not the label.

Extend the existing random-dot renderer with two alternating age cohorts, each visible for two frames. On each transition one cohort survives and moves in d_t, while the other is reborn at uniform positions. Preserve the circular aperture and periodic underlying domain; keep displacement magnitude constant within an episode and initially preserve coherent cardinal displacement. Merely concatenating independently rendered pairs would introduce unlabelled jumps between pairs.

Initial stream lengths:8,16,32 transitions. Keep a margin between largest and second-largest count; report by normalized margin (Tmax-Tsecond)/L. Counterbalance the winner against initial/final direction and construct diagnostic pairs with different winners but the same final direction, and where feasible the same net displacement. Avoid a predictable fixed direction order. Later64-transition episodes are length extrapolation, reported separately.

**Other scalar domains: accumulate changes matching the cue.** Let x_t be a physically rendered scalar and c in{-1,+1} the visible instructed sign:

\[
A_c=\sum_t\max(0,c\Delta x_t),\qquad y=\mathbf1[A_c>\Theta].
\]

The instruction displays the target sign and a threshold bar. Bar length maps monotonically to a domain-specific amount; this mapping and every threshold level used for the main test are trained. Use thresholds from the feasible range for each stream length, with labels balanced within cue, threshold and length. Exclude equality and sweep the evidence margin |Ac-Theta|. No numeric threshold or latent increment enters the model.

For orientation, the user's target-aligned event-detection example is the easy special case: at least one cued rotation exceeding a declared size. Repeated-event accumulation is the harder case. For instance, +10,-10,+10,-10degrees has zero net rotation but20degrees of CW evidence. Use signed axial increments of magnitude below90degrees, so direction between successive gratings is identifiable. Independent phase is allowed but the rotation signal itself must remain visually recoverable.

| Family | Evidence statistic for integration | What must be preserved or controlled |
|---|---|---|
| Motion | Total duration per cardinal direction; output the winning direction | Revisited directions; final direction and largest single block are unreliable |
| Orientation | Total CW or CCW angular travel, selected by cue | Cumulative path can differ with identical initial/final orientations |
| Contrast | Total increases or decreases in log modulation amplitude, selected by cue | Keep amplitudes within the valid renderer range; avoid clipping and amplitude/label shortcuts |
| Spatial frequency | Total increases or decreases in log2 spatial frequency | Include cancelling changes; match starting/ending frequencies in paired diagnostics |
| Chromatic increment | Total signed travel along the existing approximately equiluminant RGB axis | Keep RGB values valid and luminance nuisance balanced; cue selects one polarity |
| Contour | Total visible frames containing organized contour versus shuffled arrangement, selected by cue; compare with threshold | Same positions, element count and nuisance laws; do not label contour strength from jitter alone |
| Natural spectrum | Total sharpening or smoothing travel, x=-beta, selected by cue | Same source crop/phase within a stream; per-frame mean/RMS matching as before; bounded feasible beta |

For contour use Ac=sum_t 1[z_t=c], with z_t denoting the renderer's organized/shuffled condition. This is categorical occupancy, not a claim that humans see every generated organized frame equally well. Longer durations and easy jitter establish sensory access first.

Construct feasible scalar trajectories rather than clipping increments after assigning labels. Include return trajectories and evidence placed early/middle/late. For signed scalar comparisons, do not promise mathematically impossible complete marginal equality: measure cue-only/current-frame shortcuts and use matched endpoints, especially matched visible suffixes, as the decisive contrasts.

The primary cue is visible throughout the stream, so integration can first be separated from cue retention. A matched condition flashes the SAME cue only at the beginning. In both, all physical increments remain in the stream; opposite-sign events are not removed from the input. Initially retain one rule throughout an episode rather than adding task switching.

**Interpretation:** four duration counters can solve motion; one or two gated sums can solve the scalar tasks. These demonstrate evidence accumulation and selective computation, not multi-item sensory memory. Context-dependent integration is inspired by [Mante et al.,2013](https://www.nature.com/articles/nature12742); the present seven-domain protocols are adaptations.

### C. Delayed report and interference

After the evidence stream, insert D in{0,2,4,8,16} blank frames before the same report marker. Delay32 is a separate extrapolation condition initially. D counts extra blank frames; the report marker itself is also an update and must be included in total history age.

At an anchor delay8, replace blanks with visually active, irrelevant stimuli of matched duration, first from the same family. Distractors are generated independently of the label, and the trained instruction makes their irrelevance explicit. Compare equal temporal durations; never compare a long distractor condition against a shorter blank condition.

The sequence already determines the answer before the delay, so success can reflect retention of one decision bit or one of four categories. Name this **decision retention**. It does not establish storage of the original sensory stream.

### D. Retrospective probe, item load and binding

Timeline: remember instruction -> N identity-tagged sample items -> blank delay D1 -> identity retrocue -> D2 -> new probe/report. All items appear serially at the same central sensory location with distinct trained identity glyphs. This avoids making simultaneous spatial crowding the primary capacity manipulation.

Initial loads N=1,2,4; use all six identity glyphs during training even at low load. N=6 is a separately reported load extrapolation initially. Each sample receives the same short exposure (two frames; for motion, one informative transition), independent of N. Sample order and queried identity/serial position are balanced. Increasing N necessarily changes the age of earlier items: compare conditions at matched target age and include matched-duration one-item controls with irrelevant events occupying the other sample slots. Do not attribute a serial-position/retention-age effect solely to capacity.

The retrocue identifies the target AFTER encoding. D2=2 frames in the primary retrocue condition allows a period between selection and probe. D2=0 is the matched postcue comparison; it does not demonstrate internal reprioritization. For precue controls, identify the target before the samples but keep total target age, visual-cue energy and probe timing matched with neutral cues at the other cue time.

| Family | Sample content retained | Query/report |
|---|---|---|
| Motion | A cardinal direction associated with each identity | Report the retrocued item's direction, four classes; categorical feature retention, not continuous motion precision |
| Orientation | Item-specific orientation | Does a new Gabor match the cued orientation? Vary angular mismatch and independently resample phase |
| Contrast | Item-specific modulation amplitude | Does the probe match the cued amplitude? Hold nuisance geometry stable or vary it identically in trained bridge and memory tasks |
| Spatial frequency | Item-specific frequency | Match/mismatch on frequency with independently sampled phase |
| Chromatic increment | Item-specific color-axis coordinate | Match/mismatch on that coordinate with matched geometry and balanced nuisance |
| Contour | Item-specific contour path geometry | Same versus changed path, both organized contours with fresh clutter and matched element/extent statistics; this extends grouping to contour-shape memory |
| Natural spectrum | Spectral setting of an item-specific natural crop | Same versus changed beta on the same crop/phase, with mean/RMS matched; changing source crop is excluded initially |

Primary binary memory labels are0=match and1=change rather than higher/lower. Construct balanced A/A,B/B matches versus A/B,B/A nonmatches, sampling feature pairs or a balanced finite value grid. Resample nuisance independently in both classes. This permits exact matching of the relevant single-sample/probe marginals across labels. Use ordered higher/lower probes only as a separately declared later variant, since they can retain residual marginal cues.

In multi-item nonmatch trials, deliberately use another stored item's value on a balanced subset. These **binding lures** distinguish item confusion from generic feature mismatch. For motion, report the confusion toward uncued stored directions. For the six binary domains, the final probe's match status is unknown at encoding; the model cannot simply precompute the final yes/no answer before the delay. Motion is a coarser categorical recall test, explicitly labelled as such.

For natural stimuli, choose mean/RMS and any no-clipping gain from a label/trajectory-independent rendering rule or candidate set, never from the particular future selected frames/probe. A natural binding lure applies the uncued item's beta to the QUERIED item's base crop; it does not substitute a different photograph. This preserves the spectral-memory question and prevents source identity from answering it.

Anchor D1=4,D2=2 for the load comparison; anchor N=1 for the blank-delay curve, then N=2 for the interference comparison. Do not launch the full Cartesian product of all lengths, loads, delays, cue timings, thresholds and families. Train the conditions used for the principal comparisons; extrapolation tests are segregated.

The motivations are delayed sensory report, retrospective selection and binding/precision under load, informed by [Griffin and Nobre2003](https://pubmed.ncbi.nlm.nih.gov/14709235/) and [Bays and Husain2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2532743/). No fixed human four-item limit is imposed.

## Training and assessment

Train one opponent model on the new inputs and task rules, including the easiest instances of every probe type. After the bridge establishes those mappings, a suggested joint mix is10% sensory anchors,45% integration/decision-retention episodes and45% retrospective-probe episodes, with families balanced within each group. These percentages are a proposed initial allocation, not a scientific law; the old50% contour schedule is not automatically carried into a different task battery.

Short and long in-distribution sequences must both receive training, along with blank and explicitly irrelevant distractor intervals, persistent/transient cues and trained item identities. Test fresh episodes of trained combinations separately from longer lengths/delays or increased loads. Do not claim untrained-length failure is a measured in-distribution capacity ceiling.

Count complete episodes, optimizer updates, distinct sensory events, repeated frames, cue/probe/blank presentations and total encoder updates separately. Longer episodes cannot be called exposure-matched because update counts match. Use streamed procedural examples, not a fixed miniature bank. Acquisition curves and actual training exposure accompany every conclusion; no finite horizon is presumed sufficient in advance.

Report a capability profile rather than one overall score:

- Integration BA/AUC versus sequence length, evidence margin and temporal position; motion four-way confusion and count margin; wrong-sign distractor effects for cued tasks.
- Decision-retention BA versus delay, with matched blank/distractor comparisons.
- Retrospective recall/match BA, sensitivity/criterion, psychometric mismatch curves, identity-lure errors and serial position versus load and retention age.
- Cue-visible versus flashed-cue effects, and precue versus retrocue effects with matched timing.
- Fresh-episode uncertainty (paired when histories are shared; source-image clusters for natural stimuli) and explicit one-training-seed limits for an initial screen. Do not treat multiple probes of one memory set as independent episodes.

Generator oracles may compute exact totals/labels for correctness checking, but those statistics never enter the model. A last-frame/last-direction or recency heuristic is an analysis reference, not another trained model sweep. State-reset and cue-swap evaluations should be paired with controls for the changed input/state distribution. No guarantee of chance is asserted without checking the actual marginal information.

## When would an additional memory computation earn its place?

The current opponent state has no learned retention/protection gate and no recurrent feedback of decoded motion evidence. A trained encoder CAN condition its input features on a cue visible in the current image; it cannot access a remembered cue to modulate its per-frame computation. Separate visible-cue selection from retaining a flashed cue. For two histories followed by the SAME input suffix of D frames:

\[
\delta F_D=.25^D\delta F_0,\qquad \delta L_D=.75^D\delta L_0.
\]

This attenuation is architectural even when the encoder is trainable. It predicts a useful stress axis, not accuracy or an impossibility theorem: tiny residual differences can remain readable, and finite precision/readout amplification matter.

If immediate cue/sensory performance is good but matched delays, distractors or retrospective load cause reproducible deficits despite meaningful acquisition, we have a specific missing capability to address. A future memory addition must rescue that deficit under comparable training while retaining sensory performance. Altering the existing trace update is an alternative to adding a separate module. If the current model succeeds, record that capacity and increase one demand deliberately; do not engineer a failure to justify the planned component.

This battery measures operational memory functions before selecting a working-memory architecture. Interleaving accumulators remains a separate possible architectural experiment; it is not silently included here.

Research notes: [neuroscience rationale](Research/neuroscience_task_rationale.md), [sequence-generation constraints](Research/sequence_generator_notes.md).
