"""One-time, CPU-only assembly of the September 13 lab-journal baseline.

This documents existing evidence; it never imports the model runtime. Do not rerun
over subsequently edited journal pages. Future updates follow MAINTENANCE.md.
"""
from pathlib import Path
import datetime, hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'LabJournal'
(OUT / 'experiments').mkdir(exist_ok=True)
STAMP = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest = {}

def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    manifest[path] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return raw.decode('utf-8-sig')

def put(path, content):
    (OUT / path).write_text(content.strip()+'\n', encoding='utf-8')

def tables(text):
    return re.findall(r'(?m)^\|[^\n]+\n\|[-: |]+\n(?:\|[^\n]+\n?)+', text)

entries = [
('01-pav-encoder-screen', 'Five convolutional encoders: the first corrected sensory screen',
 'PreAttentiveVision/runs/multitask_20260912_141316/report.md',
 '''**Question.** Which compact shared convolutional encoder learns a useful representation across seven two-frame sensory tasks? The user corrected the initial task to four cardinal directions of random-dot motion, grounded in Krauzlis-style stimuli. The original binary/static dot benchmark was stopped and has no locked final test. Its surviving artifacts are superseded diagnostics within this post-reset project, not a completed comparison.

**Design and ancestry.** Five new compact encoder adaptations—anti-aliased ResNet, ConvNeXt-GRN, InceptionNeXt, MobileNet-SE and VOne-ResNet—used a common ordered decoder. Each frame is encoded separately with shared weights. Signed differences and local displacement correlations preserve motion direction; a swap-symmetric change decoder cannot distinguish reversed motion. Encoder weights are not shared between candidate models.

**Exposure and selection.** One seed per model; 756 updates × 32 pairs = 24,192 pairs each, only 3,456 per task. All selected checkpoints were update 756 using validation mean task AUC. Final tests had 448 matched pairs per task; natural examples cluster by source photograph.

**What we learned.** ConvNeXt-GRN was the best broad fitted candidate here, with 80.6% motion but only 60.3% contour. MobileNet-SE and VOne-ResNet were stronger on contour. This was a practical short acquisition screen, not a reason to commit permanently to one architecture. Some low accuracies coexisted with useful AUC: poorly placed class decisions do not prove absent information.

**Decision.** Preserve all candidates, investigate complementary contour errors, and continue the useful learned ConvNeXt lineage. High-level neuroscience resemblance did not reliably predict performance: VOne's Gabor front end had 23.9% motion in this run. That is not evidence against biological oriented filters generally.''', [0,1],
 ['PreAttentiveVision/research.md','PreAttentiveVision/krauzlis_stimulus.md','PreAttentiveVision/two_frame_task_research.md','PreAttentiveVision/results_multitask.json']),
('02-saved-ensemble-audit', 'Can complementary encoder errors be combined cheaply?',
 'PreAttentiveVision/combination_analysis.md',
 '''**Question.** Do the existing probability vectors contain useful complementary errors before we spend GPU time combining components?

**Design.** A CPU-only saved-score analysis fixed equal-weight ConvNeXt+MobileNet and ConvNeXt+VOne ensembles. Validation on motion and contour selected MobileNet as the partner before loading test predictions. No encoder inference, optimizer update or new training examples were used. Runtime was 4.86 seconds.

**Result.** ConvNeXt+MobileNet increased contour from the stronger constituent's 71.7% to 74.8%, but lowered motion from 80.6% to 70.8%. On contour it fixed 19 MobileNet errors and broke five correct answers; on motion it broke 48 correct ConvNeXt decisions while fixing only four.

**Interpretation.** Error complementarity is useful evidence for a combination experiment, but indiscriminate probability averaging destroys part of the strong model's performance. An oracle that selects a correct constituent using the true label is not a deployable model. Neither oracle accuracy nor ensemble improvement identifies which feature computation caused complementarity.

**Decision.** Test small learned feature additions against ordinary continuation. This reused already inspected test scores, so it is exploratory development evidence rather than a fresh confirmation.''', [0,1], ['PreAttentiveVision/combination_analysis.json','PreAttentiveVision/component_combination_research.md']),
('03-hybrid-continuation', 'Gabor and late-SE additions did not solve contour',
 'PreAttentiveVision/runs/hbrids_20260912_152427/report.md'.replace('hbrids','hybrids'),
 '''**Question.** Can a compact oriented-feature branch or channel gate reproduce the useful properties suggested by the ensemble audit?

**Design and ancestry.** Three siblings start from the exact trained ConvNeXt step-756 parent: unchanged continuation, a fixed-Gabor residual branch, and a late-SE residual gate. Compatible weights, Adam states and sampler progress are preserved. The two additions initialize to the parent function. Architecture additions and their subsequent learning are the interventions.

**Exposure.** Every arm receives 756 new updates / 24,192 new pairs; all select global step 1512. New paired final examples also evaluate the frozen parent. The predeclared useful-change screen required at least +3 pp contour BA against continuation, with motion regression no worse than −2 pp.

**Result.** Neither addition passed. Gabor contour was 58.7% against control 63.8%; late-SE was 63.4%. Continued training alone improved the frozen parent's contour by 12.50 pp. Both additions reached almost perfect motion decisions, but control motion ranking was already nearly perfect.

**Decision.** Do not claim that copying plausible components solved grouping. Keep late-SE as a useful low-cost candidate and test whether the remaining weak task simply needs more of the existing training budget. The negative result concerns these initialized additions and exposure, not every Gabor/SE design.''', [0,1], ['PreAttentiveVision/results_hybrids.json','PreAttentiveVision/results_hybrids_analysis.json']),
('04-contour-task-allocation', 'Contour succeeds when it receives enough training allocation',
 'PreAttentiveVision/runs/allocation_20260912_160414/report.md',
 '''**Question.** Is contour weakness caused partly by how the common training budget is distributed?

**Design and ancestry.** Both arms start from late-SE step 1512 with identical architecture, weights and Adam state. Uniform training gives each of seven tasks 1/7 of updates; focused training gives contour 1/2 and each other task 1/12. Explicit new task-local RNG streams provide matching per-task prefixes, rather than pretending to replay the old single-stream sampler.

**Exposure and selection.** Profiled costs pin 756 additional updates / 24,192 pairs per arm. Uniform gets 3,456 contour pairs; focus gets 12,096, with 2,016 per other task. Both select terminal step 2268 by validation minimum task BA, then mean task AUC. A fresh test has 448 paired examples per task.

**Result.** Contour rises 67.9% → 96.2% with a paired +28.35 pp interval [23.91, 32.59]. AUC rises .771 → .994, showing an improvement in discrimination rather than just an argmax shift. All seven task point estimates exceed 95%. Contrast falls 1.56 pp, and its interval extends beyond the descriptive two-point preservation tolerance.

**Decision.** Adopt focused step 2268 as the PAV reference. No new architecture was necessary to close this particular contour gap. The task schedule is the experimentally changed package; gradient conflict or a specific optimization mechanism was not identified. Early focused motion temporarily fell before recovering, so a single bad early checkpoint would have given the wrong conclusion.''', [0,1,2], ['PreAttentiveVision/results_allocation.json','PreAttentiveVision/results_allocation_analysis.json','PreAttentiveVision/next_experiment_task_allocation.md']),
('05-causal-temporal-accumulators', 'One frame at a time: opponent traces beat the tested KDA and ConvGRU fits',
 'PreAttentiveVision/TemporalIntegration/report.md',
 '''**Question.** Can a causal fixed-size state replace direct access to two encoded images while retaining the sensory capabilities?

**Design and ancestry.** The selected contour-focused encoder at step 2268 is frozen. New common input projections/readout train around three competitors: spatial KDA, ConvGRU, and fast/slow opponent traces with fixed quadrature energies. Each processes the current frame and previous state, never both original encodings at once. The preserved original pair system remains a reference.

**Exposure.** Each new temporal model receives 4,032 updates × 32 = 129,024 fresh pairs under the contour-focused schedule. All select terminal 4032. Their encoder gets zero new updates; projection, temporal output and readout learning still has substantial representational freedom. The opponent has 141,968 trainable parameters in this specific frozen-encoder experiment, not in later fully trainable memory models.

**Result.** Opponent scores 98.44–100% across seven tasks, including 100% motion. KDA and ConvGRU motion are 31.70% and 75.67%. Resetting opponent history before frame two lowers motion to 25.89%; reversed frames with transformed labels yield 99.78%.

**Decision.** Continue with the opponent winner, holding other cores aside. History use and order sensitivity are established for these two-step tasks. They do not establish long-duration working memory, an MT-equivalent circuit, or inability of KDA/ConvGRU to improve with different training. Opponent contour gains over the pair reference also include new readout and extra training, so they do not isolate energy-channel necessity.''', [0,2,3], ['PreAttentiveVision/TemporalIntegration/README.md','PreAttentiveVision/TemporalIntegration/accumulators.py','PreAttentiveVision/TemporalIntegration/results_temporal.json','PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/completion_receipt.json']),
('06-sequence-battery', 'The broad sequence battery did not isolate memory capacity',
 'WorkingMemory/report.md',
 '''**Question.** Does the existing sensory opponent model support integration, retaining a decision, and remembering an item until a later query/probe?

**Design.** Extend all seven sensory families with sequence rules, visible/flashed cues, integration lengths, delays, distractors and retrospective probes. Warm-start opponent4032 and allow all 408,728 learned parameters to adapt; fixed trace coefficients and energy equations remain fixed. No separate working-memory module is added.

**Exposure and selection.** 39,200 fresh episodes / 9,800 updates were completed. Validation selected update 6,860 after 27,440 episodes; terminal exposure must not be attributed to that selected checkpoint. Each final cell has 128 held-out episodes.

**Result.** Sensory anchor ranking remained strong (mean AUC .9964), while integration/decision and retrospective cell means were .5154 and .5184. Several simplest new-rule conditions already stayed near chance. Thus “minimal-delay rules remained weak” means the model often could not perform the newly instructed task even when almost no waiting was required.

**Decision.** Focus the next comparison on learnable motion-duration and one-item orientation tasks with explicit recurrent memory. Longer-delay failure here cannot be cleanly called a retention limit because the baseline rule was not consistently acquired. This broad battery is retained as a recorded negative acquisition result, not silently discarded or rewritten.''', [0,1,2,3,4], ['WorkingMemory/TASK_BATTERY.md','WorkingMemory/results.json','WorkingMemory/completion_receipt.json']),
('07-lstm-versus-ei', 'Focused recurrent memories learn the rules; LSTM leads on motion duration',
 'WorkingMemory/RecurrentComparison/report.md',
 '''**Question.** Does an added recurrent population improve focused sequence learning, and how does a conventional gated baseline compare with a rate/adaptation circuit?

**Design and ancestry.** Both inherit the selected sequence6860 sensory model. A common 64×13×13 → 8×13×13 → 128 interface feeds either a 256-cell normalized LSTM or dense 256-unit E/I adaptive network. Each adds 512 state scalars. All learned sensory parameters remain trainable at lower learning rate. There is no attention or frame-growing cache.

**Exposure and optimization.** Both receive 40,000 new episodes / 5,000 updates on matched six-cell streams and select step 5000. LSTM trains locally; E/I is explicitly moved to a cloud RTX3090. Gate normalization, initialization, sign constraints and clipping are documented in the source; frequent clipping in both arms is an optimization qualification. Cross-platform differences preclude a pure controlled hardware comparison.

**Result.** Both learn the new tasks. LSTM scores 79.69% on eight-transition motion versus E/I 68.95%, a paired 10.74 pp gap. Delayed orientation is 83.98% versus 85.74%; the small E/I advantage is uncertain. Resetting the added recurrent state before every frame collapses motion choices toward chance.

**Decision.** Investigate what the E/I state retained rather than invent a new circuit immediately. Neither an LSTM “counter” nor a “carousel” was explicitly implemented; learned gating can accumulate evidence, and behavior alone does not specify its internal algorithm. This is a comparison of fitted systems, not a biological ranking.''', [0,2,4], ['WorkingMemory/Research/recurrent_memory_without_attention.md','WorkingMemory/RecurrentComparison/model.py','WorkingMemory/RecurrentComparison/analysis.json','WorkingMemory/RecurrentComparison/cloud_cleanup_receipt.json']),
('08-recency-diagnostic', 'Ordering sensitivity: the E/I model benefits from later winning evidence',
 'WorkingMemory/RecurrentComparison/RecencyDiagnostic/report.md',
 '''**Question.** Is the E/I duration deficit associated with the order of evidence even when the correct answer and final visual suffix are fixed?

**Design.** Freeze both selected step-5000 models. Construct 512 early/late matched pairs of continuous eight-transition dot movies. Match the complete direction count vector, winner, count margin, first direction, final two directions, total switch count and exact final two evidence rasters. Regenerate movies with restored nuisance RNG; do not shuffle images into physically discontinuous motion.

**Result.** E/I improves 72.27% → 78.52% when winning evidence occurs later: +6.25 pp [2.73, 9.58]. LSTM changes 85.74% → 84.18%, with an interval spanning zero. The model-by-order interaction is +7.81 pp [3.90, 11.53].

**What it identifies.** E/I is more sensitive to this controlled reordering. Moving the winner also changes distractor placement and local run structure; this does not uniquely identify leak, adaptation or saturation. The test is a selected schedule bank, not the earlier random six-cell distribution. Direction rotations are not independent schedule templates.

**Decision.** Test recoverability of early evidence from frozen E/I states. This diagnostic took 71.95 seconds with zero training and no cloud restart. It justified a question, not an automatic slow-synapse modification.''', [0], ['WorkingMemory/RecurrentComparison/RecencyDiagnostic/README.md','WorkingMemory/RecurrentComparison/RecencyDiagnostic/summary.json','WorkingMemory/RecurrentComparison/RecencyDiagnostic/completion_receipt.json']),
('09-state-accessibility', 'Earlier motion evidence remains in the firing-rate population',
 'WorkingMemory/RecurrentComparison/StateDiagnostic/report.md',
 '''**Question.** Is early motion information missing from the E/I state, or is the deployed output failing to use it?

**Design.** Freeze EI5000. Fit analysis-only ridge and small MLP readouts using 3,072 training movies, 768 validation and 1,024 held-out movies. Canonical templates and rotations are disjoint between splits. The duration probe sees sensory128 plus firing rates256; adaptation is not needed for this rescue. Count probes target early directional evidence through independent count coordinates.

**Result.** A linear duration readout improves deployed 68.65% to 78.13%, +9.47 pp [6.84, 12.11]. The MLP yields 77.64%, no observed advantage. Early-count paired-difference R² is .875 at transition four and .800 at final firing rates, while final sensory features alone give .002.

**Interpretation.** Useful early evidence remains accessible through the ordinary firing-rate route. The deployment problem includes fitting or use of the representation; changing recurrence is not yet necessary. A successful probe is constructive evidence of accessibility. A failed probe would not establish erasure.

**Decision.** Defer slow excitatory synapses and refit the existing output only. These controlled suffix-matched movies and raw-score AUC are distinct from the normal task stream and earlier softmax AUC. A Windows int32 target issue was corrected to int64; saved features and failed-attempt evidence were preserved. Total successful supervisor time was 247.40 seconds within the 1,200-second allowance.''', [0,1], ['WorkingMemory/RecurrentComparison/StateDiagnostic/summary.json','WorkingMemory/RecurrentComparison/StateDiagnostic/completion_receipt.json']),
('10-readout-refit', 'An existing output-only refit recovers much of motion performance',
 'WorkingMemory/RecurrentComparison/ReadoutRefit/report.md',
 '''**Question.** Does the diagnostic readout finding transfer into an improvement of the existing deployed computation without changing its architecture?

**Design and ancestry.** Continue EI5000 but train only its existing memory_output and motion/orientation heads: 33,670 parameters. Keep encoder, sensory traces, memory input and recurrent computation fixed in evaluation mode. Preserve compatible weights, Adam, sampler and RNG through a recorded training-policy migration. Frozen upstream computation avoids unnecessary sequence BPTT.

**Exposure.** 38,720 fresh standard-task episodes; validation selects terminal global step 9840. Final tests pair 512 examples in each of the six existing cells against the untouched parent.

**Result.** Eight-transition motion improves 70.31% → 79.30%, a paired +8.98 pp [5.86, 12.11]. Orientation and anchors remain high without a clearly detected large cost. All upstream parameters remain exactly unchanged.

**Decision.** Retain the refitted model and test blank-period retention. This demonstrates a useful output-fitting limitation in the parent; it does not establish superiority over equally exposed end-to-end continuation. It also does not establish equality to the old LSTM score, which used a different held-out draw. The probe from the preceding study was not silently installed as a new architecture.''', [0], ['WorkingMemory/RecurrentComparison/ReadoutRefit/results.json','WorkingMemory/RecurrentComparison/ReadoutRefit/analysis.json','WorkingMemory/RecurrentComparison/ReadoutRefit/completion_receipt.json']),
('11-retention-learning', 'Retaining a motion judgment improves; delayed orientation comparison remains weak',
 'WorkingMemory/RecurrentComparison/Retention/report.md',
 '''**Question.** What happens when the existing model must process 0, 4, 12 or 24 blank frames after the relevant evidence?

**Design and ancestry.** Start refit9840. Motion asks which of four directions occupied the greatest total duration in eight transitions, followed by D blank frames and report. Orientation shows a sample, D blanks, an identity query and a comparison probe. The renderer's post_delay is query-to-probe and is fixed to zero; it is not the manipulated sample retention interval. Match underlying evidence/probes across delays.

**Training.** 39,680 fresh episodes; select terminal global14800. Train the existing recurrent core and outputs, with sensory/opponent/memory_input fixed. The primary delays are trained, not extrapolated. No new architecture or attention is added.

**Result.** Motion becomes about 79–80% across all four delays. At D24 the parent moves from 25% to 80.08%. Orientation reaches 95.70% at D0, 65.82% at D12, and 49.02% at D24.

**Clear reading.** On about one in five motion trials the model chooses the wrong duration winner even without inserted blanks. Waiting through 24 blanks adds little observed error. Orientation differs: the answer depends on a later probe, so a previously computed category alone is insufficient.

**Decision.** Diagnose whether orientation is inaccessible, transformed, or badly compared. Parent motion D24 has AUC .789 despite 25% choices, so chance decisions cannot be called erased information. This trained near-flat motion curve is not a guarantee of perfect retention or proof of distinct biological stores.''', [0], ['WorkingMemory/RecurrentComparison/Retention/results.json','WorkingMemory/RecurrentComparison/Retention/analysis.json','WorkingMemory/RecurrentComparison/Retention/completion_receipt.json']),
('12-orientation-accessibility', 'Orientation survives the delay but the ordinary comparison fails',
 'WorkingMemory/RecurrentComparison/OrientationDiagnostic/report.md',
 '''**Question.** Does the long-delay orientation failure reflect absent sample information, an unstable decoding map, or ineffective probe comparison?

**Design.** Freeze Retention14800. Use 2,048 probe-training, 512 validation and 512 independent test base episodes, each paired across four delays. Decode the actual rendered axial angle, not the latent stimulus level: random context angle makes that shortcut invalid. Fit train-only scaling and validation-selected linear/MLP probes. Compare pre-probe firing rates with separately encoded probe images.

**Result.** At D24 a delay-specific linear readout recovers orientation from firing rates with 4.25° mean absolute error. The sample-trained decoder transferred to those states instead has 44.96° error. A label-trained comparator without angle inputs reaches 73.44% against deployed 50.00%; a separate angle-supervised diagnostic route reaches 87.30%. A post-probe label readout reaches only 52.73%.

**Interpretation.** The firing-rate state contains useful orientation information before the probe. Its useful decoding map is not stable across time; scale/offset changes can contribute. The operational comparator rescue shows that extra angle supervision is not necessary to demonstrate accessible comparison information. Post-probe probe failure does not prove erasure or rule out nonlinear information.

**Decision.** Investigate the memory/comparison interface and the user's spatial-memory competitor. Preserve the original model. The smallest 7.5° changes remain difficult even for the angle-based diagnostic comparator; this is not a universal replacement readout.''', [0,1,3], ['WorkingMemory/RecurrentComparison/OrientationDiagnostic/summary.json','WorkingMemory/RecurrentComparison/OrientationDiagnostic/paired_analysis.json','WorkingMemory/RecurrentComparison/OrientationDiagnostic/completion_receipt.json']),
('13-spatial-memory', 'Keeping a spatial field helps binding but does not solve every task',
 'WorkingMemory/SpatialComparison/report.md',
 '''**Question.** Does retaining a 64-channel 13×13 memory field avoid the information bottleneck of compressing everything into a vector?

**Design and ancestry.** Start compatible sensory weights from Retention14800. Compare a dense E/I memory and a convolutional E/I rate/adaptation field. Both receive an explicit learned old-memory/current-sensory comparator before classification. Dense inherits its trained core; spatial initializes a new core. All learned components may train, with lower sensory LR.

**Tasks and exposure.** Existing single-item orientation, new two-location orientation preserve/swap binding at D0/4/12/24, and a 10% motion-duration allocation. Both get 35,200 episodes / 4,400 updates and select terminal4400. Motion is excluded from the eight-primary-cell AUC checkpoint selection. Four additional binding-center evaluations measure interpolation.

**Result.** Spatial binding D24 is 93.36% against dense49.61%, and unseen-center binding is 94.53%. Single D24 remains near chance in both. Spatial motion D0/D24 drops to 46.68%/25.00%, versus the same-draw untouched parent's 79.10%/77.54%.

**Interpretation.** Spatial organization is a useful binding candidate, not a general fix. Binding is not a harder two-item version of native orientation: rendering, change sizes and query differ; a full swap can be detected by storing just one location. Spatial has 42.25× the dense recurrent-state scalars, with different geometry and prior experience. Locality is not isolated.

**Decision.** Preserve the motion-competent parent and spatial candidate, investigate selective maintenance without changing teaching. The major motion regression begins here, before the later attention module.''', [0,1], ['WorkingMemory/Research/spatial_ei_memory.md','WorkingMemory/SpatialComparison/model.py','WorkingMemory/SpatialComparison/analysis.json','WorkingMemory/SpatialComparison/completion_receipt.json']),
('14-selective-maintenance', 'Additive controller feedback gives a narrow gain, not a maintenance explanation',
 'WorkingMemory/SelectiveMaintenance/report.md',
 '''**Question.** Can a learned recurrent controller preserve relevant memory through blanks while teaching stays exactly the same?

**User correction.** A proposed delayed orientation-report objective with explicit angle supervision was rejected before implementation. The experiment therefore changes architecture only, retaining existing tasks, cues, labels, losses, heads and the 90/10 schedule.

**Design and ancestry.** Both arms continue spatial4400. Control is unchanged; competitor adds a 32-unit E/I controller with rate/adaptation states observing full sensory/memory summaries. It supplies additive previous-state currents to memory, with no cue crop, blank oracle, hold/forget gate or classifier shortcut. Added package: 45,537 parameters and 64 state scalars.

**Exposure and result.** Both complete 32,000 additional episodes and select global8400. Single D24 rises 58.40% control → 64.26% feedback, but single D12 and motion worsen. Silencing only controller-to-memory currents during the 24 blanks changes single D24 by −0.20 pp [−1.17, .78] and no binding/motion choices.

**Interpretation and decision.** The gain does not require ongoing additive feedback into spatial memory during blanks under this intervention. Controller state continues updating and can influence later frames, so this does not remove every controller contribution. Test pre-update joint sensory/memory attention as the user's next alternative.

**Execution note.** A Windows status-file sharing violation interrupted the original supervisor, not completed training. A replacement adopted the independently completed validation and resumed fixed remaining jobs without repeating training or extending the deadline. Recovery evidence remains preserved.''', [0], ['WorkingMemory/SelectiveMaintenance/README.md','WorkingMemory/SelectiveMaintenance/analysis.json','WorkingMemory/SelectiveMaintenance/completion_receipt.json']),
('15-preupdate-attention', 'Joint attention substantially improves delayed orientation, with a motion cost',
 'WorkingMemory/PreUpdateAttention/report.md',
 '''**Question.** Before the next recurrent update, can the model select between incoming sensory information and its own remembered information?

**Design and ancestry.** Start the same spatial4400 parent as ordinary continuation and additive feedback. Previous spatial memory supplies 169 queries. Current sensory plus previous memory supplies 338 keys/values. Two heads of dimension32 may attend to either source; they are not hardwired “visual” and “memory” heads. The attended field replaces raw sensory drive into the existing E/I memory input. Original sensory and old-memory/current-sensory comparison routes remain.

**Exposure.** 32,000 additional episodes / 4,000 updates, selecting global8400. Tasks, teaching, 90/10 schedule and compatible Adam state stay unchanged. The added module has 27,590 parameters, total model556,128, with no added persistent state. Cloud RTX3090 versus local controls is a disclosed platform difference.

**Result.** Single orientation D24 rises from continuation58.40% to attention79.30%, a paired +20.90 pp [15.82,25.98]. Binding stays near ceiling. Immediate motion falls 50.78% → 39.65%, while motion D24 is 35.35% against31.64%. It is a substantial specific gain, not an all-task winner.

**Decision.** Preserve the advance and investigate the regressions. Frozen phase interventions and a saved-score motion audit follow in pages15a/15b. The old cloud pod was retrieved and deleted; it is separate from the new TrainingExposure pod. High attention mass by itself cannot demonstrate useful memory content or biological attention.''', [0], ['WorkingMemory/PreUpdateAttention/model.py','WorkingMemory/PreUpdateAttention/analysis.json','WorkingMemory/PreUpdateAttention/retrieval_receipt.json','WorkingMemory/PreUpdateAttention/cloud_cleanup_receipt.json']),
('15a-attention-mechanism', 'The orientation improvement depends on memory-source routing during blanks',
 'WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md',
 '''**Question.** When is memory-source attention used, and does changing evidence-period routing recover motion?

**Design.** Freeze attention8400 and continuation8400. Reuse 512 paired orientation and 512 paired motion episodes; D0 orientation uses128. An analysis wrapper excludes memory keys/values separately during sample, inserted blanks, identity query or probe. Previous-memory queries and ordinary E/I recurrence remain. Moving-frame motion also tests replacing attention drive with raw sensory H. No main-model training occurs.

**Result.** Excluding memory source only during D24 blanks lowers orientation79.30% → 50.00%, −29.30 pp [−33.01,−25.78]. Sample, query and probe exclusions show no comparable cost. For motion, memory exclusion improves D0 by3.12 pp; raw sensory bypass improves D0 by3.91 pp and D24 by5.86 pp.

**Interpretation.** The learned blank-period routing matters. Exclusion both removes recirculated values and renormalizes onto blank visual input, so active refresh and rejection of irrelevant drive remain entangled. Acute bypass is out of distribution and cannot prove that training with it would retain orientation gains.

**Decision.** Examine allocation and class decisions before adding new recurrence. These small routing rescues and the calibration rescue on page15b cannot be arithmetically added: they are different interventions on overlapping errors. The model's sensory branch already contains opponent history, not only current-frame information.''', [0], ['WorkingMemory/PreUpdateAttention/MechanismDiagnostic/results.json','WorkingMemory/PreUpdateAttention/MechanismDiagnostic/branch_statistics.json','WorkingMemory/PreUpdateAttention/MechanismDiagnostic/completion_receipt.json']),
('15b-motion-audit', 'Motion errors include decision bias and predate attention',
 'WorkingMemory/PreUpdateAttention/MotionAudit/report.md',
 '''**Question.** How much of weak motion accuracy reflects class-score bias, and what did the actual training schedule prioritize?

**Design.** CPU-only reuse of saved predictions, without Torch/model inference. Audit all four trained models on the same512 motion bases. Fit three effective zero-sum class offsets using only128 existing D0 validation episodes, fixed regularization, then evaluate saved D0/D24 scores. This diagnostic fit is not installed in the trained checkpoint.

**Result.** Attention predicts right/up/left/down 13/348/151/0 times at D0: it never chooses down despite down-versus-rest AUC .738. Calibration improves39.65% → 52.54%, +12.89 pp [8.59,16.99], but D24 transfer is small and uncertain. Continuation also benefits from calibration and still has stronger discrimination.

**Historical finding.** The earlier separately paired spatial experiment already showed a major motion loss. Attention cannot explain that earlier decline. The current 90/10 runs gave each motion delay only1,600 episodes and excluded motion from checkpoint selection. Logged gradients are finite; clipping declines in all three arms, with no unique evidence of attention-only numerical collapse.

**Decision.** Run a controlled equal-total-update allocation comparison: continued10% versus50% motion, preserving architecture and protecting every task in selection. This is the current Stage1. The audit used previously inspected tests and fit offsets conditional on a small validation set; it is exploratory, not a fresh benchmark proving calibration solves the system.''', [0,1,2], ['WorkingMemory/PreUpdateAttention/MotionAudit/findings.json','WorkingMemory/PreUpdateAttention/MotionAudit/completion_receipt.json']),
]

for i,(slug,title,source,narrative,chosen,extra) in enumerate(entries):
    src=read(source)
    ts=tables(src)
    result_tables='\n\n'.join(ts[j].strip() for j in chosen if j<len(ts))
    links=[source]+extra
    for p in extra:
        # Hash short evidence summaries/source, without checkpoint deserialization.
        if (ROOT/p).is_file() and (ROOT/p).stat().st_size<15_000_000: read(p)
    evidence='\n'.join(f'- [{p}](../../{p})' for p in links)
    put(f'experiments/{slug}.md',f'''# {title}

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: {STAMP}. Source experiment dates and costs are retained in the linked run records.

{narrative}

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.

{result_tables}

## Evidence

{evidence}

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
''')

index='\n'.join(f'| {slug.split("-")[0]} | [{title}](experiments/{slug}.md) | Complete |' for slug,title,*_ in entries)
put('README.md',f'''# Visual Attention and Working Memory — lab journal

This is the project's research wiki: what we built, what actually happened, why the next experiment followed, and what remains uncertain. It covers extant work after the user-authorized repository reset, through the dated snapshot below. Deleted pre-reset projects and plans have not been recovered.

**Current question:** can additional motion training recover the pre-update attention model's motion accuracy while preserving its substantial delayed-orientation gain? The matched10% versus50% motion continuation is underway. See [current status](CURRENT_STATUS.md) and [Stage1 experiment](experiments/16-training-exposure.md); provisional validation is separate from final held-out results.

## Start here

- [Current state, capabilities and unresolved failures](CURRENT_STATUS.md)
- [Chronology: observations, decisions and corrections](CHRONOLOGY.md)
- [Current architecture: tensors, equations and what learns](ARCHITECTURE.md)
- [Tasks, timing, datasets and metric definitions](TASKS_AND_METRICS.md)
- [Open questions and what evidence would answer them](OPEN_QUESTIONS.md)
- [Research foundations and limits of biological claims](RESEARCH_FOUNDATIONS.md)
- [How to update this journal](MAINTENANCE.md) and [new experiment template](EXPERIMENT_TEMPLATE.md)
- [Source inventory and hashes](evidence_manifest.json)

## Experiment catalogue

| ID | Experiment | Status at journal entry |
|---|---|---|
{index}
| 16 | [Training exposure:10% versus50% motion](experiments/16-training-exposure.md) | Running snapshot |

The useful trajectory is not a sequence of architectures declared permanently good or bad. It contains task acquisition failures, output-use failures, genuine improvements, and regressions that triggered targeted diagnostics. Notably, contour was solved by changing allocation; earlier motion information survived in E/I rates and benefited from output refitting; spatial memory helped binding but harmed motion; pre-update attention improved delayed orientation, and its blank-period routing is functionally important.

## Scope and provenance

The broader project follows Jeremy Wolfe's Guided Search6.0 as a functional scaffold, omits its diffuser, and provisionally treats activated long-term memory as synaptic weights at the user's request. PAV is one component. The current trained model contains sensory temporal integration, spatial recurrent memory and pre-update attention; it is not a complete GS6 implementation or a validated biological model.

This wiki adds an explanatory layer over retained code, reports, scores, checkpoints and receipts. It does not launch experiments, change models or manufacture missing evidence. Completed pages link primary local artifacts. Live statuses are explicitly dated; they require refresh rather than being treated as permanent facts.

Initial documentation compiled {STAMP}. Scientific claims describe the saved experiments, not independent replication across seeds or general human performance.
''')

put('CHRONOLOGY.md',r'''# How we got here

[Index](README.md) · [Current status](CURRENT_STATUS.md)

The chronology follows the post-reset development sequence. Run-directory dates show the main sensory work on September12,2026 and memory/attention development through September13. A run may begin one local date and finish another UTC date; ordering below is based on ancestry and saved reports, not an invented minute-by-minute history.

## 1. Establish useful sensory encoding before choosing a complete system

The user reset the repository because an inherited architecture and its justification had displaced component-wise research. The new scaffold was Guided Search6.0, with PAV first and other components later. The first binary dot-displacement design was corrected to cardinal random-dot motion with exactly two ordered frames. Its stopped artifacts are historical diagnostics, not final evidence.

[Five encoders](experiments/01-pav-encoder-screen.md) then received the same seven-task exposure. ConvNeXt learned broad sensory distinctions and motion best in that screen, while other candidates were stronger at contour grouping. The next question was therefore whether useful computations could be combined, not whether one model should be permanently mandated.

## 2. Combination ideas did not substitute for acquisition

The [CPU ensemble audit](experiments/02-saved-ensemble-audit.md) found complementary contour errors but a motion cost from averaging. A [three-arm hybrid comparison](experiments/03-hybrid-continuation.md) tested a Gabor branch and late-SE against equally trained continuation. Neither met the contour target; continuation itself improved markedly.

The decisive [allocation experiment](experiments/04-contour-task-allocation.md) changed only which existing task received more updates. Contour improved67.9%→96.2%, while all seven point accuracies stayed above95%. This is the project's first clear demonstration that weak endpoint behavior need not require a new architecture. It also exposed a small contrast cost that remains part of the record.

## 3. Replace simultaneous pair access with causal sensory state

The user asked whether motion should be computed from successive frames and an accumulator. The [KDA/ConvGRU/opponent comparison](experiments/05-causal-temporal-accumulators.md) froze the trained encoder and learned new temporal/readout components. Opponent fast/slow traces with fixed quadrature-energy computations reached98.44–100% across the two-step battery. Reset and reversed-frame tests supported use of history and order.

The user selected this neuroscience-inspired winner for further work. High two-frame accuracy was never evidence of long memory; that became the next measured question.

## 4. A broad sequence battery failed to acquire several new rules

The [sequence battery](experiments/06-sequence-battery.md) added instructions, cues, integration, delays and retrospective probes while unfreezing learned weights. Many rules stayed weak even at minimal delays. Longer-delay chance behavior therefore did not cleanly identify a retention limit. The correction was to focus acquisition and compare explicit recurrent memory, not to claim the sensory model had a measured universal memory capacity.

## 5. Focused recurrent memories acquire tasks; investigate E/I's motion gap

The [LSTM/EI comparison](experiments/07-lstm-versus-ei.md) learned focused motion-duration and orientation tasks after40,000episodes per arm. LSTM did better on longer motion integration. A [matched-order diagnostic](experiments/08-recency-diagnostic.md) showed greater E/I benefit from late winning evidence.

An appealing immediate story would have been excessive leak. The [state-accessibility diagnostic](experiments/09-state-accessibility.md) instead found early count information still available in final firing rates and a linear duration-readout rescue. The project consequently deferred slow synapses and [refit the existing outputs](experiments/10-readout-refit.md), improving standard-task motion70.31%→79.30% without changing upstream computation.

## 6. Retaining a decision and comparing a delayed visual feature separate

[Retention training](experiments/11-retention-learning.md) made motion accuracy approximately flat through24blanks. Its remaining20% error was already present without delay. Orientation comparison, whose answer requires a new probe, remained near chance at24blanks.

The [orientation diagnostic](experiments/12-orientation-accessibility.md) found accurate pre-probe angle information despite the deployed comparison failure. A decoder trained at sample time failed later, while a late-trained decoder succeeded. A label-only diagnostic comparator rescued performance. “Blanks erase the feature” was therefore too strong: accessibility and use had to be separated.

## 7. Spatial memory solves a binding task but introduces a major trade-off

The user's proposed13×13×64 memory became a [spatial competitor](experiments/13-spatial-memory.md) against dense memory, both with explicit old-memory/current-input comparison. Spatial binding improved strongly, including new location centers. Native single-item D24 did not improve, and motion deteriorated. Binding and native orientation differ in task construction, so high binding is not evidence of higher two-item capacity.

This was a useful component with a known cost, not an accepted universal replacement. The earlier motion-competent model remains preserved.

## 8. Keep teaching fixed while testing learned maintenance

The user rejected changing the target to an angle-supervised delayed report. The implemented [additive controller](experiments/14-selective-maintenance.md) therefore used the same tasks and losses. It produced only a narrow delayed-orientation gain; interrupting its blank-period current did not remove the benefit.

The alternative [pre-update attention](experiments/15-preupdate-attention.md) lets old memory query both current sensory and old-memory values. It improved single D24 to79.30% versus58.40% continuation, but hurt immediate motion. A [phase-specific intervention](experiments/15a-attention-mechanism.md) showed that orientation depends on memory-source routing during blanks. Active refresh versus rejection of blank input is still unresolved.

## 9. Preserve the gain and investigate the regression

The [motion audit](experiments/15b-motion-audit.md) found severe class bias, weak task allocation and a motion deficit predating attention. Calibration partially rescued choices but did not solve delayed performance. The user explicitly requested investigation instead of accepting progress that sacrifices another domain.

[Current Stage1](experiments/16-training-exposure.md) compares unchanged10% motion continuation against50% motion at equal total additional updates, starting the same attention8400 checkpoint. Local control starts first and the additional arm uses a concurrent authorized RunPod. Fresh selection checks all trained cells, preventing an aggregate improvement from hiding task regressions. A proposed temporal residual remains Stage2 design only, contingent on these findings.

## Decisions that remain intentionally unresolved

No universal architecture winner, biological validation, formal multi-item capacity or general attentional selection ability has been established. Slow synapses were deferred after recoverability evidence; new angle teaching was rejected; a residual has not been trained. These are scientific decisions and user constraints, not missing implementations to quietly complete.
''')

put('TASKS_AND_METRICS.md',r'''# Tasks and metrics: read results in their own experimental context

[Index](README.md) · [Architecture](ARCHITECTURE.md)

## Two-frame sensory battery

Every PAV input is two RGB100×100frames. Shared encoder weights process each separately. The ordered decoder or causal accumulator predicts a task-specific answer; task identity selects an output head rather than being inferred as unrestricted natural-language instruction.

| Family | Tested sensory judgment | Chance BA | Important boundary |
|---|---|---:|---|
| Cardinal random-dot motion | Left/right/up/down mean motion from two ordered frames |25%| A disclosed100-pixel adaptation grounded in Krauzlis methods, not a full behavioral reproduction |
| Orientation | Signed Gabor orientation difference |50%| Later sample/probe match-change tasks use a different label rule |
| Contrast | Relative contrast judgment |50%| Nuisance and label construction belong to the saved generator |
| Spatial frequency | Relative frequency judgment |50%| Two-frame sensory discrimination, not remembered frequency capacity |
| Chromatic increment | Chromatic change/increment discrimination |50%| A controlled RGB task, not a complete biological cone model |
| Contour | Grouped contour versus matched alternatives in clutter |50%| A procedural grouping paradigm, not a measured human search threshold |
| Natural spectrum | Spectral-detail comparison from natural photos |50%| Source photographs split by official BSDS partition; new crops can reuse photos |

[Sensory sources/methods](../PreAttentiveVision/two_frame_task_research.md), [dot methods](../PreAttentiveVision/krauzlis_stimulus.md), [natural-image methods](../PreAttentiveVision/natural_image_task.md), and executable [seven-task generator](../PreAttentiveVision/neuroscience_stimuli.py) define details. The early stopped binary dot benchmark is superseded.

## Sequence tasks are not the original two-frame labels

The broad [memory battery](../WorkingMemory/TASK_BATTERY.md) introduces cue/rule formats, integration lengths, decision delays, distractors and retrospective probes. Its minimal-rule conditions assess whether the instructed task was acquired. A poor long-delay score cannot be interpreted as a pure retention limit if the corresponding short-delay task is also not learned.

The later focused motion task displays eight transitions whose directions can vary among four cardinal directions. The target is the direction shown for the greatest total duration, not the final direction or the direction of a single displacement. Blank frames after evidence test whether the answer can still be reported. Since the winner can in principle be computed before blanks, success can retain a decision rather than a detailed visual item.

Single-item orientation instead requires a remembered sample and an incoming comparison probe. D denotes inserted sample-to-query blanks; the identity query is a distinct visible frame. The existing renderer's `post_delay` means query-to-probe interval and was fixed to0 in Retention. Do not relabel it as the manipulated sample delay.

## Exact current timing

Zero-based frame indices from the actual diagnostic wrapper:

| Phase | Single orientation D24 | Motion duration D24 |
|---|---|---|
| Instruction |0|0|
| Visual evidence |Sample1–2|Reference dots1, motion transitions2–9|
| Inserted blanks |3–26|10–33|
| Identity query |27|No separate identity query in this task|
| Probe / report |28|Report34|

At single D0, query/probe are3/4. Motion evidence-to-report age is D+1frames; orientation sample-to-probe age is D+2. Frame count is not a calibrated number of milliseconds. All frames, including blanks, pass through the model; there is no ordinary-path blank oracle.

## Spatial binding has different demands

The two-location task presents two oriented localized patches and tests preservation versus exchanging their locations. Full swaps preserve inventory, so inventory or probe-only shortcuts do not solve the balanced task. But remembering one location is sufficient to detect a full swap: high accuracy is not proof that two independent items were stored.

Binding uses different rendering, change sizes and query content from native full-field single-item orientation. A99% binding score and80% single-item score therefore do not constitute an inverted memory-load curve. New held-out centers test interpolation within the existing spatial layout, not unrestricted extrapolation. Four-case balanced blocks share nuisance structure and are clustered in uncertainty calculations.

## Metrics

For Kclasses and confusion matrix C with true classes in rows,

$$\mathrm{BA}=\frac1K\sum_{k=1}^{K}\frac{C_{kk}}{\sum_j C_{kj}}.$$

BA equals ordinary accuracy only when class counts are balanced. Motion chance is25%; binary task chance is50%. AUC chance is.5 in both. Macro one-versus-rest AUC averages each class's score-ranking ability against the rest; it does not equal the cross-class comparison used by argmax. A model can rank down examples reasonably but never make down the largest score.

A percentage-point change is an absolute accuracy difference:79.3%−70.3%=9.0pp, not9%relative improvement. Confidence intervals are conditional on the particular fitted models and data-generation distributions. Most runs have one training seed per arm; repeated episodes and multiple checkpoints are not independent seed replications.

Paired comparisons evaluate the same underlying examples in both conditions. Resample the underlying unit together: procedural base episodes, binding four-case blocks, canonical motion templates when applicable, or natural source photos. A fresh crop of an old photo is not a new source image. A new test seed makes a new draw, not automatically a wholly new population.

## Selection and evidence strength

- **Training:** examples used for gradient updates.
- **Validation:** planned checkpoint selection or probe/calibration choices; repeated looks make it a development set.
- **Held-out test:** separate examples scored after choices are fixed. Later reuse for exploratory diagnosis must be disclosed.
- **Probe fitting:** an analysis model trained on frozen representations; it does not change deployed weights unless a later explicit experiment does so.
- **Selected checkpoint:** weights chosen by the declared validation rule. **Terminal checkpoint:** final trained weights after the full allocation. They need not be the same.

Different experiments often reevaluate the same parent on different fresh draws. For example, parent motion percentages around69–70% in successive studies are not contradictions. Use the same-experiment paired comparison; never subtract cross-run scores and attach a paired interval.

State reset, attention exclusion and feedback interruption are acute interventions outside ordinary training. A rescue implicates the altered computation but does not prove a trainable permanent change will help. A failed readout is not proof of absent information. Perfect empirical bootstrap intervals can collapse at100%; they do not guarantee zero population error.

Current Stage1 uses fresh validation53973001 and test54973001. Its2pp validation screen is an engineering preservation rule, not proof of equivalence. Final simultaneous one-sided bounds provide a different, explicitly statistical assessment; uncertainty may remain with512examples per cell.
''')

put('ARCHITECTURE.md',r'''# The current model: causal sensory traces, spatial E/I memory and joint attention

[Index](README.md) · [Training status](CURRENT_STATUS.md) · [Task definitions](TASKS_AND_METRICS.md)

This describes the selected **attention8400** architecture and its unchanged Stage1 continuations. It is not the original vector E/I model, the frozen-encoder temporal screen, or the additive-controller arm. Current implementation is [PreUpdateAttention/model.py](../WorkingMemory/PreUpdateAttention/model.py), inheriting [SpatialComparison/model.py](../WorkingMemory/SpatialComparison/model.py) and [RecurrentComparison/model.py](../WorkingMemory/RecurrentComparison/model.py).

## 1. Shared convolutional encoding of the current frame

At each time t the input is (I_t\in\mathbb R^{B\times3\times100\times100}). The learned contour-focused ConvNeXt-GRN/late-SE lineage produces three scales, with learned1×1projections to32channels:

| Scale | Encoder tensor | Projected current tensor (U_t^{(i)}) |
|---|---|---|
|1|B×24×50×50|B×32×50×50|
|2|B×48×25×25|B×32×25×25|
|3|B×96×13×13|B×32×13×13|

The same encoder is used every frame. This sensory path has no feedback from the spatial memory. All learned sensory parameters are currently trainable at the inherited lower learning rate; “encoder frozen” applied to the original temporal comparison, not the present experiment.

## 2. Fixed fast and slow traces at each scale

For each projected field,

$$F_t=.25F_{t-1}+.75U_t,\qquad S_t=.75S_{t-1}+.25U_t.$$

Both traces initialize to the first field, (F_0=S_0=U_0), and each has the same shape as U. Retention coefficients remain fixed; gradients still propagate through these differentiable equations to learned upstream features. The response to a blank is not necessarily zero: the CNN can produce a nonzero blank representation.

## 3. Fixed quadrature energies preserve ordered temporal interactions

Eight zero-mean unit-norm7×7Gabor kernels represent horizontal/vertical coordinates, two frequencies(.125,.25feature-site cycles), and even/odd quadrature phases. The fixed bank filters every feature channel of both traces with reflection padding. For one axis/frequency pair let (f_e,f_o,s_e,s_o) be the even/odd filtered responses. The directional opponent energies are

$$E_+=(f_e+s_o)^2+(f_o-s_e)^2,\qquad E_-=(f_e-s_o)^2+(f_o+s_e)^2.$$

Each energy averages across the32feature channels. Four axis/frequency pairs × two signs yield eight channels. A3×3average pool smooths each channel; normalization is

$$\widehat E_k(x,y)=\frac{\operatorname{Pool}_{3\times3}E_k(x,y)}{10^{-4}+\frac18\sum_{j=1}^{8}\operatorname{Pool}_{3\times3}E_j(x,y)}.$$

The learned temporal output at each scale is a1×1convolution of

$$X_t=[(F_t+S_t)/2,\;F_t-S_t,\;\widehat E_t]\in\mathbb R^{B\times72\times h_i\times w_i},\quad O_t=\operatorname{Conv}_{72\to32}(X_t).$$

The Gabor bank, squares, smoothing and normalization are fixed computations; which learned features they filter and how their results are mixed learn. Fixed equations are an inductive bias, not a fixed entire representation. See [accumulator source](../PreAttentiveVision/TemporalIntegration/accumulators.py).

## 4. Fuse scale-specific sensory history into a64-channel field

Concatenate each current U and emitted O to64channels, apply its learned local convolutional block to32channels, and adaptive-average-pool to13×13. Concatenate the three outputs to96channels and apply the learned fusion block:

$$H_t\in\mathbb R^{B\times64\times13\times13}.$$

H already combines current visual features and short temporal history. Calling it “current image only” would be inaccurate.

## 5. Joint sensory/memory attention supplies the memory input

Flatten spatial sites only for the attention matrix: sensory V and old rate memory M are each B×169×64. The model retains the spatial field as its recurrent state; tokenization is not the old64→8channel compression into one128vector.

With learned site encoding P and source encodings (e_v,e_m), the source code implements

$$Q=W_Q(\operatorname{LN}_q(M)+P+e_m),$$
$$K=W_K(\operatorname{LN}_k([V;M])+[P+e_v;P+e_m]),\quad \mathcal V=W_V[V;M].$$

Split64channels into two32dimensional heads:

$$A_h=\operatorname{softmax}_{338}\left(\frac{Q_hK_h^\top}{\sqrt{32}}+b_{h,\mathrm{source}}-\operatorname{softplus}(\lambda_h)d_{ij}^{2}\right).$$

Q has shape B×2×169×32, K and values B×2×338×32, and A has B×2×169×338. The learned distance bias is an initial/local preference, not a hard spatial mask. Both heads may read both sources. Learned source bias initially favors vision; locality initializes to coefficient4 in grid-site squared distance. The two heads' outputs concatenate and receive a learned64→64projection, producing attended field (J_t\in\mathbb R^{B\times64\times13\times13}).

The input to recurrent memory is (Z_t=\operatorname{LN}_{channels}(J_t)) at each spatial site. Attention replaces the raw sensory drive here. It does not overwrite the separate sensory/comparator routes below. There is no explicit instruction that blanks must attend to memory, no phase input, and no growing key/value cache.

## 6. Spatial excitatory/inhibitory dynamics with adaptation

Persistent rate R and adaptation A each have B×64×13×13shape. There are51excitatory and13inhibitory source channels. A learned3×3kernel has sign constrained by its presynaptic channel:

$$K_{o,i,\Delta}=\operatorname{softplus}(\theta_{o,i,\Delta})s_i,\qquad s_i\in\{+1,-1\}.$$

For learned per-channel time constants and adaptation strength,

$$\tau_r=1+31\sigma(\theta_r),\quad \tau_a=4+124\sigma(\theta_a),\quad g=.5\sigma(\theta_g),$$
$$\alpha=1-e^{-1/\tau_r},\quad\beta=1-e^{-1/\tau_a},$$
$$D_t=W_{in}*Z_t+K*R_{t-1}-g\odot A_{t-1}+b,$$
$$R_t=(1-\alpha)\odot R_{t-1}+\alpha\odot\operatorname{ReLU}(D_t),$$
$$A_t=(1-\beta)\odot A_{t-1}+\beta\odot R_{t-1}.$$

Updates use old states synchronously. Both start at zero. Rates and recurrent currents are not layer-normalized; normalizing nonnegative rates would change their circuit interpretation. Sign constraints, finite time-constant ranges and leak do not mathematically guarantee stability of the full recurrent loop.

## 7. Compare old memory with current sensory history and produce the decision

Before the memory update, a learned comparator receives `[old R, H]`, B×128×13×13. It applies1×1Conv128→64, SiLU,3×3Conv64→64, SiLU, then concatenated spatial mean/max pooling to B×128. This gives a direct learned route for comparing remembered content with a new probe without first overwriting the rate state.

At the final frame three128vectors are added:

$$y_T=f_{sens}([\operatorname{mean}H_T,\operatorname{max}H_T])+W_{mem}[\operatorname{mean}R_T,\operatorname{max}R_T]+W_{cmp}c_T,$$
$$\ell_T=W_{task}y_T+b_{task}.$$

The motion head has four logits; native orientation and binding have two each. Diagnostic branch-logit decomposition adds final head bias once. The comparator uses old memory and original H, not the attended field, so a probe-only memory-input lesion leaves this comparator route intact by construction.

## State, parameter count and learning

The selected attention model has556,128learned parameters; attention adds27,590over spatial528,538. Stage1 introduces zero architecture parameters. Spatial R/A contribute21,632persistent scalars per example =84.5KiB in fp32. Opponent traces add64(50²+25²+13²)=210,816scalars ≈823.5KiB. Together these explicit persistent histories are about908KiB/example, excluding temporary activations, gradients, optimizer state and runtime memory. Attention probabilities are temporary per-frame computations.

Persistent state size does not grow with frame number at inference. Total sequence computation grows approximately linearly with frames because each must be processed. Training with full BPTT retains/recomputes information for gradients across time; activation checkpointing trades compute for memory. This differs from an inference cache that stores every old frame.

The old vector model instead used64→8channels, flatten1352→128, then256-unit dense memory with512state scalars (2KiB). Spatial state is42.25times that memory state. These2KiB and141,968trainable-parameter figures belong to distinct earlier configurations, not the current system.

Stage1 uses fp32, full BPTT, Adam epsilon1e-10, clipping norm1 and inherited LR3e-4 for memory/attention/comparator/binding head versus3e-5 for sensory and ordinary task heads. The code preserves compatible optimizer history. Fixed sensory filter/trace computations remain fixed, while learned inputs and outputs can adapt.

Neuroscience-inspired parts include local oriented energy computations, signed recurrent excitation/inhibition and adaptive rate dynamics. Dot-product attention, LayerNorm, exact channel counts, Adam and BPTT are engineering choices; performance is not validation of a literal cortical circuit.
''')

put('OPEN_QUESTIONS.md',r'''# What remains unresolved, and what would move it forward

[Index](README.md) · [Current experiment](experiments/16-training-exposure.md)

| Question | Existing evidence | Useful discriminating result |
|---|---|---|
| Does motion lack sufficient training allocation? | It had10% of updates and was excluded from selection; earlier contour responded strongly to allocation | Stage1 equal-update10% vs50% motion, with every orientation/binding cell retained in reporting |
| Can motion recover without sacrificing delayed orientation? | Attention improved singleD24 but worsened immediate motion | Fresh paired parent/selected/terminal evaluations; report regressions even if parent is selected |
| Is attention actively refreshing memory or protecting it from blank input? | Excluding memory values only during blanks collapses orientation | A targeted future manipulation separating recurrent value content from sensory-drive renormalization; current exclusion changes both |
| Is early temporal information filtered before attention? | Motion bypass gives a partial acute rescue; fixed fast/slow states contain earlier feature interactions | If Stage1 leaves a deficit, compare one direct learned temporal residual with a matched continuation from the same parent |
| Are remaining motion errors mostly decision bias? | Validation-fitted class offsets rescue D0 but transfer weakly toD24; AUC also falls | Per-class curves and fresh decisions alongside AUC; do not infer the answer from argmax alone |
| Does this model store multiple bound items independently? | Full-swap binding can be solved using one location | A later specifically authorized task that cannot be solved from one stored location; current high binding is insufficient |
| Is the useful orientation code stable over time? | Sample-trained angle decoder transfers badly; late-trained decoder succeeds | Distinguish offsets/scales from content geometry; present diagnostic supports changing accessibility, not a unique coding mechanism |
| Do conclusions replicate across initialization and broader stimuli? | Most comparisons have one seed and one procedural distribution | Replicated training/transfer when warranted; current intervals do not estimate seed variability |

## Current authorized next decision

Finish Stage1 at its fixed allocation. Evaluate parent, validation-selected and terminal models, all14cells, uncertainty and training curves. Compare both total updates and motion examples; equal updates intentionally differ in family exposure and total frame workload. The focused arm at+800updates and control at+4000have equal motion counts but different nonmotion experience and delay assignment, so that view is descriptive rather than a perfectly isolated motion-exposure match.

The proposed Stage2 residual would pool earlier72-channel opponent features from each scale, concatenate216channels on13×13, normalize per site and use a zero-initialized216→64projection into attention's visual source. It adds13,824parameters and no persistent state. It would bypass some learned sensory mixing, not the entire encoder or attention. It is documented in [the design](../WorkingMemory/Research/training_exposure_and_temporal_residual.md); **it has not been trained and must not be reported as a result**.

If tested later, use a single schedule and common parent for residual versus no-residual. Improvement could reflect capacity or optimization as well as information access; residual success alone would not prove earlier layers had erased information. A classifier-only short-trace skip is unlikely to retain evidence through24blanks by itself, since old contributions to the slow trace shrink as.75²⁴≈.001, but nonlinear amplification and learned representations prevent treating that scalar decay as a model-level impossibility proof.

## Interpretive commitments

Preserve a substantial improvement without pretending it is an all-domain upgrade. Keep negative results, unsuccessful hypotheses, and older useful models. Change one justified factor when possible, and allow healthy training to finish its pinned allocation rather than declaring failure from an early checkpoint. A successful source-grounded diagnostic should influence the next implementation; it is not merely another check to collect.
''')

put('RESEARCH_FOUNDATIONS.md',r'''# Research foundations and what was actually adopted

[Index](README.md) · [Architecture](ARCHITECTURE.md)

This page organizes the existing repository's literature rationale. It does not claim a new literature search or independently revalidate every paper. Follow the linked research files for primary-source references, mathematical choices and disclosed adaptations.

| Research area | Adopted computational idea | Where documented | What is not established |
|---|---|---|---|
| Guided Search6.0 | Separate early vision, selection, memory and later integration as functional components | [PAV research](../PreAttentiveVision/research.md), [root scope](../README.md) | The complete GS6 system has not been implemented; treating activated LTM as weights is the user's approximation |
| Modern convolutional networks | Residual local processing, depthwise spatial kernels, channel mixing/normalization, multi-kernel branches and SE modulation | [Encoder rationale](../PreAttentiveVision/research.md) | Tiny100-pixel adaptations are not reproductions of published benchmark results |
| Krauzlis-style random-dot neuroscience | Controlled cardinal mean motion with noise/coherence adaptations | [Exact stimulus sources](../PreAttentiveVision/krauzlis_stimulus.md) | Two-frame100-pixel task does not reproduce every psychophysical/neural condition |
| Motion-energy computations | Quadrature spatial filters combined with unequal temporal traces and opponent energies | [Temporal design](../PreAttentiveVision/TemporalIntegration/README.md) | A useful motion-energy-inspired module is not a fitted V1/MT population model |
| Recurrent WM and E/I dynamics | State-dependent recurrent updates, signed presynaptic influence, leaky firing rates and adaptation | [Recurrent research](../WorkingMemory/Research/recurrent_memory_without_attention.md) | Dale-like signs alone do not validate biological implementation; bounded leak does not ensure full-loop stability |
| Spatial mnemonic representations and lateral recurrence | Preserve feature-location fields and local convolutional interaction | [Spatial rationale](../WorkingMemory/Research/spatial_ei_memory.md) | Shared convolutional weights are an engineering assumption; high swap accuracy does not establish item capacity |
| Selection and maintenance | Let remembered content influence which sensory/memory information drives the next update | [Attention implementation](../WorkingMemory/PreUpdateAttention/README.md) | Literal dot-product attention is not established as the brain's algorithm |
| Multi-task optimization and residual paths | Test allocation before architecture; preserve useful short paths when motivated | [Allocation/residual design](../WorkingMemory/Research/training_exposure_and_temporal_residual.md) | Allocation benefits do not uniquely prove gradient conflict; a residual could help for several reasons |

The existing research discusses modern ML precedents such as xLSTM and gated recurrent/associative systems alongside neuroscience on persistent activity, adaptation, practice-dependent representations, and selecting remembered information for use. These motivate mechanisms to consider, not a requirement to clone a paper wholesale. The implemented normalized LSTM is not an xLSTM reproduction. Later attention authorization supersedes earlier design-stage “no attention yet” restrictions only for the specified experiments.

The project's strongest claims are presently computational and behavioral within its generated tasks: a particular schedule improves contour; particular frozen states support useful readouts; a particular trained attention path is required during blanks. Biological correspondence would require additional predictions and comparisons to neural/behavioral data. Those were not part of the reported training runs.
''')

put('MAINTENANCE.md',r'''# Keeping the lab journal current

[Index](README.md) · [Experiment template](EXPERIMENT_TEMPLATE.md)

The journal is a readable record, not a new experiment harness or approval gate. The researcher who implements or analyzes an experiment updates its page alongside the existing report/receipt. Routine updates should take minutes and use saved artifacts; they should not launch inference or delay healthy training.

## At design and launch

Create the next numbered experiment page using the template. Record the question, source rationale, exact intervention/control, parent checkpoint, changed versus retained weights/optimizer/stream semantics, tasks, planned exposure, selection and finite resource allocation. Mark design-only or running. Link the page from the index, chronology and current status. State any change requested by the user that supersedes an earlier plan.

## During a run

Append a dated snapshot when there is a meaningful validation look, failure/recovery or changed interpretation. Keep update counts, episodes and checkpoint identity together. Label validation as provisional and distinguish completed training from completed final evaluation. A live metrics.csv row is evidence of progress, not of test completion. No need to copy every minibatch into prose: link logs and curves.

## At completion

Use the final report, predictions/analysis and completion receipt. Record actual exposure, selected and terminal checkpoints, task-specific BA/AUC/confusion, important paired uncertainty, regressions and resulting decision. Include parent fallback and terminal regressions if relevant. State dataset/split identity and pairing unit; never manufacture pairing across different test seeds. Record actual compute, recovery history and cloud retrieval/cleanup status from receipts. Update CURRENT_STATUS.md and CHRONOLOGY.md, then mark the experiment completed.

## When evidence changes an interpretation

Keep the earlier hypothesis and date the correction. For example, “possible early-motion loss” became “useful early information accessible; refit output first.” Avoid rewriting history as though the correct mechanism was known initially. Do not overwrite immutable run snapshots or recovered negative results. A lightweight dated correction block on the affected page is sufficient.

## Minimal evidence convention

Each measured statement should link the report or numeric artifact that supports it. The initial journal's evidence_manifest.json records source hashes at compilation, not a promise that mutable aggregate files will never change. For later updates, append a short source/timestamp note or extend the manifest. Do not deserialize large checkpoints just for documentation; use their recorded identity when available.

The initial build_initial_journal.py is a one-time documentation assembly script using standard-library file reads only. It copies selected result tables from original reports and authors the surrounding research narrative. **Do not rerun it over later edits.** Update Markdown directly after the baseline, or deliberately revise the builder and preserve subsequent changes.

## Current Stage1 completion handoff

Update experiments/16-training-exposure.md and CURRENT_STATUS.md after both arms have completed evaluation and combined analysis. Report all14cells, including held-out centers, and parent/selected/terminal outcomes. Confirm cloud cleanup separately from local completion. The existing completion monitor should ask the responsible researcher to make this update from saved results. It does not authorize a new run, residual training or compute extension.
''')

put('EXPERIMENT_TEMPLATE.md',r'''# ID — concise experiment title

[Journal index](README.md)

Status: design / running / evaluation / completed / interrupted. Last updated: ISO timestamp with timezone.

## Question and reason for this test

State the known observation, alternative explanations and why this comparison separates useful possibilities. Link the preceding result and relevant research rationale.

## Design and ancestry

Parent checkpoint/path and recorded hash; arms; exact architecture/task/teaching changes; frozen/trainable components; compatible optimizer/RNG/sampler migration. Describe shared draws accurately (family-local versus task/delay-local) and disclose unequal prior core experience or hardware.

## Exposure, selection and resources

Planned and actual updates/episodes/frames, batch size, per-task allocation, validation looks and selection rule, final test seed and pairing unit, budget/deadline and actual cost. Selected and terminal checkpoints are separate fields.

## Dated progress

- Timestamp: observed checkpoint/update/episode count, validation snapshot or execution event, evidence link. Label provisional results.

## Results

Task-level table with BA, AUC, denominators, relevant paired differences/intervals and regressions. Include the parent and the terminal outcome even when an earlier checkpoint or parent fallback is selected. Link complete confusions, difficulty strata and curves rather than duplicating every raw output.

## What this shows, what it does not show, and the next decision

Separate measurement, interpretation and hypothesis. State limitations that actually affect the conclusion. Include negative results and corrections. Biological inspiration and empirical performance are distinct claims. Mark unrun ideas as design-only.

## Evidence and closure

Links to protocol/code/config, report/results, completion receipt, checkpoint identities, recovery and cloud retrieval/cleanup if applicable. No inference reruns are needed just to finish documentation.
''')

put('experiments/16-training-exposure.md',f'''# Training exposure: recover motion while preserving orientation

[Index](../README.md) · [Current status](../CURRENT_STATUS.md) · [Preceding motion audit](15b-motion-audit.md)

Status: **running at the dated snapshot**. Initial journal entry {STAMP}. This page will be updated from completion receipts and combined analysis; a completed validation look is not a final result.

## Question and design

Can the selected pre-update attention architecture recover motion through more training, and does allocating more of those updates to motion help? The ordinary10% arm tests continuation under the existing schedule. The50% arm changes allocation while retaining exactly the same architecture, tasks, cues, labels, losses and learning-rate policy. Both start attention global8400, preserve compatible Adam, RNG and family-local stream progress, and have fixed4,000additional updates × batch8 =32,000fresh episodes each.

The parent is [attention8400](../../WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/checkpoint_008400.pt), recorded SHA256 `e37602aa20ccfc400ea8fe9d98c11f29c508069388897c55803b97f2ccdf1bc9` in launch evidence. No diagnostic class-offset calibration is installed.

| Arm | Placement | Updates per80-cycle: each of8primary cells | Each of2motion delays | New motion episodes per delay | Each primary cell |
|---|---|---:|---:|---:|---:|
|control_10|Local RTX3070Laptop|9|4|1,600|3,600|
|focused_50|Palladio RunPod RTX3090|5|20|8,000|2,000|

The existing sampler is **family-local**, not delay-cell-local. Common family evidence prefixes match, but schedule changes can attach different delays to particular examples. Equal total updates produce different task exposure and frame workload. Cross-platform differences remain.

## Selection and final evidence plan

Fresh validation seed53973001,128examples/cell; looks at global9200,10000,10800,11600,12400. Candidate eligibility requires no BA decline greater than2pp versus freshly evaluated parent in any of10trained cells. Among eligible candidates, maximize minimum motionD0/D24BA, then mean motion AUC, with earlier ties. Parent fallback is allowed. Healthy training still reaches the fixed endpoint.

Fresh final testseed54973001,512examples/cell:10trained conditions plus4held-out binding-center conditions. Report parent, selected and terminal, paired per-task results, class recalls/confusion, exposure curves and simultaneous one-sided preservation bounds. A2pp validation screen is not a statistical equivalence guarantee. Report terminal regressions even if selection falls back to the parent.

## Execution and budget

Local control started first in [the local run](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/). Its separate14,400s cap ends2026-09-14T03:35:44.401Z. Focused cloud pod `sm1kbctuhqfpo5` was created2026-09-13T23:37:38.206Z and has a four-hour cap ending03:37:38.206Z. Recorded rate is approximately$0.225/hour total, not an invoice. The cloud arm must be retrieved and stopped/deleted promptly on completion, independently of local progress.

No exposure extension, extra architecture arm or residual training is included. The current worker/protocol uses fp32, fullBPTT, gradient clipping1, Adam epsilon1e-10, inherited sensory LR3e-5 and new-module LR3e-4. Profile states do not become production initialization.

## Dated snapshots

The independently read local snapshot and any source-qualified remote update are maintained in [CURRENT_STATUS.md](../CURRENT_STATUS.md). Initial documentation found local step10584 /17,472new episodes, with latest fully saved validation10000; see its timestamp there. Remote provisioning status is an older launch snapshot and must not be treated as live progress. Final accuracy and selection remain pending.

## Result and next decision

**No final result yet at initial journal compilation.** Preliminary improvements in a50%motion validation checkpoint are promising but cannot be compared as equal final training exposure with an earlier local checkpoint. Finish both arms, examine preservation, then decide whether the proposed earlier-temporal-feature residual is still justified. Stage2 remains unrun.

## Evidence

- [Protocol README](../../WorkingMemory/TrainingExposure/README.md)
- [Research design](../../WorkingMemory/Research/training_exposure_and_temporal_residual.md)
- [Launch receipt](../../WorkingMemory/TrainingExposure/launch_receipt.json)
- [Local aggregate](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/aggregate.json)
- [Local metrics](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/control_10/metrics.csv)
- [Local validation10000](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/validation_010000/summary.json)
- [Cloud provisioning / lifecycle coordinates](../../WorkingMemory/TrainingExposure/Cloud/cloud_provisioning.json)
- [Cloud launch receipt](../../WorkingMemory/TrainingExposure/Cloud/launch_receipt.json)
''')

# Freeze an explicitly local-only progress snapshot without cloud or GPU actions.
run='WorkingMemory/TrainingExposure/runs/exposure_20260913_163542'
line=read(run+'/control_10/metrics.csv').strip().splitlines()[-1]
step=int(line.split(',')[0]); new=(step-8400)*8
localval=json.loads(read(run+'/validation_010000/summary.json'))['cells']
parval=json.loads(read(run+'/parent_validation/summary.json'))['cells']
rows=[]
for key in ('motion_direction/motion_D0','motion_direction/motion_D24','orientation_single/single_D24'):
    if key in localval:
        rows.append(f'|{key}|{100*parval[key]["overall"]["balanced_accuracy"]:.2f}%|{100*localval[key]["overall"]["balanced_accuracy"]:.2f}%|')
put('CURRENT_STATUS.md',f'''# Current state of the project

[Journal index](README.md) · [Chronology](CHRONOLOGY.md) · [Open questions](OPEN_QUESTIONS.md)

**Snapshot compiled {STAMP}.** Active logs can advance after this page is written. The factual snapshot below uses local saved files only; it does not claim a new live RunPod read.

## What is running

[Stage1 TrainingExposure](experiments/16-training-exposure.md) continues the exact attention8400 model in two arms:10%motion locally and50%motion on the authorized RunPod. Both have4,000new updates /32,000new episodes fixed. Local metrics were at global{step}, or{new:,}new episodes ({100*new/32000:.1f}%of the training allocation), when this page was compiled. The latest fully read local validation was global10000 (12,800new episodes). Raw training progress and validation are different counters.

| Validation cell (128examples) | Fresh parent8400 | Local control10000 |
|---|---:|---:|
{chr(10).join(rows)}

Evidence: [local metrics](../{run}/control_10/metrics.csv), [validation10000](../{run}/validation_010000/summary.json), [fresh parent validation](../{run}/parent_validation/summary.json).

The earlier coordinator status reported cloud validation11600 at74.22%motionD0,65.62%motionD24 and86.72%singleD24. This was **provisional validation at a later exposure than the local checkpoint above**, not a final comparison. It is recorded here as a source-qualified communication, pending attachment of the retrieved cloud summary by the coordinator. The cloud provisioning file's observed step8561 is a stale launch observation; it is not current progress. No completion is inferred from it.

## Best-supported completed capabilities

| Capability | Completed evidence | Practical meaning |
|---|---|---|
| Two-step sensory discrimination | Opponent temporal model98.44–100% across7tasks | A useful causal sensory component; no long-memory claim |
| Duration evidence accessible in E/I | Linear frozen-state probe improves choices9.47pp | Information can survive despite poor deployed decisions |
| Motion through24blanks | Retention14800 about79–80% atD0/4/12/24 | A motion-competent earlier model remains preserved |
| Spatial feature-location binding | Spatial4400 bindingD24 93.36%; later models near99% | Strong performance on the tested full-swap task, not proved two-item capacity |
| Native orientation across24blanks | Attention8400 79.30% versus ordinary continuation58.40% on paired test | A substantial memory/comparison improvement with task costs |
| Dependence on blank-period memory routing | Acute memory-source exclusion79.30%→50.00% | The learned routing during blanks matters; refresh versus filtering is unresolved |

These entries come from different task distributions and checkpoints. They are **not** the scores of one universally superior model or a single paired benchmark. Follow the [experiment catalogue](README.md) for original comparisons.

## Current limitation

The completed attention8400 model still has poor motion choices:39.65%D0 and35.35%D24 on its original held-out draw. Its useful delayed-orientation improvement should not be accepted as an overall replacement without investigating this cost. Motion already fell during spatial training, so attention is not the sole origin. The current allocation experiment directly addresses this unresolved trade-off.

There is no demonstrated complete Guided Search system, general visual search, unbounded delay memory, multi-item capacity estimate or biological validation. Attention is now implemented, although earlier historical READMEs say it was deferred at their writing. Current scope is governed by the latest authorized experiment.

## Preserved reference lineages

- Contour-focused PAV2268 → causal opponent4032 → broad sequence selected6860.
- Sequence6860 → LSTM5000 and dense E/I5000.
- E/I5000 → output-refit9840 → retention14800 (strong retained motion).
- Retention14800 → dense-comparator4400 and spatial4400 (new spatial core; step counters are experiment-local).
- Spatial4400 → unchanged continuation8400, additive-feedback8400 and pre-update-attention8400.
- Attention8400 → current control_10/focused_50, targetglobal12400.

Checkpoint numbers are not globally comparable totals across architecture migrations. Dense/spatial state sizes and inherited prior training differ; the [architecture page](ARCHITECTURE.md) records the current computation.

## Operational state

RunPod retrieval, stop and deletion are required as soon as the cloud arm completes, independent of the slower local run. Training worker exit alone does not stop GPU billing. The existing completion monitor owns follow-through; this journal does not launch or stop jobs. Final paired results, selected/terminal comparison and cleanup cost remain pending in this snapshot.
''')

put('evidence_manifest.json', json.dumps(dict(compiled_utc=STAMP, purpose='Initial journal source identities; mutable live artifacts are snapshots, not final completion evidence.', files=manifest),indent=2))
print(json.dumps(dict(pages=len(list(OUT.rglob('*.md'))), sources=len(manifest), timestamp=STAMP, local_global_step=step)))
