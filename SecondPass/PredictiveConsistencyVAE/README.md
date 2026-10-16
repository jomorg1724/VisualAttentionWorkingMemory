# Predictive consistency with a three-frame VAE

The [ten-page mathematical specification](../../output/pdf/predictive_consistency_vae.pdf)
documents the implemented equations, normalization, architecture, source rationale,
gradient cache, evaluation and limits. See the [research journal](../../LabJournal/predictive-consistency-vae.md)
for actual queue and recovery evidence. Corrected production in `LocalRuntime/run02/`
completed20,032 updates /313,000 unique twelve-frame movies /626,000 presentations
atOctober4,7:22PM PDT. Final3,072change presentations/1,536paired contexts:
primaryBA50.42%, AUC0.504076; posterior/projectionBA50.68%/50.59%.
Selected20,032; best/latest were independently verified withall38 alignedAdam
states andfiniteweights. Allworkerprocesses exitedwithinoriginal8:11:07PM deadline.
See [finalreport](LocalRuntime/run02/FINAL_REPORT.md). These arefinalheld-out results;
the testedrecipe didnot solve the simplifiedchange task.

The user requested this as the next local experiment, subsequently cancelled the
predecessor's plain-VAE control, and corrected prediction to **three observed
frames followed by three recursively predicted frames**. The first attempt stopped
at64 updates because the validation worker referenced an undefined variable. Its
weights and source are preserved as a failed historical attempt. The corrected
model starts completely fresh; no learned state is inherited. There is no new
control arm or cloud rental.

Three ordered100x100RGB frames enter a nine-channel CNN and a128-dimensional
diagonal-Gaussian VAE. A separate nine-channel context CNN receives exactly three
observed frames, emits128 dimensions, and feeds a128->128->128 predictor. Its output
uses the shared VAE decoder to predict one next frame. The latest two context
frames and that generated frame form the next input; this repeats for three
forecast horizons. All three future-frame losses train the recursive path with
full temporal gradients. Ground-truth future frames never enter the context.
The full-clip VAE reconstructs the original three observed frames
from stochastic latent samples in training and deterministic means in evaluation.

Two independent views of each same ordered clip use coherent periodic translations
of up to4 pixels, coherent contrast factors0.8–1.2, and independent Gaussian pixel
noise with SD0.01. Posterior means pass through a128->128->64 projector. VICReg
combines mean squared view disagreement, a per-dimension variance floor1, and
sample covariance decorrelation. These invariances are explicit assumptions; they
are not evidence of arbitrary scene or motion invariance.

The objective is support-balanced reconstruction + support-balanced mean three-horizon
prediction + beta*meanKL + invariance + variance +0.04*covariance. Covariance is
the average penalty across the two views, so each view's penalty has coefficient
0.02. Beta warms from0 to1e-4 over250 updates. The support mask is derived only
from image contrast relative to the known0.5 renderer background. Angles, directions
and change labels are rendering/analysis metadata and never learning targets.

Every effective batch contains32 continuous twelve-frame movies, split into two
six-frame sequences. Each sequence has three constant-direction observed frames
and three constant-direction future targets. Each VICReg view includes the64
observed contexts only; future targets never enter augmentation or the encoder.
The partial8-pair
tail has16 clips per view. Statistics are never computed independently within
the four-pair memory microbatches. An exact two-pass gradient cache first computes
whole-batch projection gradients, then reencodes the identical materialized views
and combines those gradients with local reconstruction/prediction/KL gradients.
The CPU runner fixture compares this against an ordinary full graph.

Fresh189101/189102/189103 namespaces retain persistent coherent periodic dots,
speeds0.375/1/2 and occasional signed26/28-degree changes. The first after frame
continues the original velocity; only its subsequent two transitions may turn.
Training has two unchanged movies per changed movie, with1,000 fresh movies reused
for two shuffled epochs. The new data are simplified dots without native Krauzlis
cues, distractors or lifetimes.

Training uses one local CUDA worker, FP32 with TF32 disabled, Adam1e-4/epsilon1e-8,
no weight decay/clipping and at most2 CPU threads. All learned parameters train.
Target25,024 updates means391 pools,391,000 unique movies and782,000 presentations.
A finite4-hour total cap already began in the failed original attempt at
1791155467.341355 and ends1791169867.341355 (October4,4:11:07–8:11:07PM PDT).
The corrective run reuses that exact origin/deadline through `--budget-origin`;
it cannot renew the allowance. A new native profile measures the changed three-step
computation within the remaining time, discards its weights, and prospectively
pins the fresh corrected exposure in complete64-update pools,
using a1.25 timing margin and a600-second reporting reserve plus measured evaluation
costs. Disposable profile weights never transfer into production. Only best/latest
model/Adam/RNG/sampler checkpoints are retained and replaced atomically with bounded
Windows sharing-error retries.

Validation uses384 original six-frame change presentations for decision calibration
and384 separate twelve-frame movies for the unsupervised learning objective.
`generate_objective_evaluation(0..383)` adds60million to the index internally, so
these context/future draws are independent of the change assay. Checkpoint selection
uses their matched three-horizon reconstruction/prediction/KL/view objective, with
no labels or angles. Fresh
held-out tests use3,072 presentations from1,536 paired nuisance contexts. Scalar
score sign and threshold are calibrated on validation only and fixed for test.
The primary decision score is forecast surprise: begin with all three before
frames and autoregressively predict all three after frames, then measure the last
two forecast errors. Actual after frames never condition that rollout. The first
forecast compares identical changed/unchanged target frames and must be an exact
paired chance control. Training itself uses recursive generated contexts for all
three forecast horizons; changed-direction after clips are a deliberate departure
from each training sequence's stable direction. Final tests also report an
separate384-presentation held-out objective evaluation (192 paired nuisance
contexts,768 observed clip presentations) separately from the3,072
change presentations, avoiding inflated operational trial counts.

Reports retain each rollout timestep's support-balanced error/fullMSE, copy-last
baseline errors, constant-direction three-horizon objective prediction/copy-last
baselines, posterior/projection
cosine scores, projection variation, all six speed/angle cells, and every trial
score. A PNG sheet shows the first eight fixed-order held-out examples with actual
and forecast movies; its accompanying JSON records trial identities. Paired
presentations must be grouped when estimating uncertainty.

The CPU-only queue normally requires both complete predecessor arms. The explicit
`--accept-cancelled-control` migration instead accepts a completed joint arm plus
the user's recorded cancellation of the plain-VAE control. It waits for the
predecessor worker to exit and the shared OS GPU lock to become free, then activates
once. The corrected run passes the failed first attempt's immutable budget ledger
through `--budget-origin`; failure cannot create more time. Rerunning identical
pending arguments is idempotent, but activation
cannot be retried automatically. An independent hidden deadline guard is spawned
before profiling; it stops only the exact owned child process identity at cap or
queue-owner loss. A pending queue holds no CUDA context and starts no new budget.
The queue keeps the system awake without keeping the display awake and clears its
request on exit.

Source entry points: model.py, dataset.py, train.py, queue.py, check.py and
check_runner.py. The CPU runner fixture now exercises actual64-update production
through validation, best/latest checkpoint selection, final evaluation and receipt
generation using a tiny disposable pool, independently of the scientific run.
Imported source dependencies are included in the queue/profile
hash manifest. Existing models and the predecessor experiment are preserved.

Research background: [VICReg](https://arxiv.org/abs/2105.04906) and
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). Architecture,
augmentation choices and forecast-surprise decision are this experiment's design;
successful simplified learning would not establish native Krauzlis transfer.
