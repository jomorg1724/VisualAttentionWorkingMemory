"""Build a fixed ten-page implementation paper and inspect its rendered bounds."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import json
from html import escape
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import fitz

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output' / 'pdf' / 'predictive_consistency_vae.pdf'
QA = Path(__file__).parent / 'rendered'
QA.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
PAGES = []
def page(title, subtitle, *blocks):
    PAGES.append(dict(title=title, subtitle=subtitle, blocks=blocks))
def p(text): return ('p', text)
def h(text): return ('h', text)
def e(tex, caption): return ('e', tex, caption)

page('Predictive consistency VAE', 'Implementation specification | October 4, 2026 | 1 / 10',
h('Learning expected continuation from pixels'),
p('This experiment asks whether an encoder can learn useful visual dynamics without a motion-direction target, an angular-distance target, or a binary change label. Following the user\'s correction, it combines a three-frame variational autoencoder (VAE), a causal three-context/three-future predictor, and consistency between two altered views of the same ordered context. The design preserves the small convolutional style of PixelContrastiveVAE. Every learned parameter starts fresh and trains together.'),
p('The earlier pair penalty regresses latent cosine distance toward the distance between decayed pixel traces. That target mixes changes of dynamics with displacement and image overlap. Here the movie supplies supervision: reconstruct three observed frames, recursively predict three future frames, and retain a stable representation when appearance changes in prescribed ways. Training feeds predictions back into the next context and carries gradients through all three steps. No across-clip attraction assumes the same direction.'),
e(r'\mathcal{L}=\mathcal{L}_{rec}+\mathcal{L}_{pred}+\beta_k\mathcal{L}_{KL}+\mathcal{L}_{view}', 'The four terms are computed from image tensors and independently drawn augmentations.'),
h('What is implemented'),
p('The VAE uses a 128-dimensional diagonal-Gaussian latent and reconstructs three RGB context frames. A separate causal encoder maps those three frames to a deterministic 128-dimensional context. A two-layer predictor maps context into the shared decoder space. The decoder\'s third output slot supplies one next image; a rolling context repeats this computation three times. A 64-dimensional unnormalized projector supplies view consistency. This is predictive reconstruction plus VICReg-style consistency, not InfoNCE or a cortical microcircuit.'),
h('Execution and evidence'),
p('The pixel-loss model completed near chance; the user cancelled its plain-VAE control at saved update 7,250 and authorized the predictive model immediately. Its first one-frame attempt stopped at update 64 on a validation-save bug, now corrected. The revised three-context/three-future model starts fresh. Native profiling pinned 20,032 updates, 313,000 unique twelve-frame movies and 626,000 presentations within the original deadline. Actual GPU validation at update 64 and best/latest checkpoint verification passed; training continues. No concurrent GPU worker or budget renewal is used.'),
p('Reading map: page 2 specifies data and causality; pages 3-6 define the architecture and losses; page 7 derives memory-efficient optimization; page 8 specifies prediction-surprise evaluation; page 9 explains execution and interpretation; page 10 gives sources and their precise roles.'))

page('Continuous movies and causal inputs', 'Stimulus contract, ordered information, and split provenance | 2 / 10',
p('A training pair now contains twelve consecutive 100 by 100 RGB images: two six-frame segments A and B. Each segment has three observed context frames followed by three future targets, with constant velocity inside that segment. B continues the same dot field and occasionally has a different velocity. The boundary transition still uses A\'s velocity. This extends the training movie to support three future targets without learning to anticipate a randomly chosen turn.'),
e(r'A=(X_0,\ldots,X_5),\quad B=(X_6,\ldots,X_{11}),\quad X_t\in[0,1]^{3\times100\times100}', 'Only each segment\'s first three frames enter the initial encoder/context and view augmentations.'),
p('The renderer places 16-48 persistent Gaussian dots on a toroidal field with gray background 0.5, contrast 0.42, and dot sigma 0.7 pixels. The three speeds are 0.375, 1, and 2 pixels per frame. Changed movies rotate velocity by signed 26 or 28 degrees. These values reproduce the simplified experiment distribution. They are generation settings and evaluation metadata, never encoder inputs or optimization targets.'),
h('Training objective sees only images'),
p('Training contains two unchanged instances for every changed instance and balances six speed/angle cells over each 18-index cycle. Both A and B contribute context reconstruction and recursive future prediction. The targets are the last three actual frames of each stable six-frame segment. No future frame conditions the rollout and no boundary-change tag enters the loss. The original six-frame change assay is retained separately: three observed before frames, then three after frames, with the first after frame identical in matched changed/unchanged trials.'),
e(r'\widehat X_3=F_\theta(X_0,X_1,X_2),\qquad \widehat X_4=F_\theta(X_1,X_2,\widehat X_3)', 'The third forecast shifts the window again; there is no teacher forcing or target-conditioned latent.'),
h('Independent streams and exposure'),
p('Fresh seed namespaces are 189101 for training, 189102 for validation, and 189103 for final test. Training pools have 1,000 fresh twelve-frame movies, replayed for two shuffled epochs. Batches contain 32 pairs, except an eight-pair tail. One pool gives 64 updates and 2,000 presentations. At requested 391 pools, exposure is 391,000 unique movies and 782,000 presentations. Those presentations contain more rendered frames than the original six-frame experiment.'),
p('The change assay uses 384 validation presentations at offset 12 million and 3,072 test presentations at offset 36 million, paired by nuisance context. An independent twelve-frame objective-validation set at offset 60 million measures the actual three-future training loss for checkpoint selection. It does not replace the change assay. Uncertainty retains each paired context as its sampling unit; two related presentations are not independent nuisance draws.'))

page('Architecture and information pathways', 'Small convolutional encoders with one shared decoder | 3 / 10',
p('The full-clip encoder subtracts 0.5, concatenates three ordered RGB frames into nine channels, and applies Conv2d layers 9->24->48->64, each kernel 3, stride 2, padding 1, followed by GELU. The spatial sizes are 100->50->25->13. A 1 by 1 convolution maps 64 channels to eight; flattening produces 1,352 values. Separate linear layers return the 128-dimensional mean and log variance.'),
e(r'(\mu,\ell)=E_\phi(A),\quad \ell\in[-8,4]^{128},\quad q_\phi(z\mid A)=\mathcal N(\mu,\mathrm{diag}(e^\ell))', 'Clamping log variance controls numerical range while keeping a learned posterior.'),
p('The decoder first maps 128 values to 64 by 13 by 13. Transposed convolutions use kernel 4, stride 2, padding 1: 64->48 produces 26 by 26, cropped to 25 by 25; 48->24 produces 50 by 50; 24->9 produces 100 by 100. GELU follows the first two upsampling stages. A final sigmoid and reshape produce three RGB frames. There are no encoder-to-decoder skips.'),
h('Causal context and prediction'),
p('The context encoder uses the same convolutional widths and spatial reductions, with nine input channels for three ordered RGB frames. Its flattened features map to deterministic h in R128. The predictor is Linear(128,128), GELU, Linear(128,128). Its output enters the shared VAE decoder. Output slot index 2 supplies one next RGB image. The three-step rollout shifts the context and appends its own prediction at each step. Future targets never enter context or posterior computation.'),
e(r'h=C_\psi(X_t,X_{t+1},X_{t+2}),\quad u=G_\omega(h),\quad F_\theta(\mathrm{context})=[D_\eta(u)]_2', 'All recursive stages share context encoder, predictor and decoder parameters.'),
h('View projector and parameter policy'),
p('For view consistency, the deterministic full-clip posterior mean feeds Linear(128,128), GELU, Linear(128,64). The 64 outputs are not normalized to unit length: a unit variance floor would otherwise conflict with the bounded coordinates of a unit vector. Stochastic posterior samples serve reconstruction only. The projector has no access to labels, renderer angles, or a separately trained teacher.'),
p('The revised model has 2,124,961 parameters in 38 learned tensors. The separate causal pathway adds capacity and cannot by itself establish that the full-clip representation encodes predictive dynamics. Decoder sharing couples the objectives, but the context encoder could learn most of the motion computation. Prediction and latent-distance results are therefore separate. Fresh initialization, one learning rate, and every active parameter trainable avoid importing competence from the failed one-frame attempt.'))

page('Variational reconstruction and causal prediction', 'Loss normalization and the boundary between likelihood and surrogate | 4 / 10',
p('Training samples z by reparameterization: Gaussian noise is independent of the input, while mean and scale remain differentiable. Evaluation reconstructs from the posterior mean. These operations adopt the VAE method of Kingma and Welling [1]. The deterministic prediction branch uses no posterior draw, reducing noise in its continuation target.'),
e(r'z=\mu+\exp(\ell/2)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I)', 'Stochastic z is used during training; deterministic mu is used for reconstruction evaluation.'),
p('Sparse dots make an ordinary full-image MSE reward predicting the background. The implementation therefore averages foreground and background errors separately. For each image, support M is one where the target differs from gray 0.5 by more than 1/255. Support is derived from the target pixels and has no gradient. Sums include RGB channels as well as spatial coordinates.'),
e(r'R(\widehat X,X)=\frac12\left[\frac{\sum M(\widehat X-X)^2}{\sum M}+\frac{\sum(1-M)(\widehat X-X)^2}{\sum(1-M)}\right]', 'Counts are clamped at one; an empty class removes the factor of two rather than halving its counterpart.'),
p('Reconstruction averages R over the three observed context frames of each segment. Prediction averages R over all three recursively generated future images against the segment\'s three actual future targets, then over segments and examples. All horizon weights are one third. Gradients from later losses flow through earlier generated images; no detach or teacher forcing interrupts that path. Partial batches use actual example counts. Unweighted MSE is logged separately.'),
e(r'\mathcal L_{pred}=\frac{1}{3N}\sum_{i=1}^{N}\sum_{h=1}^{3}R(\widehat X_{i,2+h},X_{i,2+h})', 'Future targets are outside the three observed context frames; N counts both segments of every pair.'),
e(r'\mathcal L_{KL}=\frac{1}{2Nd_z}\sum_{i=1}^{N}\sum_{j=1}^{d_z}(\mu_{ij}^2+e^{\ell_{ij}}-1-\ell_{ij})', 'Here N=2B clips and dz=128; the KL is averaged per latent coordinate.'),
p('The KL coefficient is beta_k=10^-4 min(1,k/250). Reconstruction and prediction coefficients are each one. These normalizations define this experiment and should not be mistaken for the canonical unscaled VAE evidence lower bound. The target-dependent support weights and small KL coefficient create a regularized reconstruction surrogate. We make no exact marginal-likelihood claim.'),
p('A gray predictor, a copy-last-frame predictor, and a model that predicts blur can all achieve deceptively small full-image MSE. Support-balanced error, copy-last baselines, and saved predicted images make those failures visible. The known background and support threshold are renderer assumptions; natural scenes require a different weighting rule.'))

page('View consistency with variance and covariance', 'A graded representation objective without explicit negatives | 5 / 10',
p('For each clip, independently draw two views. Each view applies an integer toroidal translation with horizontal and vertical shifts in [-4,4], shared across all three frames; multiplies contrast about background 0.5 by a uniform factor in [0.8,1.2]; adds independent Gaussian pixel noise with standard deviation 0.01; and clamps values to [0,1]. Spatial position and brightness can change, but the ordered event remains. Rotations, flips, time reversal, and framewise independent shifts are absent.'),
e(r'Y^{(v)}=H_\rho(\mu_\phi(T_v(X))),\qquad Y^{(v)}\in\mathbb R^{N\times64},\quad N=2B', 'A and B are separate clips; attraction occurs between two views of each same clip.'),
p('The loss adopts the invariance, variance, and covariance structure of VICReg [2,3]. Let centered outputs have sample covariance C=(Y-mean(Y))^T(Y-mean(Y))/(N-1). Standard deviations use the same unbiased sample variance and epsilon=10^-4. The invariance term is a mean over examples and coordinates.'),
e(r'I=\frac{\|Y^{(1)}-Y^{(2)}\|_F^2}{Nd},\quad V(Y)=\frac1d\sum_j\max(0,1-\sqrt{\mathrm{Var}(Y_j)+10^{-4}})', 'd=64. The unit standard-deviation floor penalizes constant projector outputs.'),
e(r'Q(Y)=\frac1d\sum_{j\ne k}C_{jk}^2,\quad \mathcal L_{view}=I+\frac{V(Y^{(1)})+V(Y^{(2)})}{2}+0.02[Q(Y^{(1)})+Q(Y^{(2)})]', 'Covariance is averaged across views, then weighted by 0.04; these are custom coefficients.'),
h('Why the representation cannot simply be normalized'),
p('A constant vector achieves perfect view agreement but pays the variance penalty. Unnormalized outputs can increase coordinate variation while covariance discourages duplication across channels. These constraints reduce a particular collapse route; they do not guarantee motion selectivity, semantic features, or a globally optimal solution. The VAE KL and projector variance act on different variables: the projector can rescale small posterior variation.'),
p('Training uses full effective-batch statistics, including both clips of each pair. Those clips are correlated; this is a defined empirical regularizer rather than an assumption of independent observations. Validation computes its statistics over the whole fixed validation set, so its covariance objective differs from the expectation of training minibatch covariance. It remains consistent across checkpoint looks. Appearance invariance may overemphasize static dot identity, which is why prediction quality and representation quality have separate tests.'))

page('Neuroscience rationale and theoretical boundaries', 'Predictive error, temporal association, and efficient representation | 6 / 10',
h('Predictive coding supplies the question, not the circuit'),
p('Rao and Ballard [4] model descending predictions of lower-level activity and ascending residual errors. O\'Reilly and colleagues [5] propose prediction/outcome differences involving cortex and pulvinar as a learning signal. These accounts motivate learning from expected versus observed sensory input. Our model instead uses feedforward convolutional networks, automatic differentiation, and Adam. It has no implemented laminar populations, pulvinar, local synaptic update rule, or iterative inference dynamics.'),
e(r'e_{t+1}=X_{t+1}-F_\theta(X_{t-2},X_{t-1},X_t)', 'Prediction residual and its scalar energy are operational errors, not inferred cortical cell types.'),
h('Temporal association supplies permissible invariances'),
p('Földiák [6] proposes learning invariance from transformation sequences. Slow-feature analysis [7] formalizes slowly varying outputs under nontriviality and decorrelation constraints. These ideas motivate representations that remain useful when retinal appearance changes. This implementation uses two synthetic views of the same clip, rather than a biological trace rule or unconditional attraction between adjacent clips. It intentionally does not pull changed and unchanged dynamics together.'),
h('Motion-energy theory explains why order matters'),
p('Adelson and Bergen [8] represent motion through filters oriented in space-time, quadrature energy, and opponent responses. Their account motivates retaining local spatial and temporal structure. Our convolution sees ordered frame channels but contains no prescribed direction bank, quadrature pairs, or opponent stage. Any learned motion selectivity must be measured after training; the architecture label alone provides no physiological grounding.'),
h('Relationship to contrastive predictive coding'),
p('CPC [9] predicts future representations using a contrastive classification objective with sampled negatives. We adopt the broad predictive-learning motivation, but train a future image with a squared-error surrogate and enforce same-clip view consistency without sampled negatives. There is no InfoNCE denominator, temperature, or mutual-information bound in the implemented objective. Calling this model CPC would conceal a material difference.'),
p('Prediction error is sensitive to any unexpected image change, including illumination, occlusion, new objects, and motion. That is potentially useful for generality, but does not give task relevance or attention selection. The current experiment evaluates coherent motion in a known renderer. Theoretical connections are hypotheses about useful learning principles; measured performance must establish whether these particular engineering choices learn transferable information.'))

page('Exact optimization under GPU memory limits', 'Global consistency gradients with microbatch recomputation | 7 / 10',
p('The effective batch has B=32 pairs, or N=64 clips per augmentation view. Directly retaining graphs for reconstruction, prediction, and two augmented encodings can exceed the laptop GPU\'s memory. Computing VICReg independently on eight-clip microbatches would change its variance and covariance terms. The runner therefore caches gradients in representation space and recomputes encoder microbatches.'),
h('Pass one: global loss and representation derivatives'),
p('Generate the two augmented views once. In no-gradient microbatches, encode all views with the deterministic posterior mean and projector. Concatenate in a fixed clip order into Y1,Y2. Detach them as leaf tensors with requires_grad=True, compute the global view loss, and use autograd to obtain G1=dLview/dY1 and G2=dLview/dY2. These arrays retain only N by 64 values, rather than convolutional activation graphs.'),
e(r'G^{(v)}=\frac{\partial\mathcal L_{view}}{\partial Y^{(v)}},\qquad \nabla_\theta\mathcal L_{view}=\sum_{v,i}(G_i^{(v)})^T\frac{\partial Y_i^{(v)}}{\partial\theta}', 'The cached derivative already contains the full-batch normalization; do not divide it by microbatch count.'),
h('Pass two: reconstruct, predict, and recompute views'),
p('For each microbatch, run the ordinary VAE and causal predictor, weight reconstruction/prediction/KL by its pair count divided by B, and recompute projector outputs for the exact saved views. Backpropagate their inner product with the corresponding cached gradients. Accumulate all parameter gradients, then make one Adam step. No optimizer update occurs between the two passes or between microbatches.'),
e(r'\mathcal L_m^{back}=\frac{b_m}{B}(\mathcal L_{rec,m}+\mathcal L_{pred,m}+\beta_k\mathcal L_{KL,m})+\sum_{v,i\in m}\langle G_i^{(v)},Y_i^{(v)}\rangle', 'This scalar supplies the correct first derivative; its numerical value is not the reported global loss.'),
p('Exact first-derivative equivalence requires deterministic view encodings, identical augmentations, unchanged parameters, and matching clip order. The encoder and projector have no dropout or batch normalization. Stochastic VAE sampling happens only in the ordinary reconstruction branch, outside cached consistency. A CPU numerical check compares cached and direct full-batch gradients. Higher-order derivatives through the cached gradients are not preserved or needed.'),
h('Optimizer and observability'),
p('Adam uses one learning rate 10^-4, epsilon 10^-8, default betas, no weight decay, no clipping, FP32, and disabled TF32. KL warms over 250 updates. Logs include every loss component, finite-gradient checks, gradient norm, projection variation, reconstruction MSE, and training exposure. First-step evidence confirms parameter changes and an Adam state for every active tensor. Partial final batches retain their actual sizes in all reductions.'))

page('Surprise evaluation and downstream calibration', 'Autoregressive continuation separates prediction from observation | 8 / 10',
p('In the original six-frame change assay, the primary score forecasts B from all three observed A frames. It predicts X3, shifts the three-frame window, and uses its own generated images to predict X4 and X5. It never conditions later steps on observed B. Paired changed and unchanged trials sharing A therefore receive the same forecast; only observed outcomes differ. Training now uses exactly this recursive mechanism over three horizons.'),
e(r'\widehat X_3=F(X_0,X_1,X_2),\quad \widehat X_4=F(X_1,X_2,\widehat X_3)', 'The third prediction uses (X2, predicted X3, predicted X4); all three use shared weights.'),
e(r'S_{late}=\frac{R(\widehat X_4,X_4)+R(\widehat X_5,X_5)}{2},\qquad S_{first}=R(\widehat X_3,X_3)', 'Primary late surprise excludes the first boundary frame, which is identical in paired conditions.'),
p('The first-frame score is a negative control: X3 is identical in matched changed/unchanged movies, so it cannot separate their labels. A non-chance matched result indicates pairing or analysis trouble. Later scores can separate a turn but also accumulate forecast blur and drift. Generated-context behavior is now explicitly trained. Training uses stable continuations, whereas changed assay movies introduce an unpredictable turn. Unweighted MSE and copy-last baselines remain visible.'),
h('No labels in representation learning'),
p('Checkpoints minimize the unsupervised objective on the independent twelve-frame objective-validation set, with looks at updates 64, 5,000, 10,000, 15,000, 20,000 and terminal target. This measures reconstruction, three-future prediction, KL and view consistency on stable segments. Separately, six-frame validation change labels calibrate score sign and a BA-maximizing threshold. One sign and threshold per score are fixed across test cells. Test labels alter neither weights nor selection.'),
e(r'BA=\frac{TPR+TNR}{2},\qquad d_{latent}=1-\frac{\mu_A^T\mu_B}{\|\mu_A\|\,\|\mu_B\|}', 'Posterior-mean distance is a secondary representation score, not the primary predictive score.'),
p('Save direct AUC as well as sign-calibrated AUC and BA; an inverted relation must remain visible. Report each speed and angle cell separately, with 512 final presentations per cell and paired context IDs. Prediction may work while latent distance fails, or vice versa. These outcomes answer different questions. If uncertainty or model differences are computed, resample paired nuisance contexts together and account for the different fresh training streams across experiments.'),
p('This simplified held-out distribution establishes neither transfer to natural scenes nor native Krauzlis behavior. No direction-selective attention, cue selection, distractor rejection, or working-memory delay is part of this experiment.'))

page('Queue, checkpoints, and interpretation', 'Finite local execution with recoverable progress | 9 / 10',
p('The original queue waited for the pixel-loss VAE and its plain control. The user explicitly cancelled the control after the joint model completed; its checkpoint at update 7,250 and 20 Adam states are preserved. An explicit cancellation receipt now permits immediate predictive activation, without calling the control complete. The first predictive attempt failed on a validation-save NameError at update 64. Its checkpoint and source are preserved; the revised model starts fresh under the user\'s three-context/three-future correction.'),
h('Budget begins at activation'),
p('The original predictive budget began October 4 at 4:11:07 PM Pacific and ends at 8:11:07 PM. The revision inherits that deadline; it does not renew the four-hour allowance. New native profiling measures the changed computation against the remaining time with a 1.25 safety factor and evaluation reserve. The requested 25,024 updates were prospectively reduced to 20,032 before fresh revised production, in complete 64-update pools. Profile weights are discarded. One local GPU lock and an independent owned-process guard enforce sequential execution and the deadline.'),
p('Atomic checkpoints retain model state, Adam state, RNG states, shuffle order, pool index, epoch, cursor, update count, unique exposure, and selection state. Best and latest have distinct roles: best supplies final evaluation; latest preserves terminal optimizer-compatible progress. Once best weights are loaded for testing, failure handling must not save them with the terminal Adam state. Source hashes, config, profile, launch, deadline, and completion receipts accompany the artifacts.'),
h('Status files must not govern scientific progress'),
p('The pixel predecessor stopped at update 2,873 on Windows live-status replacement denial and resumed exactly. Transient sharing violations now have bounded retries; optional telemetry can warn and skip while checkpoints and completion receipts require success. A separate validation-save bug stopped predictive update 64 despite helper tests. The correction therefore checks the actual production path through validation, selected-weight evaluation, best/latest preservation and completion receipts, in addition to mathematical gradient tests.'),
h('What the next result can establish'),
p('Improved predictive surprise would support a pixel-supervised temporal prediction recipe on this dot distribution. Improved latent separation would additionally support the learned representation as a useful change metric. Reconstruction alone does not establish either result. A low consistency loss could reflect static scene coding, and a low pixel error could reflect background prediction. A result must retain these alternatives rather than claim neuroscience validation.'),
p('Comparison with prior VAE arms is not a single-factor ablation: context encoder, recursive objective, augmentations, longer training targets and fresh streams change together, and the user cancelled the plain control. Natural-video transfer, uncertainty-aware prediction, encoder sharing and isolated loss ablations remain later questions. This request starts one revised model and preserves the negative evidence; it does not launch a sweep.'))

page('Primary sources and implementation map', 'Adopted properties, inspiration, and explicit departures | 10 / 10',
p('[1] Kingma, D. P., and Welling, M. (2013/2014). Auto-Encoding Variational Bayes. arXiv:1312.6114. Adopted: diagonal-Gaussian posterior, reparameterization, analytic Gaussian KL. Departure: support-balanced reconstruction and experimental KL scaling are surrogate choices. <link href="https://arxiv.org/abs/1312.6114" color="#1A5276">arxiv.org/abs/1312.6114</link>'),
p('[2] Bardes, A., Ponce, J., and LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR; arXiv:2105.04906. Adopted: agreement, per-coordinate standard-deviation floor, off-diagonal covariance penalty. Departure: movie views, small projector, custom weights. <link href="https://arxiv.org/abs/2105.04906" color="#1A5276">arxiv.org/abs/2105.04906</link>'),
p('[3] facebookresearch/vicreg, official main_vicreg.py. Source for unbiased variance, epsilon, covariance normalization and off-diagonal computation. Our code is a small independent implementation with gradient caching, not the repository\'s distributed image-training recipe. <link href="https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py" color="#1A5276">github.com/facebookresearch/vicreg</link>'),
p('[4] Rao, R. P. N., and Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience 2, 79-87. DOI:10.1038/4580. Role: prediction/residual motivation; no circuit reproduction. <link href="https://doi.org/10.1038/4580" color="#1A5276">doi.org/10.1038/4580</link>'),
p('[5] O\'Reilly, R. C., Russin, J. L., Zolfaghar, M., and Rohrlich, J. (2021). Deep Predictive Learning in Neocortex and Pulvinar. Journal of Cognitive Neuroscience 33, 1158-1196. DOI:10.1162/jocn_a_01708. Role: prediction/outcome learning hypothesis; no pulvinar or biological update rule. <link href="https://arxiv.org/abs/2006.14800" color="#1A5276">arxiv.org/abs/2006.14800</link>'),
p('[6] Földiák, P. (1991). Learning invariance from transformation sequences. Neural Computation 3, 194-200. DOI:10.1162/neco.1991.3.2.194. Role: temporal association as invariance motivation; no trace-rule implementation. <link href="https://doi.org/10.1162/neco.1991.3.2.194" color="#1A5276">doi.org/10.1162/neco.1991.3.2.194</link>'),
p('[7] Wiskott, L., and Sejnowski, T. J. (2002). Slow Feature Analysis: Unsupervised Learning of Invariances. Neural Computation 14, 715-770. DOI:10.1162/089976602317318938. Role: nontriviality and decorrelation rationale; no unconditional temporal slowness penalty. <link href="https://doi.org/10.1162/089976602317318938" color="#1A5276">doi.org/10.1162/089976602317318938</link>'),
p('[8] Adelson, E. H., and Bergen, J. R. (1985). Spatiotemporal energy models for the perception of motion. JOSA A 2, 284-299. DOI:10.1364/JOSAA.2.000284. Role: preserve ordered space-time information; no fixed motion-energy filters. <link href="https://persci.mit.edu/pub_pdfs/spatio85.pdf" color="#1A5276">persci.mit.edu/pub_pdfs/spatio85.pdf</link>'),
p('[9] van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748. Role: predictive representation context; InfoNCE and its bound are not used. <link href="https://arxiv.org/abs/1807.03748" color="#1A5276">arxiv.org/abs/1807.03748</link>'),
p('Code map: SecondPass/PredictiveConsistencyVAE/model.py defines encoders, decoder, losses, augmentation and rollout; dataset.py defines fresh streams; train.py implements gradient caching, fitting and evaluation; queue.py enforces predecessor and budget conditions; check.py records CPU checks. Runtime receipts, rather than this document, establish actual starts, exposure, selected steps, and results.'))

BODY = ParagraphStyle('body', fontName='Helvetica', fontSize=10.6, leading=14.6, textColor=HexColor('#22313F'), spaceAfter=9)
HEAD = ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=12.2, leading=15.2, textColor=HexColor('#123D54'))
CAP = ParagraphStyle('caption', fontName='Helvetica', fontSize=9.2, leading=12, textColor=HexColor('#536471'))
WIDTH = 504

def math_png(tex, index):
    fig = plt.figure(figsize=(8, .65))
    fig.patch.set_alpha(0)
    tex = tex.replace(r'\mathcal N', r'\mathcal{N}').replace(r'\mathbb R', r'\mathbb{R}')
    tex = tex.replace(r'\mathcal L', r'\mathcal{L}').replace(r'\frac12', r'\frac{1}{2}').replace(r'\frac1d', r'\frac{1}{d}')
    fig.text(.01,.48,'$'+tex+'$',fontsize=13, va='center', color='#153747')
    path = QA / f'equation_{index:02d}.png'
    fig.savefig(path, dpi=220, transparent=True, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)
    return path

def draw_para(pdf, text, y, style=BODY):
    flow = Paragraph(text, style)
    _, height = flow.wrap(WIDTH, 1000)
    flow.drawOn(pdf,54,y-height)
    return y-height-style.spaceAfter

def main():
    assert len(PAGES)==10
    pdf = canvas.Canvas(str(OUT), pagesize=(612,792))
    pdf.setTitle('Predictive Consistency VAE: Mathematical Implementation and Sources')
    pdf.setAuthor('Visual Attention and Working Memory Research')
    bounds=[]; eq_index=0
    for index, item in enumerate(PAGES,1):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        title_size = min(19, 19 * WIDTH / stringWidth(item['title'], 'Helvetica-Bold', 19))
        pdf.setFillColor(HexColor('#123D54')); pdf.setFont('Helvetica-Bold',title_size)
        pdf.drawString(54,742,item['title'])
        pdf.setFont('Helvetica',9); pdf.setFillColor(HexColor('#536471'))
        pdf.drawString(54,722,item['subtitle'])
        pdf.setStrokeColor(HexColor('#B5C7D0')); pdf.line(54,709,558,709)
        y=693
        for block in item['blocks']:
            if block[0]=='p': y=draw_para(pdf,block[1],y)
            elif block[0]=='h':
                y-=3; y=draw_para(pdf,block[1],y,HEAD)-5
            else:
                eq_index+=1; path=math_png(block[1],eq_index)
                img=ImageReader(str(path)); iw,ih=img.getSize()
                width=min(WIDTH,iw/220*72); height=ih/iw*width
                pdf.drawImage(img,54,y-height,width=width,height=height,mask='auto')
                y-=height+5
                y=draw_para(pdf,escape(block[2]),y,CAP)-4
        if y<52: raise RuntimeError(f'Page {index} overflows: bottom={y}')
        bounds.append(dict(page=index,bottom=y))
        pdf.setFillColor(HexColor('#536471')); pdf.setFont('Helvetica',8)
        pdf.drawString(54,31,'Visual Attention and Working Memory | Implementation specification')
        pdf.drawRightString(558,31,str(index))
        pdf.showPage()
    pdf.save()
    document=fitz.open(OUT)
    assert len(document)==10
    for i,pg in enumerate(document):
        pg.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(QA/f'page_{i+1:02d}.png')
    (QA/'qa.json').write_text(json.dumps(dict(pages=10,bounds=bounds,output=str(OUT),equations=eq_index),indent=2))
    (Path(__file__).parent/'document_content.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),pages=len(document),bounds=bounds,equations=eq_index)))

if __name__=='__main__': main()
