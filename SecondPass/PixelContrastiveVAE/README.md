# Pixel-derived pair learning with a three-frame VAE

## Question and current authorization

On October 4, 2026 the user requested a generalizable alternative to the successful angle-supervised encoder: a three-frame VAE supplemented by a contrastive objective derived from raw images. The user confirmed that the second clip continues the same dot movie and explicitly requested full local training. No direction, direction difference, or change label is supplied to the training loss.

This is a new experiment. It starts with fresh learned weights and fresh Adam states. Existing models and checkpoints are preserved. No cloud work is included.

## Stimuli

Six consecutive 100 x 100 RGB frames contain persistent Gaussian dots on a periodic field. Frames 0/1/2 form the first clip. Frames 3/4/5 form the second clip. Direction is constant within each clip. Frame 3 follows the original velocity; transitions 3->4 and 4->5 use the second velocity. There is no dot-layout reset between clips. The renderer is reused from VariationalMotionPredictor; its trained weights and seeds are not reused.

Training contains two unchanged cases for each changed case. Speeds are 0.375, 1, and 2 pixels/frame; changed cases rotate by signed 26 or 28 degrees. These angles control rendering and are analysis metadata, never inputs to the model or loss. Validation and test have balanced change/no-change pairs with matched nuisance variables, independent seed namespaces, and all six speed/angle cells. This is the simplified coherent-dot task, not native Krauzlis cue/distractor/lifetime behavior.

## Computation and objective

A small shared convolutional VAE takes each **ordered** three-frame clip and reconstructs all three frames from a sampled vector latent. There is no encoder-to-decoder skip. Its deterministic posterior mean feeds a normalized pair projection. All active learned parameters train together with one learning rate.

For oldest-to-newest frames, the pixel trace is

    P(X) = [(X0 - 0.5) + 2 (X1 - 0.5) + 4 (X2 - 0.5)] / 7.

The target is the cosine dissimilarity of flattened traces. Centering by 0.5 uses the known renderer background; this assumption must be revisited for natural scenes. The target is detached from the gradient graph.

    r_pixel(A,B) = 1 - cosine(P(A), P(B))
    d_embedding(A,B) = 1 - cosine(e(A), e(B))
    L_pair = 0.5 * mean[(d_embedding - stop_gradient(r_pixel))^2]
    L = L_reconstruction + beta * L_KL + lambda * L_pair.

Reconstruction balances pixel-derived foreground and background per frame; KL is the mean diagonal-Gaussian KL. beta warms to 1e-4. lambda is 1 for the proposed model and 0 for the plain-VAE control. The pair loss uses deterministic means, not independent posterior samples.

This is self-supervised **pixel-distance matching**, a graded pair objective rather than binary-label contrastive learning. An unconditional negative similarity would push apart unchanged motion as well as changed motion; a negative cosine used as a minimized loss would instead encourage all pairs to become similar. Matching an explicit pixel target resolves that sign ambiguity without supplying angular targets.

## What the cheap raw-target check found

A researcher measured the raw decayed-image score on 128 matched contexts per speed/angle cell before training. Direct high-distance-means-change AUC was about 0.108-0.142 at speed 0.375, 0.398-0.420 at speed 1, and 0.490-0.496 at speed 2. Turning can increase image overlap at slow speed; fast motion saturates the score. Independent layouts gave approximately chance scores. This is a limitation of the proposed target, not a measured learned-model result. The reproducible pixel audit records the exact values for this experiment's fresh namespaces separately.

The experiment deliberately tests the user's literal pixel hypothesis. No optical flow, simulator direction, angular target, learned teacher, or new anti-collapse objective is silently substituted. Reconstruction does not guarantee that the separate pair projection cannot collapse; posterior/projection variation and the constant-embedding loss baseline must be recorded.

## Local training and comparison

One Windows CUDA worker trains the VAE-plus-pair-loss arm first and the plain VAE control next, sequentially. They share initialization, generated examples, exposure, and replay order. The model has 1,861,625 learned parameters. FP32, full gradients, Adam 1e-4, epsilon 1e-8, no weight decay, no gradient clipping, and at most two CPU threads. Effective batch is 32 pairs, initially microbatched at four pairs. Pools contain 1,000 fresh pairs, each replayed in two shuffled epochs before replacement.

The planned horizon is 25,024 updates per arm (391 pools; 391,000 unique training pairs and 782,000 presentations per arm). A new finite eight-hour **total** local cap includes profile, both arms, validation, final evaluation, and reporting, beginning with the first GPU profile. If measured costs cannot fit, reduce equal exposure in complete 64-update pool increments before production. Do not lower one arm's exposure after production starts or extend the cap automatically. Save only best/latest per arm, plus source, protocol, optimizer and runtime evidence.

Selection uses each arm's validation objective; labels calibrate a downstream score sign and threshold on validation only and never enter training. The sign is +1 when validation direct AUC is at least 0.5 and -1 otherwise; the signed threshold is then fixed for all test cells. Preserve direct AUC alongside calibrated AUC/BA. Report raw-pixel, common posterior-mean, and pair-projection scores separately. The control's pair projector has no training signal, so its random projection is a secondary reference rather than the primary control comparison. Final tests use the same fresh matched examples for both arms, 3,072 presentations from 1,536 paired contexts per model; paired presentations are not counted as independent contexts. Report all six cells, reconstruction, KL, pair calibration, latent variation, and actual exposure. Final test indices begin at 36,000,000, separately from both renderer and worker engineering fixtures. The raw audit uses training-namespace indices beginning at 12,000,000, rather than final test draws.

## Interpretation

Success would support a pixel-derived auxiliary objective on this particular continuous-dot setting. It would not establish direction invariance across unrelated scenes, native Krauzlis acquisition, or biological attention. Failure would test this target and recipe; it would not rule out self-supervised motion learning in general.

Research background: Kingma and Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). The decay weights, pixel metric, balance rule, and combined objective are experimental engineering choices.
