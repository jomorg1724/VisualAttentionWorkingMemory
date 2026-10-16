## User-directed stop — October5,7:55PM PDT

User explicitly cancelled SharedAttentionKrauzlis. Trainer30284 and CPU evaluator12188 stopped and identities verified exited. Last live update1339; preserved best1000/latest1300, all43 finite aligned Adam states independently CPU-verified. Latest completed monitoring validation1300: BA0.500000, AUC0.509180. No final test; incomplete user-cancelled run, not a spontaneous crash. No automatic restart, successor experiment, diagnostic training, queue or budget extension. Preserve source/checkpoints/logs. Runtime cancellation_receipt.json records ownership and stop evidence. Earlier active descriptions below are historical.

# Shared spatial attention on native Krauzlis

October 5, 2026: user requested stopping the simplified-dot experiment and training the shared appearance/motion attention variant on native Krauzlis, initially with one RViT head. Attention allocation and stimulation/inhibition experiments follow only if task learning is demonstrated.

## Architecture

Fresh 331,635 learned parameters in 43 tensors. The input is the unchanged native RGB movie. Native channels are identical, so RAM storage retains one channel and expands it losslessly before the network. No cue location, event time, motion angle, simulator velocity, or trial metadata enters the model.

At each of 100 positions on a 10 by 10 grid, a learned appearance CNN supplies 32 features. Local RGB means and contrast provide four more. A fixed normalized image correspondence bank compares Hann-windowed 24 by 24 neighborhoods at causal lags 1, 2, and 4 frames. Each comparison searches 81 integer displacement candidates from -4 through +4 pixels on each axis. These are generic image displacement hypotheses, not stimulus direction labels. A shared learned residual filter and three learned temperatures transform correlation into displacement populations.

Local motion features include the populations, pixel-derived reliability, their inferred velocities, and their differences from a learned-rate causal exponential trace. An inferred flow at time t-1 transports the previous image to predict time t; local normalized prediction error supplies another channel. This spatial alignment allows the same location to carry appearance and motion evidence. Overlapping 24-pixel receptive fields mean a map cell is not a perfectly isolated 10-pixel stimulus region.

Fusion projects the 292 local features to width 64 and adds learned spatial positions. Let v_i be the fused visual token, h_i the previous spatial memory, and c the previous CLS memory. The single shared attention head is:

    q = Wq LN(c + learned_CLS)
    k_i = Wk (LN(v_i) + LN(h_i))
    alpha_i = softmax_i(q dot k_i / sqrt(64))
    u_i = Wv v_i + Wm h_i + FFN(Wv v_i + Wm h_i)
    selected = sum_i alpha_i u_i
    h_new = SpatialGRU(100 alpha_i u_i, h)
    c_new = CLSGRU(selected, c)

The final CLS state enters a 64→64→2 nonlinear classifier. One scalar alpha per location gates the fused appearance and motion channels for both readout and spatial writes. Classification has no separate global motion comparator or ungated feature path. The local sensory trace itself remains available at every location.

This is explicitly a **single CLS-query recurrent attention variant**, not an unchanged full pairwise self-attention ViT. It retains patchwise spatial GRU feedback and CLS memory while making the shared spatial selection hypothesis directly testable. It does not yet implement an object-memory architecture for the entire task suite.

## Native task and objective

Original SpatialBatteryStream rendering is reused unchanged: two competing dot apertures, target cue, fixation, lifetime resets, dot direction dispersion, speed .375 pixels/frame, angular changes 26/28 degrees, and baseline durations B12/B20/B28. Total sequences are 29/37/45 frames. Full native 100-event cycles contain 57 target changes, 29 foil changes, and 14 catch trials. Partial pool tails retain the renderer's native sampling. The positive choice means target change, not any change anywhere.

Loss is terminal two-class cross entropy plus 0.05 times pixel-derived causal transport prediction error. Prediction error is normalized by local image contrast and weighted by reliability from adjacent image energies. Native labels supervise only the terminal decision; no direction regression or supplied motion difference is used. The prior simplified paired-appearance VICReg loss is not retained because native trials are not constructed as matched dynamics pairs. The correspondence bank is a substantial motion inductive bias, and success would not establish that a generic raw-pixel ViT learns motion unaided.

## Training and evaluation

One fresh local model and Adam3e-4, FP32, no clipping, batch32/micro4, full temporal backpropagation. Seeds195211/195212; disjoint train/validation/test namespaces195101/195102/195103. Each balanced-duration pool has 1,000 native movies, shuffled for two epochs (66 updates) then replaced. No predecessor model or optimizer weights are loaded.

Requested exposure is 6,600 updates, 200,000 presentations and 100,000 unique movies. The prospective CUDA profile may lower this to complete 66-update pools. A finite 14,400-second cap includes discarded tiny-fit/profile weights, production, evaluation and reporting, independently enforced by the guard; no renewal or cloud launch. See `LocalRuntime/run01/profile.json` for the actual pinned exposure. A training-only eight-movie fit check is bounded at128 updates and is not evidence of generalization.

Validation uses100 native movies per duration, with minimum mean cross entropy selecting best.pt. Final test uses1,000 per duration (3,000 total), terminal argmax with no test threshold fitting. Report BA/AUC and target/foil/catch response rates. Only best.pt and latest.pt are retained; complete optimizer/RNG/pool position is saved. Profile/final result claims require their actual receipts.

## Planned causal analysis, conditional on learning

The model can return alpha for every frame and accepts an optional additive spatial score bias before softmax. Training does not use this hook. Future excitation/inhibition comparisons should use matched trials and locations, compare target and foil changes, separate cue/event timing, and quantify both behavioral changes and attention allocation. Softmax interventions redistribute attention; a negative bias is a computational selection perturbation, not a literal neuronal silencing model. Feature-general attention and biological SC equivalence remain hypotheses.

The CPU checks verify finite gradients, all43 tensors changing after two updates, causal normalized attention, zero-bias identity, and the expected first-frame response to a positive local score bias. This is software validation, not evidence for the scientific hypothesis.

## Scientific motivation

- Morgan, Albanna and Herman RViT work: https://arxiv.org/abs/2502.10955
- Zénon and Krauzlis, 2012, SC contribution to attention: https://pmc.ncbi.nlm.nih.gov/articles/PMC3448852/
- Müller et al., SC stimulation and attention: https://pubmed.ncbi.nlm.nih.gov/15601760/
- RAFT, correspondence volumes as an optical-flow representation: https://arxiv.org/abs/2003.12039
- Rao and Ballard, predictive coding: https://www.nature.com/articles/nn0199_79

The implementation combines engineering mechanisms motivated by these ideas; it does not claim to reproduce any paper's full circuit or algorithm.

Actual CUDA profile pins 2,574 updates (39 pools,78,000 presentations,39,000 unique movies), batch32/micro4. Worst-duration update median3.656s; pool generation27.323s. Original finite deadline10:36:52PM PDT unchanged. Tiny-fit8/8 passed at80updates, training-only and discarded. Production starts fresh; first validation66, then250/1000/2000/2574. Final3,000 native trials.


October5, after validation1000: user objects to the long gap before the next result. Added an independent CPU-only saved-checkpoint evaluator every100updates (roughly5-6minutes), using the identical300-trial validation set and originaldeadline. Below-normal-priority monitor12188, oneCPUthread; training process30284 remains unchanged. Results in LocalRuntime/run01/frequent_validation; these monitoring results do not change production best-checkpoint selection. No extra training, GPU worker, new budget, or automatic extension. Initial evaluation cross-checks CPU scores against the existing CUDA validation when available.
