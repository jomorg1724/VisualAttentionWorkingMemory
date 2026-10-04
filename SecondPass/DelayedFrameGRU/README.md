# Delayed-frame residual CNN and standard GRU

**Local MPS training is running**, authorized2026-10-02 after implementation. Fresh production optimizer progress has been independently verified; nativeGPUprofilepinned2,631updates/84,192episodes within a new finite8hour localcap. Effectivebatch32/micro1 was selected for36GiBunifiedmemory. [Live run details](RUN_STATUS.md). The cloudthree-layerKDA run and earlier models remain preserved; no new cloud job was created. [Model](model.py), [training adapter](worker.py), [focused check evidence](verification/model_cpu.json).

Each step consumes two explicitly ordered representations:

```text
current frame I_t     -> current CNN  -> 256 features --+
                                                      +-> concatenate512 -> standard GRU256 -> binary head
previous frame I_t-1  -> previous CNN -> 256 features --+
```

The CNNs have separate trainable weights and identical architectures. Each has a32-channel3x3stem and five residual blocks with two3x3convolutions, GroupNorm8 and GELU, with dilation1/1/2/4/1. Every convolution has stride1; the full100x100 spatial field is retained throughout. A learned1x1convolution maps32→4channels; all40,000 spatially addressed values then feed a learned256-dimensional readout with LayerNorm/GELU. There is no spatial pooling. This keeps pixel positions available to the learned readout; it does not guarantee information retention. The native dots move0.375pixels/frame, so early spatial reduction is an avoidable concern.

Current256 and delayed256 features concatenate, in that order, into one standard `torch.nn.GRU(input_size=512,hidden_size=256,num_layers=1)`. Its last state passes through Linear256→128/GELU/Linear128→2. There is no KDA, attention, custom GRU gate or additional recurrent module. Total **21,293,770 parameters across86 trainable tensors**, with24 stride-one convolutions. Most parameters belong to the full-field learned readouts. This is a larger architecture package, not a parameter-matched control.

At t=0 the previous representation is zero. Later it is the previous-branch encoding of the actual preceding image, not a duplicated current image or the current-branch encoding. Each branch processes each relevant image once; the delayed branch skips the unused last image in complete-movie mode. The streaming API returns the delayed encoding for the next step and the ordinary GRU hidden state. Nothing is detached within a movie. Complete-movie forward resets state; every cue, fixation, motion and report image is processed with no phase metadata or oracle gating. Native100Hz frame spacing means this engineering delay is10ms, rather than a fitted neural time constant.

Training adapter preserves unchanged native KrauzlisB12/B20/B28,26/28degree changes, cues/rendering/labels and57/29/14 target/foil/catch proportions. Everything starts fresh: model, Adam, RNG, scheduler, streams and counters. All parameters train with Adam1e-4,no clipping,batch32/micro4,strictFP32 including CNN/GRU arithmetic,TF32/autocast disabled,fulltemporalgradients. The local native profile reduced the4,216update target to2,631updates prospectively, with validation-only selection and fresh paired selected/terminal tests. The localcapbegan18:57:51UTC and hardends02:57:51UTC /October2 7:57:51PM PDT; no renewal. Local execution usesmicro1 rather than the cloud-adapter defaultmicro4. The adapter contains no cloud provisioning tools.

Three focused CPU model checks passed: exact previous-frame alignment/t0zero, no future dependence, separate weights/no pooling/stride1, full-movie versus streaming outputs, state reset, early/delayed-frame gradients and all86 parameter tensors advancing. Two adapter checks passed: constructor identity/dynamiccount/RNG preservation, complete fresh Adam parameter coverage/empty initial state and native API/source closure. These are engineering checks, not task acquisition or native GPU throughput evidence.

## Biological rationale and limits

Motion depends on signals from different positions and times. Delayed and faster pathways can bring successive spatial activations into coincidence; nonlinear comparison can then become direction selective. Whole-cell recordings in fly motion pathways found different temporal responses upstream of direction-selective computations: [Behnia et al.2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243710/). A classical human cortical computation instead uses spatiotemporally oriented filters, paired squared responses and directional opponency: [Adelson and Bergen1985](https://persci.mit.edu/pub_pdfs/spatio85.pdf).

Our one-frame buffer supplies adjacent-time features to a learned nonlinear GRU. It is an engineering adaptation of temporal comparison, not an implementation of a Reichardt correlator, retinal inhibitory circuit or quadrature motion-energy model. Two frame embeddings alone do not ensure direction selectivity: the network still must learn spatial correspondence, baseline-versus-event changes and use of the cue. No outcome from this untrained model explains the preceding failures or establishes neural plausibility.
