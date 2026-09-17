# Bibliography of external sources cited in the research notes

Compiled 2026-09-17 from a full read of the nineteen research/design notes listed in the task plus a repository-wide grep of `.md` and `.py` files (excluding `.git`, `runs/`, `pulled/` and bundle directories) for `doi`, `arxiv`, `et al`, parenthesised years and external URLs.

Rules used:

- One record per distinct work. Titles, author strings, years, venues and locators are reproduced **as written in the repository**; nothing was added from memory. Where the notes give no title, venue or locator the field says `not given`.
- `Cited in` lists the repository files (relative to the repository root) in which the work is referenced, whether by locator or by name only.
- `Supported` gives one sentence on the claim or design decision the citation carried in those files.
- Locators were not re-fetched or independently verified in this pass; see the final section for records whose locator or authorship is incomplete.

Counts: group 1 = 5, group 2 = 21, group 3 = 14, group 4 = 20, group 5 = 2, group 6 = 3 (65 located references), plus 11 named methods referred to without any citation.

---

## 1. Guided Search and visual attention theory (5)

### wolfe2021gs6
- **Title:** Guided Search 6.0 (title as written: "Guided Search 6.0"; the PDF filename is `Wolfe2021_GS6.pdf`)
- **Authors as written:** Wolfe; "Jeremy Wolfe's Guided Search 6.0"
- **Year:** 2021
- **Venue:** not given
- **Locator:** https://search.bwh.harvard.edu/new/pubs/Wolfe2021_GS6.pdf ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/
- **Cited in:** `PreAttentiveVision/research.md`, `README.md`, `AGENTS.md`, `PreAttentiveVision/README.md`, `LabJournal/README.md`, `LabJournal/RESEARCH_FOUNDATIONS.md`, `PAPER_HANDOFF.md`, `LabJournal/build_initial_journal.py`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/render_report_multitask.py`, `PreAttentiveVision/render_report_hybrids.py`, `PreAttentiveVision/render_report_task_allocation.py`
- **Supported:** The project's functional scaffold: early visual encoding, selective guidance, working memory and later identification are treated as distinct components; the diffuser is omitted and activated LTM is approximated as synaptic weights, explicitly flagged as the user's approximation rather than Wolfe's claim.

### griffin2003
- **Title:** not given
- **Authors as written:** Griffin and Nobre
- **Year:** 2003
- **Venue:** not given
- **Locator:** https://pubmed.ncbi.nlm.nih.gov/14709235/ ; https://ora.ox.ac.uk/objects/uuid%3Aee268292-114c-46f8-b11b-416eafa5823b
- **Cited in:** `WorkingMemory/TASK_BATTERY.md`, `WorkingMemory/Research/neuroscience_task_rationale.md`
- **Supported:** Motivates the retrocue-versus-precue contrast (cues directed to internal representations after stimulus offset) in the retrospective-probe protocol, with the project's cue timing declared an adaptation.

### ester2018
- **Title:** not given
- **Authors as written:** Ester, Nouri and Rodriguez
- **Year:** 2018
- **Venue:** not given
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6596221/
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Retrospective cues reduced loss of spatial information reconstructed from EEG, adopted as the hypothesis that internal selection can influence maintenance and not only the final response.

### panichello2021
- **Title:** not given
- **Authors as written:** Panichello and Buschman
- **Year:** 2021
- **Venue:** not given (Nature URL)
- **Locator:** https://www.nature.com/articles/s41586-021-03390-w ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8223505/
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`, `WorkingMemory/Research/recurrent_memory_without_attention.md`, `WorkingMemory/PreUpdateAttention/README.md`
- **Supported:** Monkey recordings distinguishing maintenance from selection/transformation of remembered content into a usable format; used to justify keeping selection as a separately tested component and to motivate goal-dependent reconfiguration, without claiming dot-product heads are the biological circuit.

### bays2018
- **Title:** not given
- **Authors as written:** Bays and Taylor
- **Year:** 2018
- **Venue:** not given
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5788052/
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** A population-coding model fit retrocue benefits via gain and swap-frequency changes; cited as motivation for gain modulation while noting it does not implement the recurrent maintenance mechanism.

---

## 2. Working memory and neural dynamics (E/I, persistent activity, synaptic memory) (21)

### luck1997
- **Title:** The capacity of visual working memory for features and conjunctions
- **Authors as written:** Luck and Vogel
- **Year:** 1997
- **Venue:** Nature
- **Locator:** DOI 10.1038/36846 ; https://www.nature.com/articles/36846
- **Cited in:** `WorkingMemory/SpatialTaskBattery/SOURCES.md`
- **Supported:** Background for controlled feature/conjunction memory comparisons; the four-region target-versus-foil swap task is explicitly not a reproduction and does not borrow its capacity estimate.

### bays2008
- **Title:** not given
- **Authors as written:** Bays and Husain
- **Year:** 2008
- **Venue:** not given
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC2532743/
- **Cited in:** `WorkingMemory/TASK_BATTERY.md`, `WorkingMemory/Research/neuroscience_task_rationale.md`
- **Supported:** Memory precision and its allocation vary with item load, so no fixed four-item human limit is imposed on the load/binding protocol.

### driscoll2024
- **Title:** not given
- **Authors as written:** Driscoll, Shenoy and Sussillo
- **Year:** 2024
- **Venue:** not given (Nature Neuroscience URL)
- **Locator:** https://www.nature.com/articles/s41593-024-01668-6
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Task-trained RNNs reuse context-dependent dynamical motifs including ring attractors; adopted as support for training a shared recurrent system to enter a memory regime under a learned context.

### sagodi2024
- **Title:** not given
- **Authors as written:** Sagodi et al.
- **Year:** 2024
- **Venue:** NeurIPS 2024
- **Locator:** https://proceedings.neurips.cc/paper_files/paper/2024/file/7b78a2a7360d5a9ad750834dc5a33bfb-Paper-Conference.pdf
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Approximate continuous attractors remain useful as attractive slow manifolds over finite horizons; adopted as the target of slow content drift plus transverse stability rather than a perfect ring.

### schmitt2017
- **Title:** not given
- **Authors as written:** Schmitt et al.
- **Year:** 2017
- **Venue:** not given (Nature URL)
- **Locator:** https://www.nature.com/articles/nature22073
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Mediodorsal thalamus sustains cortical rule representations through functional connectivity; motivates distributed control without equating the cue controller with thalamus.

### guo2017
- **Title:** not given
- **Authors as written:** Guo et al.
- **Year:** 2017
- **Venue:** not given (Nature URL)
- **Locator:** https://www.nature.com/articles/nature22324
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Motor preparatory persistence depends on a frontal thalamocortical loop; cited alongside Schmitt et al. as motivation for distributed rule control.

### phillips2025
- **Title:** not given
- **Authors as written:** Phillips et al.
- **Year:** 2025
- **Venue:** not given (ScienceDirect / Neuron-style PII)
- **Locator:** https://www.sciencedirect.com/science/article/pii/S0896627325002211
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Primate recordings and a circuit model linking thalamic populations to abstract-rule selection; supports continued investigation of recurrent rule control without specifying the implementation.

### rawat2024organics
- **Title:** not given (referred to as "ORGaNICs-style divisive normalization")
- **Authors as written:** Rawat, Heeger and Martiniani
- **Year:** 2024/2025
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/html/2409.18946v3 (arXiv 2409.18946)
- **Cited in:** `WorkingMemory/Research/attention_as_selective_stability.md`
- **Supported:** Considered and rejected as the maintenance mechanism: its stability theorem assumes identity recurrence and stability of a normalization equilibrium is not retention of arbitrary samples.

### soni2025
- **Title:** not given
- **Authors as written:** Soni and Frank
- **Year:** 2025
- **Venue:** eLife
- **Locator:** DOI 10.7554/eLife.97894
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** A PFC/basal-ganglia model that learns gating policies for working-memory storage motivates learned protection/update control at a functional level; its RL training and chunking are not adopted.

### bellafard2024
- **Title:** Volatile working memory representations crystallize with practice
- **Authors as written:** Bellafard et al.
- **Year:** 2024
- **Venue:** not given (Nature URL)
- **Locator:** https://www.nature.com/articles/s41586-024-07425-w
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Longitudinal mouse recordings show memory representations develop with practice, so learning history matters when interpreting a weak newly trained task.

### inagaki2019
- **Title:** not given
- **Authors as written:** Inagaki et al.
- **Year:** 2019
- **Venue:** not given (Nature URL)
- **Locator:** https://www.nature.com/articles/s41586-019-0919-7
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Discrete attractor dynamics underlie persistent activity in mouse frontal cortex during delayed response; supports investigating maintained recurrent state without establishing a general visual item store.

### liu2025
- **Title:** not given
- **Authors as written:** Liu et al.
- **Year:** 2025
- **Venue:** not given (Nature Portfolio s42003 URL)
- **Locator:** https://www.nature.com/articles/s42003-024-07282-3
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Inspiration for the adaptive excitatory/inhibitory rate-network competitor (recurrent E/I plus adaptation dynamics), explicitly not a reproduction of every component.

### soo2023
- **Title:** not given
- **Authors as written:** Soo, Goudar and Wang
- **Year:** 2023
- **Venue:** NeurIPS 2023
- **Locator:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Documents the training difficulty of constrained (biologically inspired) RNNs, supporting the rule not to equate a hard-to-optimize E/I network with a disproven neuroscience mechanism.

### sprague2014
- **Title:** not given
- **Authors as written:** Sprague, Ester and Serences
- **Year:** 2014
- **Venue:** not given
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4181677/
- **Cited in:** `WorkingMemory/Research/spatial_ei_memory.md`
- **Supported:** Remembered locations reconstructed from occipital/parietal/frontal population activity, degraded under load; supports spatial mnemonic content, not a literal convolutional memory.

### oldenburg2024
- **Title:** not given
- **Authors as written:** Oldenburg et al.
- **Year:** 2024
- **Venue:** not given (Nature Neuroscience URL)
- **Locator:** https://www.nature.com/articles/s41593-023-01510-5
- **Cited in:** `WorkingMemory/Research/spatial_ei_memory.md`
- **Supported:** Spatial arrangement and feature preference jointly govern recurrent activation/suppression in mouse V1, motivating local feature-dependent E/I connectivity in the spatial E/I memory.

### murray2017
- **Title:** not given
- **Authors as written:** Murray et al.
- **Year:** 2017
- **Venue:** PNAS
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5240715/
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`
- **Supported:** A stable working-memory subspace persists despite strong single-neuron dynamics, motivating cross-time decoders that separate stimulus information from overall state movement.

### mante2013
- **Title:** not given
- **Authors as written:** Mante et al.; "Mante and colleagues"
- **Year:** 2013
- **Venue:** Nature
- **Locator:** https://www.nature.com/articles/nature12742
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`, `WorkingMemory/TASK_BATTERY.md`, `WorkingMemory/Research/neuroscience_task_rationale.md`
- **Supported:** Context-dependent selection and integration of evidence in prefrontal cortex and trained RNNs; motivates the cue-conditioned integration protocols and the separation of evidence, context and choice axes in latent analysis.

### libby2021
- **Title:** not given
- **Authors as written:** Libby and Buschman
- **Year:** 2021
- **Venue:** Nature Neuroscience
- **Locator:** https://www.nature.com/articles/s41593-021-00821-9
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`
- **Supported:** Sensory and memory representations occupy different population subspaces (auditory sequences in mice); motivates a geometric hypothesis for the latent atlas without claiming the same circuit.

### pu2024
- **Title:** not given
- **Authors as written:** Pu et al.
- **Year:** 2024
- **Venue:** Nature Communications
- **Locator:** https://www.nature.com/articles/s41467-024-50717-y
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`
- **Supported:** Some prefrontal subspace rotations exist before task training, so a rotation alone is not evidence of deliberate maintenance; the atlas asks whether geometry relates to accessibility and errors.

### funahashi1989
- **Title:** not given
- **Authors as written:** Funahashi, Bruce and Goldman-Rakic; "Funahashi and colleagues"
- **Year:** 1989
- **Venue:** not given (Journal of Neurophysiology DOI)
- **Locator:** https://journals.physiology.org/doi/10.1152/jn.1989.61.2.331 (DOI 10.1152/jn.1989.61.2.331)
- **Cited in:** `WorkingMemory/Research/neuroscience_task_rationale.md`
- **Supported:** Classic oculomotor delayed-response task with location-dependent prefrontal delay activity, grounding the delayed-report manipulation.

### miller1996
- **Title:** not given
- **Authors as written:** Miller, Erickson and Desimone; Miller et al.
- **Year:** 1996
- **Venue:** not given
- **Locator:** https://ekmillerlab.mit.edu/wp-content/uploads/2013/03/Miller-et-al-1996.pdf (author-hosted)
- **Cited in:** `WorkingMemory/Research/neuroscience_task_rationale.md`
- **Supported:** Prefrontal responses retained sample information through intervening test stimuli, motivating distractor-interference conditions rather than blank-only delays.

---

## 3. Motion, orientation and psychophysics stimuli (including Krauzlis) (14)

### zenon2012
- **Title:** Attention deficits without cortical neuronal deficits
- **Authors as written:** Zénon and Krauzlis
- **Year:** 2012
- **Venue:** Nature 489, 434–437
- **Locator:** DOI 10.1038/nature11497 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3448852/
- **Cited in:** `PreAttentiveVision/krauzlis_stimulus.md`, `WorkingMemory/SpatialTaskBattery/SOURCES.md`, `WorkingMemory/SpatialTaskBattery/PROTOCOL.md`
- **Supported:** Most direct source for the two-patch opposite-direction random-dot attentional task (16° direction SD, 8-frame lifetime, cue/baseline timings); later explicitly not blended into the Arcizet recipe.

### arcizet2018
- **Title:** Covert spatial selection in primate basal ganglia
- **Authors as written:** Arcizet, F. and Krauzlis, R. J.; Arcizet and Krauzlis
- **Year:** 2018
- **Venue:** PLOS Biology
- **Locator:** DOI 10.1371/journal.pbio.2005930 ; https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005930
- **Cited in:** `PreAttentiveVision/krauzlis_stimulus.md`, `WorkingMemory/SpatialTaskBattery/SOURCES.md`, `WorkingMemory/SpatialTaskBattery/PROTOCOL.md`, `WorkingMemory/SpatialTaskBattery/stimuli.py` (comment), `LabJournal/experiments/18-unbiased-attention-spatial-battery.md`
- **Supported:** The single primary recipe for the cued dot-change task (16° SD, 10-frame/100 ms lifetime, 15°/s, 25 dots/deg²/s, 90° patch separation, 57/29/14 target/foil/catch, ~26–28° changes), with compression to 100 pixels declared as adaptation.

### lovejoy2010
- **Title:** Inactivation of primate superior colliculus impairs covert selection of signals for perceptual judgments
- **Authors as written:** Lovejoy and Krauzlis
- **Year:** 2010
- **Venue:** not given (Nature Neuroscience DOI)
- **Locator:** DOI 10.1038/nn.2470 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3412590/
- **Cited in:** `PreAttentiveVision/krauzlis_stimulus.md`, `PreAttentiveVision/cardinal_motion.md`, `PreAttentiveVision/neuroscience_stimuli.py` (protocol `source` field and docstring), `WorkingMemory/RecurrentComparison/OrientationDiagnostic/source/PreAttentiveVision__neuroscience_stimuli.py`, `PreAttentiveVision/render_report_multitask.py`, `WorkingMemory/SpatialTaskBattery/SOURCES.md`
- **Supported:** Source for the two-frame cardinal random-dot motion task: 4.25° aperture radius mapped to 85 pixels, two-refresh dot lifetime, ~0.2° (2-pixel) displacement anchor and coherence-as-fraction construction.

### zhang2010
- **Title:** Rule-Based Learning Explains Visual Perceptual Learning and Its Specificity and Transfer
- **Authors as written:** Zhang et al.; "Zhang and colleagues"
- **Year:** 2010
- **Venue:** not given
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3842491/
- **Cited in:** `PreAttentiveVision/two_frame_task_research.md`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Two-interval Gabor orientation/contrast discrimination methods (92 ms presentations, 600 ms ISI, phase varied) grounding the signed orientation comparison task.

### legge1980
- **Title:** Contrast masking in human vision
- **Authors as written:** Legge & Foley
- **Year:** 1980
- **Venue:** JOSA (from URL)
- **Locator:** https://opg.optica.org/josa/abstract.cfm?uri=josa-70-12-1458 ; https://legge.dl8.umn.edu/sites/legge.psych.umn.edu/files/files/media/legge80_contrast_masking_in_human_vision.pdf
- **Cited in:** `PreAttentiveVision/two_frame_task_research.md`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Pedestal-plus-increment contrast discrimination motivates varying baseline contrast rather than only detecting a pattern against blank.

### campbell1970
- **Title:** not given
- **Authors as written:** Campbell, Nachmias and Jukes; Campbell et al.
- **Year:** 1970
- **Venue:** JOSA (from URL)
- **Locator:** https://opg.optica.org/abstract.cfm?URI=josa-60-4-555
- **Cited in:** `PreAttentiveVision/two_frame_task_research.md`
- **Supported:** Spatial-frequency discrimination depends strongly on frequency ratio, motivating the octave-ratio construction of the spatial-frequency comparison task.

### bias2012
- **Title:** A New Perceptual Bias Reveals Suboptimal Population Decoding of Sensory Responses
- **Authors as written:** not given
- **Year:** 2012
- **Venue:** not given
- **Locator:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3325184/
- **Cited in:** `PreAttentiveVision/two_frame_task_research.md`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Two-interval frequency discrimination in which filtered noise biases perceived spatial frequency; cited as the primary two-interval paradigm and a possible later filtered-noise question.

### krauskopf1992
- **Title:** Color Discrimination and Adaptation
- **Authors as written:** Krauskopf & Gegenfurtner
- **Year:** 1992
- **Venue:** not given
- **Locator:** https://www.allpsych.uni-giessen.de/karl/pdf/01.coldisc.pdf (author-hosted)
- **Cited in:** `PreAttentiveVision/two_frame_task_research.md`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Isoluminant-plane colour discrimination motivates controlling base colour and luminance; the project's linear-RGB luminance-orthogonal axis is explicitly not a cone-isolating replication.

### field1993
- **Title:** Contour Integration by the Human Visual System
- **Authors as written:** Field, Hayes and Hess; Field et al.
- **Year:** 1993
- **Venue:** not given (Vision Research DOI prefix)
- **Locator:** DOI 10.1016/0042-6989(93)90156-Q ; https://dev.ipol.im/~blusseau/biblio/psychophysics/1993-field-hayes-hess--contour-integration-by-the-human-visual-system.pdf
- **Cited in:** `PreAttentiveVision/two_frame_task_research.md`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Association-field path detection with oriented elements in clutter is the paradigm behind the contour-versus-scrambled grouping task.

### tadmor1994
- **Title:** Discrimination of changes in the second-order statistics of natural and synthetic images
- **Authors as written:** Tadmor and Tolhurst
- **Year:** 1994
- **Venue:** not given (Vision Research DOI prefix)
- **Locator:** DOI 10.1016/0042-6989(94)90167-8
- **Cited in:** `PreAttentiveVision/natural_image_task.md`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Human discrimination of altered natural-image spectral statistics grounds the natural-spectrum (beta-weighting) task.

### tajima2010
- **Title:** Discriminating Natural Image Statistics from Neuronal Population Codes
- **Authors as written:** Tajima and Okada
- **Year:** 2010
- **Venue:** not given (PLOS ONE DOI)
- **Locator:** DOI 10.1371/journal.pone.0009704
- **Cited in:** `PreAttentiveVision/natural_image_task.md`
- **Supported:** Computational population-coding treatment of the same discrimination; cited as model context, not as new human data.

### adelson1985
- **Title:** not given (PDF filename `spatio85.pdf`)
- **Authors as written:** Adelson and Bergen
- **Year:** 1985
- **Venue:** not given (JOSA A DOI)
- **Locator:** DOI 10.1364/JOSAA.2.000284 ; https://persci.mit.edu/pub_pdfs/spatio85.pdf
- **Cited in:** `PreAttentiveVision/TemporalIntegration/README.md`, `PreAttentiveVision/TemporalIntegration/neuroscience_design_notes.md`, `PreAttentiveVision/TemporalIntegration/render_report_temporal.py`, `LabJournal/RESEARCH_FOUNDATIONS.md` (by topic)
- **Supported:** Spatiotemporal quadrature energy and directional opponency principles behind the opponent accumulator, with the temporal filters replaced by two causal leaky traces.

### simoncelli1998
- **Title:** not given (referred to as the "V1-to-MT model")
- **Authors as written:** Simoncelli and Heeger
- **Year:** 1998
- **Venue:** not given (Vision Research DOI prefix)
- **Locator:** DOI 10.1016/S0042-6989(97)00183-1 ; https://www.cns.nyu.edu/pub/lcv/simoncelli96-reprint.pdf
- **Cited in:** `PreAttentiveVision/TemporalIntegration/README.md`, `PreAttentiveVision/TemporalIntegration/neuroscience_design_notes.md`, `PreAttentiveVision/TemporalIntegration/render_report_temporal.py`
- **Supported:** Pooling, rectification and divisive normalization adopted as design inspiration for the opponent-energy channels; velocity-plane pooling and MT pattern selectivity are not implemented.

### ponce2008
- **Title:** not given
- **Authors as written:** Ponce, Lomber and Born
- **Year:** 2008
- **Venue:** Nature Neuroscience
- **Locator:** https://www.nature.com/articles/nn2039
- **Cited in:** `WorkingMemory/Research/training_exposure_and_temporal_residual.md`
- **Supported:** Direct V1-to-MT and indirect V1-V2/V3-MT pathways motivate parallel shorter and longer visual routes, i.e. the proposed pre-attention temporal residual.

---

## 4. Architectures: linear attention, delta rule, Kimi Linear / KDA, ConvGRU, opponent energy models, ConvNeXt/SE (20)

### xlstm2024
- **Title:** xLSTM
- **Authors as written:** not given
- **Year:** 2024
- **Venue:** NeurIPS 2024
- **Locator:** https://arxiv.org/abs/2405.04517 (arXiv 2405.04517) ; https://github.com/NX-AI/xlstm
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`, `LabJournal/RESEARCH_FOUNDATIONS.md`
- **Supported:** sLSTM recurrent memory mixing and exponential gating motivate letting previous recurrent state influence updates; the implemented normalized LSTM is explicitly not an xLSTM reproduction.

### gateddeltanet2025
- **Title:** Gated DeltaNet
- **Authors as written:** not given
- **Year:** 2025
- **Venue:** ICLR 2025
- **Locator:** https://arxiv.org/abs/2412.06464 (arXiv 2412.06464)
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Distinguishes retention from targeted updates; motivates independent retain/write gates while the associative matrix state and key/query addressing are excluded.

### gateddeltanet2_2026
- **Title:** Gated DeltaNet-2
- **Authors as written:** not given
- **Year:** 2026 (May 2026 preprint)
- **Venue:** preprint
- **Locator:** https://arxiv.org/abs/2605.22791 (arXiv 2605.22791)
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Separates erase and write controls; cited with Gated DeltaNet as motivation for independent update operations.

### kimilinear2025
- **Title:** Kimi Linear (Kimi Delta Attention, "KDA equation 1")
- **Authors as written:** not given
- **Year:** 2025 (from arXiv id)
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/html/2510.26692v1#S3 (arXiv 2510.26692) ; https://github.com/MoonshotAI/Kimi-Linear
- **Cited in:** `PreAttentiveVision/TemporalIntegration/README.md`, `PreAttentiveVision/TemporalIntegration/render_report_temporal.py`, `WorkingMemory/TASK_BATTERY.md` (by name), `LabJournal/experiments/27-accumulator-conv-stack.md` (by name)
- **Supported:** The decay-then-error-correction associative recurrence (equation 1) used per spatial site as the "spatial KDA" accumulator candidate.

### ballas2016convgru
- **Title:** not given
- **Authors as written:** Ballas et al.
- **Year:** 2016
- **Venue:** ICLR 2016
- **Locator:** https://arxiv.org/abs/1511.06432 (arXiv 1511.06432)
- **Cited in:** `PreAttentiveVision/TemporalIntegration/README.md`, `PreAttentiveVision/TemporalIntegration/render_report_temporal.py`
- **Supported:** Convolutional GRU over CNN feature maps is the precedent for the ConvGRU accumulator candidate.

### santoro2018
- **Title:** Relational recurrent neural networks
- **Authors as written:** Santoro et al.
- **Year:** 2018
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/1806.01822 (arXiv 1806.01822)
- **Cited in:** `WorkingMemory/PreUpdateAttention/README.md`
- **Supported:** Joint multihead recurrent-memory interaction precedent for pre-update joint visual/memory attention; its gating is not inherited.

### layernorm2016
- **Title:** Layer Normalization
- **Authors as written:** not given
- **Year:** 2016 (from arXiv id)
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/1607.06450 (arXiv 1607.06450)
- **Cited in:** `WorkingMemory/Research/recurrent_memory_without_attention.md`
- **Supported:** Per-example LayerNorm on the projected sensory input and gate preactivations as a numerical design choice, not biological evidence.

### he2016resnet
- **Title:** not given
- **Authors as written:** He et al.
- **Year:** 2016
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/1512.03385 (arXiv 1512.03385)
- **Cited in:** `PreAttentiveVision/research.md`
- **Supported:** Residual principle (identity shortcuts) adopted in the anti-aliased residual encoder without the original stage schedule.

### he2016identity
- **Title:** Identity Mappings in Deep Residual Networks
- **Authors as written:** He et al.
- **Year:** 2016
- **Venue:** ECCV 2016
- **Locator:** https://arxiv.org/abs/1603.05027 (arXiv 1603.05027)
- **Cited in:** `WorkingMemory/Research/training_exposure_and_temporal_residual.md`
- **Supported:** Direct signal/gradient routes as an optimization idea behind the proposed 216-to-64 temporal-feature shortcut.

### zhang2019antialias
- **Title:** not given
- **Authors as written:** Zhang
- **Year:** 2019
- **Venue:** ICML (PMLR v97)
- **Locator:** https://proceedings.mlr.press/v97/zhang19a.html
- **Cited in:** `PreAttentiveVision/research.md`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Binomial low-pass filtering before decimation follows the anti-aliasing motivation; exact shift invariance is not claimed.

### woo2023convnextv2
- **Title:** not given (ConvNeXt-V2 / global response normalization)
- **Authors as written:** Woo et al.
- **Year:** 2023
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/2301.00808 (arXiv 2301.00808) ; https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
- **Cited in:** `PreAttentiveVision/research.md`, `PreAttentiveVision/component_combination_research.md`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** GRN formula and ConvNeXt block structure for the `convnext_grn` encoder, with zero-initialised gamma/beta checked against the official implementation.

### yu2023inceptionnext
- **Title:** InceptionNeXt
- **Authors as written:** Yu et al.
- **Year:** 2023 (from arXiv id)
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/2303.16900 (arXiv 2303.16900) ; https://github.com/sail-sg/inceptionnext/blob/main/models/inceptionnext.py
- **Cited in:** `PreAttentiveVision/research.md`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Identity plus 3x3, 1x11 and 11x1 depthwise branch split for the `inceptionnext` encoder.

### howard2019mobilenetv3
- **Title:** not given (MobileNetV3)
- **Authors as written:** Howard et al.
- **Year:** 2019
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/1905.02244 (arXiv 1905.02244) ; https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
- **Cited in:** `PreAttentiveVision/research.md`, `PreAttentiveVision/component_combination_research.md`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/render_report_multitask.py`
- **Supported:** Inverted residual blocks with h-swish and SE gating for the `mobilenet_se` encoder; no architecture search or hardware-optimised configuration reproduced.

### hu2017se
- **Title:** not given (Squeeze-and-Excitation)
- **Authors as written:** Hu et al.
- **Year:** 2017 (from arXiv id)
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/abs/1709.01507 (arXiv 1709.01507)
- **Cited in:** `PreAttentiveVision/component_combination_research.md`, `PreAttentiveVision/render_report_hybrids.py`
- **Supported:** Portable SE channel-gate function behind the centred SE-style residual gate added to the ConvNeXt 13x13 output.

### dapello2020vone
- **Title:** not given (VOneNet)
- **Authors as written:** Dapello et al.
- **Year:** 2020
- **Venue:** NeurIPS 2020 (papers.neurips.cc)
- **Locator:** https://papers.neurips.cc/paper_files/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html ; https://github.com/dicarlolab/vonenet/blob/master/vonenet/modules.py
- **Cited in:** `PreAttentiveVision/research.md`, `PreAttentiveVision/component_combination_research.md`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/render_report_multitask.py`, `PreAttentiveVision/render_report_hybrids.py`
- **Supported:** Fixed Gabor simple/complex front-end before a learned hierarchy (`vone_resnet` and the Gabor side branch); fitted parameter distributions and neuronal stochasticity are not adopted.

### linsley2018hgru
- **Title:** not given (horizontal gated recurrent units)
- **Authors as written:** Linsley et al.
- **Year:** 2018
- **Venue:** NeurIPS 2018
- **Locator:** https://papers.nips.cc/paper_files/paper/2018/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html
- **Cited in:** `WorkingMemory/Research/spatial_ei_memory.md`
- **Supported:** ML precedent for recurrence within feature maps supporting the convolutional spatial E/I memory; does not establish delay-period storage.

### park2025
- **Title:** not given
- **Authors as written:** Park, Zhang and Choe
- **Year:** 2025 (preprint)
- **Venue:** not given (arXiv)
- **Locator:** https://arxiv.org/html/2509.15460v1 (arXiv 2509.15460)
- **Cited in:** `WorkingMemory/Research/spatial_ei_memory.md`
- **Supported:** Convolutional lateral recurrence with E/I-inspired weight regularisation; the project keeps hard presynaptic sign constraints instead of their soft regulariser.

### chen2018gradnorm
- **Title:** GradNorm
- **Authors as written:** Chen et al.
- **Year:** 2018
- **Venue:** ICML 2018 (PMLR v80)
- **Locator:** https://proceedings.mlr.press/v80/chen18a.html
- **Cited in:** `WorkingMemory/Research/training_exposure_and_temporal_residual.md`, `PreAttentiveVision/next_experiment_task_allocation.md`, `PreAttentiveVision/render_report_task_allocation.py`
- **Supported:** Task imbalance and unequal learning progress are established multitask concerns; fixed sampling schedules are tested first and GradNorm is not implemented.

### yu2020pcgrad
- **Title:** Gradient Surgery
- **Authors as written:** Yu et al.
- **Year:** 2020
- **Venue:** NeurIPS 2020
- **Locator:** https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- **Cited in:** `WorkingMemory/Research/training_exposure_and_temporal_residual.md`
- **Supported:** Conflicting task gradients treated as a hypothesis for the motion/orientation trade-off; no gradient-surgery method is added.

### recon2023
- **Title:** Recon
- **Authors as written:** not given
- **Year:** 2023
- **Venue:** ICLR 2023
- **Locator:** https://arxiv.org/abs/2302.11289 (arXiv 2302.11289)
- **Cited in:** `WorkingMemory/Research/training_exposure_and_temporal_residual.md`
- **Supported:** Cited with Gradient Surgery as motivation for treating gradient conflict as a hypothesis, not a method to adopt.

---

## 5. Datasets (2)

### bsds500_arbelaez2011
- **Title:** Contour Detection and Hierarchical Image Segmentation (BSDS500)
- **Authors as written:** Arbelaez, Maire, Fowlkes and Malik
- **Year:** 2011
- **Venue:** not given
- **Locator:** https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html ; archive https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz (no DOI given)
- **Cited in:** `WorkingMemory/SpatialTaskBattery/SOURCES.md`, `PreAttentiveVision/natural_stimuli.py` (manifest `citation` field), `PreAttentiveVision/data/bsds500/README.md`, `PreAttentiveVision/render_report_multitask.py`, `WorkingMemory/TASK_BATTERY.md` (by name), `LabJournal/experiments/18-unbiased-attention-spatial-battery.md` (by name)
- **Supported:** Source of natural photographs (200/100/200 official splits kept disjoint) for the natural-spectrum task and the image-recognition task; segmentation annotations and benchmark scores are not used.

### cifar10_krizhevsky2009
- **Title:** Learning Multiple Layers of Features from Tiny Images (CIFAR-10)
- **Authors as written:** Alex Krizhevsky
- **Year:** 2009
- **Venue:** not given (technical report)
- **Locator:** https://www.cs.toronto.edu/~kriz/cifar.html ; archive https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz (MD5 c32a1d4ab5d03f1284b67883e8d87530 as written)
- **Cited in:** `PreAttentiveVision/task.md`, `PreAttentiveVision/stimuli.py`, `PreAttentiveVision/render_report.py`, `PreAttentiveVision/natural_image_task.md` (by name)
- **Supported:** Natural-image family of the superseded first two-frame change-detection benchmark (bilinearly upsampled 32x32 images); later judged unsuitable for native 100-pixel detail, motivating BSDS500.

---

## 6. Other (analysis methods) (3)

### kobak2019tsne
- **Title:** not given
- **Authors as written:** Kobak and Berens
- **Year:** 2019
- **Venue:** Nature Communications
- **Locator:** DOI 10.1038/s41467-019-13056-x
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`
- **Supported:** t-SNE/UMAP distances, densities and cluster sizes depend on the embedding procedure, so embeddings are labelled descriptive rather than held-out decoders.

### kobak2016dpca
- **Title:** not given (demixed PCA)
- **Authors as written:** Kobak et al.
- **Year:** 2016
- **Venue:** eLife
- **Locator:** https://elifesciences.org/articles/10989
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`
- **Supported:** Demixed PCA as the proper method for separating task-variable dependencies; generic label regression must not be called dPCA.

### schneider2023cebra
- **Title:** not given (CEBRA)
- **Authors as written:** Schneider, Lee and Mathis
- **Year:** 2023
- **Venue:** Nature
- **Locator:** https://www.nature.com/articles/s41586-023-06031-6
- **Cited in:** `WorkingMemory/LatentDynamics/RESEARCH.md`
- **Supported:** Contrastive label-shaped embeddings noted as a later option; label-supervised embeddings would not independently show that unsupervised structure discovered orientation.

---

## Unverified: references or named methods without a locator, or with incomplete attribution

Records above that carry a locator but lack authors or a title in the notes (locator present, attribution incomplete):

| Key | What is missing |
|---|---|
| bias2012 | Authors not given anywhere in the repository; only the title and a PMC link. |
| layernorm2016 | Authors not given; only the title and arXiv id. |
| recon2023 | Authors not given; only the short name "Recon", venue and arXiv id. |
| xlstm2024, gateddeltanet2025, gateddeltanet2_2026, kimilinear2025 | Authors not given; identified by system name, venue and arXiv id. |
| gateddeltanet2_2026 | arXiv 2605.22791 (May 2026) could not be checked in this pass; treat the id as unconfirmed. |
| sagodi2024, schmitt2017, guo2017, phillips2025, liu2025, oldenburg2024, inagaki2019, bellafard2024 (title given), pu2024, murray2017 | "et al." author lists and no title; only a URL. |
| wolfe2021gs6 | No journal/venue stated; two different locators (BWH PDF and PMC8965574) refer to the same work. |
| cifar10_krizhevsky2009 | Technical report cited by name only; the locator is the dataset page, not the report. |

Named methods or concepts referred to without any citation (cited from memory, no locator; flagged `unverified`):

| Name | Where mentioned | Context |
|---|---|---|
| Adam optimiser | `WorkingMemory/Research/recurrent_memory_without_attention.md`, `WorkingMemory/PreUpdateAttention/README.md`, `PreAttentiveVision/TemporalIntegration/README.md`, many run files | Optimiser choice; no paper cited. |
| Xavier initialisation | `WorkingMemory/PreUpdateAttention/README.md` ("Xavier gain 0.1") | Q/K projection initialisation; no paper cited. |
| LSTM (input/forget/output gates) | `WorkingMemory/Research/recurrent_memory_without_attention.md` | Baseline memory cell; described as a "conventional LSTM-style cell" with no citation. |
| GRU / ConvGRU gates | `PreAttentiveVision/TemporalIntegration/README.md` | Gate equations given; only Ballas et al. (ConvGRU) is cited, the original GRU is not. |
| Dale's principle ("Dale-style"/"Dale-like" signs) | `WorkingMemory/Research/spatial_ei_memory.md`, `LabJournal/RESEARCH_FOUNDATIONS.md` | Fixed presynaptic sign constraint; no citation. |
| UMAP | `WorkingMemory/LatentDynamics/RESEARCH.md`, `WorkingMemory/LatentDynamics/README.md` | Named as an embedding option; only Kobak and Berens (t-SNE critique) is cited. |
| t-SNE, PCA | `WorkingMemory/LatentDynamics/RESEARCH.md` | Named methods; t-SNE caveats via Kobak and Berens, no original t-SNE/PCA citation. |
| Lucas–Kanade optical flow | `WorkingMemory/BatteryAudit/README.md`, `WorkingMemory/BatteryAudit/observers.py`, `LabJournal/experiments/25-battery-audit.md` | Hand-coded ideal observer for the Krauzlis task; no citation. |
| GroupNorm, GELU, SiLU, h-swish/ReLU6 | `PreAttentiveVision/research.md` | Normalisation/activation choices named without citation (h-swish is defined by formula and attributed implicitly to MobileNetV3). |
| Gabor filters / quadrature energy | `PreAttentiveVision/research.md`, `PreAttentiveVision/TemporalIntegration/neuroscience_design_notes.md` | Fixed filter banks described by formula; grounded only indirectly via VOneNet and Adelson–Bergen. |
| BPTT / activation recomputation | `WorkingMemory/TASK_BATTERY.md`, `WorkingMemory/Research/recurrent_memory_without_attention.md` | Training procedure named without citation. |

Non-external items deliberately excluded: the internal "researcher `/root/temporal_neuroscience_design`" consultation named in `recurrent_memory_without_attention.md`, internal repository documents, run receipts, and GitHub implementation links that accompany a paper record (listed under that paper's locator instead).
