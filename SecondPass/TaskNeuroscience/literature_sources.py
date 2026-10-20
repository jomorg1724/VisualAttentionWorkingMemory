"""Verified primary literature for the frozen-model comparison (18 October 2026).

Summaries are original paraphrases. URLs identify the primary publication or its
PubMed author abstract. Empirical studies and theoretical contributions are tagged
separately; a shared task name does not imply identical experimental protocols.
"""
SOURCES = {}

def add(key, authors, year, title, url, kind, evidence, finding):
    SOURCES[key] = dict(authors=authors, year=year, title=title, url=url,
        kind=kind, evidence=evidence, finding=finding, checked='2026-10-18')

add('britten', 'Britten, Shadlen, Newsome and Movshon', 1992,
    'The analysis of visual motion: a comparison of neuronal and psychophysical performance',
    'https://pubmed.ncbi.nlm.nih.gov/1464765/', 'macaque experiment', 'author abstract',
    'In stochastic-motion discrimination, sensitivity of many individual MT neurons approached behavioral sensitivity when stimuli were tailored to the recorded neuron. The study compared neuronal and behavioral discrimination on the same task.')
add('salzman', 'Salzman, Britten and Newsome', 1990,
    'Cortical microstimulation influences perceptual judgements of motion direction',
    'https://pubmed.ncbi.nlm.nih.gov/2366872/', 'macaque experiment', 'author abstract',
    'Electrical stimulation of direction-selective MT clusters biased reported motion toward the preferred direction of the stimulated population, linking a characterized sensory code to a perceptual decision.')
add('devalois_o', 'De Valois, Yund and Hepler', 1982,
    'The orientation and direction selectivity of cells in macaque visual cortex',
    'https://pubmed.ncbi.nlm.nih.gov/7112953/', 'macaque experiment', 'author abstract',
    'Striate neurons exhibited a broad distribution of orientation bandwidths and both direction-selective and direction-insensitive responses. Orientation and motion selectivity were measured as tuning properties of individual cells.')
add('harrison', 'Harrison and Tong', 2009,
    'Decoding reveals the contents of visual working memory in early visual areas',
    'https://pubmed.ncbi.nlm.nih.gov/19225460/', 'human experiment', 'author abstract',
    'Human fMRI patterns in V1-V4 identified the remembered grating orientation during a blank delay, including cases with little sustained average activation. Information in a population pattern could persist despite low mean activity.')
add('mcadams', 'McAdams and Maunsell', 1999,
    'Effects of attention on orientation-tuning functions of single neurons in macaque cortical area V4',
    'https://pubmed.ncbi.nlm.nih.gov/9870971/', 'macaque experiment', 'author abstract',
    'During an orientation attention task, V4 responses increased with attention largely through scaling of tuning curves, without a systematic change in tuning width. V1 modulation was smaller.')
add('legge', 'Legge and Foley', 1980, 'Contrast masking in human vision',
    'https://pubmed.ncbi.nlm.nih.gov/7463185/', 'human experiment', 'author abstract',
    'Human grating discrimination showed facilitation at weak masking contrasts and increased thresholds at stronger masks. A nonlinear contrast transducer, spatial filtering and pooling explained the dependence on masker strength and frequency.')
add('reynolds00', 'Reynolds, Pasternak and Desimone', 2000,
    'Attention increases sensitivity of V4 neurons',
    'https://pubmed.ncbi.nlm.nih.gov/10896165/', 'macaque experiment', 'author abstract',
    'Attention increased V4 responses more strongly at low than high stimulus contrast. The pattern was consistent with an increase in effective contrast, connecting attentional state with sensory sensitivity.')
add('normalization', 'Reynolds and Heeger', 2009, 'The normalization model of attention',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC2752446/', 'theory', 'article abstract and model description',
    'An attention field modulates stimulus drive before a shared suppressive normalization. Changing stimulus size and attentional spread changes whether predicted responses resemble contrast gain, response gain or other modulation patterns.')
add('campbell', 'Campbell and Robson', 1968, 'Application of Fourier analysis to the visibility of gratings',
    'https://doi.org/10.1113/jphysiol.1968.sp008574', 'human experiment', 'publisher abstract',
    'Thresholds for several grating waveforms were explained by the visibility of their Fourier components. Higher harmonics became behaviorally useful when they exceeded their own thresholds, supporting frequency-selective visual mechanisms.')
add('devalois_f', 'De Valois, Albrecht and Thorell', 1982,
    'Spatial frequency selectivity of cells in macaque visual cortex',
    'https://pubmed.ncbi.nlm.nih.gov/7112954/', 'macaque experiment', 'author abstract',
    'Measurements of striate neurons revealed many narrow spatial-frequency bandwidths and a wide range of preferred frequencies at a given eccentricity, supporting multiple spatial-frequency channels.')
add('regan', 'Regan and Beverley', 1983,
    'Spatial-frequency discrimination and detection: comparison of postadaptation thresholds',
    'https://pubmed.ncbi.nlm.nih.gov/6663371/', 'human experiment', 'author abstract',
    'Adaptation altered frequency-discrimination and grating-detection thresholds differently. The dissociation supported a distinction between detecting channel activity and comparing relative responses across overlapping channels.')
add('derrington', 'Derrington, Krauskopf and Lennie', 1984,
    'Chromatic mechanisms in lateral geniculate nucleus of macaque',
    'https://pubmed.ncbi.nlm.nih.gov/6512691/', 'macaque experiment', 'author abstract',
    'Color-space null measurements characterized how LGN neurons combine cone signals. The results exposed opponent chromatic mechanisms and separated chromatic from luminance-dependent response directions.')
add('lennie', 'Lennie, Krauskopf and Sclar', 1990, 'Chromatic mechanisms in striate cortex of macaque',
    'https://pubmed.ncbi.nlm.nih.gov/2303866/', 'macaque experiment', 'author abstract',
    'V1 neurons responded to combinations of chromatic and luminance modulation. Their distribution of preferred color directions differed from the more distinctly grouped LGN mechanisms, showing a cortical transformation of cone-derived signals.')
add('flanagan', 'Flanagan, Cavanagh and Favreau', 1990,
    'Independent orientation-selective mechanisms for the cardinal directions of colour space',
    'https://pubmed.ncbi.nlm.nih.gov/2378069/', 'human experiment', 'author abstract',
    'Opposite tilt aftereffects could be induced for chromatic and luminance gratings. The psychophysical dissociation supported orientation-selective mechanisms operating on multiple chromatic and achromatic signals.')
add('field', 'Field, Hayes and Hess', 1993,
    'Contour integration by the human visual system: evidence for a local association field',
    'https://pubmed.ncbi.nlm.nih.gov/8447091/', 'human experiment', 'author abstract',
    'People detected paths made of aligned Gabor elements within randomly oriented backgrounds. Detection depended strongly on the alignment and continuation of neighboring elements, motivating an association field linking compatible local orientations.')
add('kapadia', 'Kapadia, Ito, Gilbert and Westheimer', 1995,
    'Improvement in visual sensitivity by changes in local context: parallel studies in human observers and in V1 of alert monkeys',
    'https://pubmed.ncbi.nlm.nih.gov/7576633/', 'human and macaque experiments', 'author abstract',
    'Collinear context improved human bar detection and facilitated responses in a subset of monkey V1 neurons. Relative position, orientation and intervening elements controlled the contextual effect.')
add('li', 'Li, Piech and Gilbert', 2006, 'Contour saliency in primary visual cortex',
    'https://pubmed.ncbi.nlm.nih.gov/16772175/', 'macaque experiment', 'author abstract',
    'During contour detection, V1 responses tracked contour salience and were weaker when contours were missed or unattended. Neural discrimination connected contextual responses with perceptual performance.')
add('tolhurst97', 'Tolhurst and Tadmor', 1997,
    'Discrimination of changes in the slopes of the amplitude spectra of natural images: band-limited contrast and psychometric functions',
    'https://pubmed.ncbi.nlm.nih.gov/9509160/', 'human experiment', 'author abstract',
    'Sensitivity to spectral-slope changes could be explained by contrast changes within limited frequency bands. Which band was informative depended on the reference spectrum, linking global-looking judgments to selective spatial filtering.')
add('tolhurst00', 'Tolhurst and Tadmor', 2000,
    'Discrimination of spectrally blended natural images: optimisation of the human visual system for encoding natural images',
    'https://pubmed.ncbi.nlm.nih.gov/11144821/', 'human experiment', 'author abstract',
    'Experiments blending photograph spectra found enhanced sensitivity around natural spectral statistics after controlling image properties. The manipulation tested natural-image specialization more directly than successful discrimination of arbitrary slopes alone.')
add('olshausen', 'Olshausen and Field', 1996,
    'Emergence of simple-cell receptive field properties by learning a sparse code for natural images',
    'https://www.nature.com/articles/381607a0', 'theory', 'publisher abstract',
    'Learning a sparse representation of natural image patches produced localized, oriented, bandpass basis functions. The theory connects environmental statistics to an efficient sensory representation.')
add('griffin', 'Griffin and Nobre', 2003,
    'Orienting attention to locations in internal representations',
    'https://pubmed.ncbi.nlm.nih.gov/14709235/', 'human experiment', 'author abstract',
    'Cues presented before or after a visual array improved recognition of relevant locations. Behavioral and electrophysiological results linked orienting in perception and memory while revealing differences between precue and retrocue processing.')
add('kiani', 'Kiani, Hanks and Shadlen', 2008,
    'Bounded integration in parietal cortex underlies decisions even when viewing duration is dictated by the environment',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC6670720/', 'macaque experiment', 'author abstract',
    'Monkeys could commit to a motion choice before externally controlled viewing ended. LIP activity supported accumulation to a bound, demonstrating that late available evidence need not influence the decision.')
add('stine', 'Stine, Zylberberg, Ditterich and Shadlen', 2020,
    'Differentiating between integration and non-integration strategies in perceptual decision making',
    'https://pubmed.ncbi.nlm.nih.gov/32338595/', 'human experiment', 'author abstract',
    'Task manipulations distinguished prolonged evidence integration from nonintegration strategies that can reproduce common behavioral signatures. Most tested observers showed evidence for prolonged integration under the diagnostic design.')
add('mante', 'Mante, Sussillo, Shenoy and Newsome', 2013,
    'Context-dependent computation by recurrent dynamics in prefrontal cortex',
    'https://pubmed.ncbi.nlm.nih.gov/24201281/', 'macaque experiment and theory', 'author abstract',
    'Monkeys selected motion or color evidence according to context. Prefrontal population activity and trained recurrent-network dynamics linked context-dependent selection with evidence integration.')
add('lovejoy', 'Lovejoy and Krauzlis', 2010,
    'Inactivation of primate superior colliculus impairs covert selection of signals for perceptual judgments',
    'https://pubmed.ncbi.nlm.nih.gov/20023651/', 'macaque experiment', 'author abstract and figure legends',
    'SC inactivation shifted motion judgments away from signals in the affected field toward competing information elsewhere. Deficits depended strongly on informative distractors; affecting the foil location favored judgments based on the cued signal.')
add('bollimunta', 'Bollimunta, Bogadhi and Krauzlis', 2018,
    'Comparing frontal eye field and superior colliculus contributions to covert spatial attention',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC6120922/', 'macaque experiment', 'primary article',
    'Reversible FEF and SC inactivation impaired covert motion-change detection. Comparing the two interventions in a shared behavioral framework tested their contributions to selection during maintained fixation.')
add('arcizet', 'Arcizet and Krauzlis', 2018, 'Covert spatial selection in primate basal ganglia',
    'https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005930', 'macaque experiment', 'primary article abstract and introduction',
    'Monkeys detected cued motion changes with a joystick while maintaining fixation. Caudate neurons encoded relevant location and task epoch, extending spatial-selection signals beyond overt orienting movements.')
add('zenon', 'Zenon and Krauzlis', 2012, 'Attention deficits without cortical neuronal deficits',
    'https://pubmed.ncbi.nlm.nih.gov/22972195/', 'macaque experiment', 'author abstract',
    'SC inactivation impaired attentive behavior while established attentional modulation in visual cortex persisted. Behavioral selection and cortical response enhancement could therefore dissociate.')
add('wheeler', 'Wheeler and Treisman', 2002, 'Binding in short-term visual memory',
    'https://pubmed.ncbi.nlm.nih.gov/11900102/', 'human experiment', 'author abstract',
    'Change-detection experiments separated memory for features from memory for their combinations. Binding was vulnerable to interference, motivating a distinction between retained feature information and attention-dependent integration.')
add('bays', 'Bays, Catalao and Husain', 2009,
    'The precision of visual working memory is set by allocation of a shared resource',
    'https://pubmed.ncbi.nlm.nih.gov/19810788/', 'human experiment and theory', 'author abstract',
    'Continuous-report errors reflected both feature and location memory. Accounting for their joint uncertainty supported a shared-resource description in which precision changes with the number of remembered objects.')
add('miller93', 'Miller, Li and Desimone', 1993,
    'Activity of neurons in anterior inferior temporal cortex during a short-term memory task',
    'https://pubmed.ncbi.nlm.nih.gov/8463829/', 'macaque experiment', 'author abstract',
    'IT responses carried match-related information during delayed matching. Stimulus-selective delay activity could be disrupted by intervening stimuli, separating visible delay activity from successful later recognition.')
add('miller96', 'Miller, Erickson and Desimone', 1996,
    'Neural mechanisms of visual working memory in prefrontal cortex of the macaque',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC6579322/', 'macaque experiment', 'primary article',
    'A delayed matching task with intervening test objects examined prefrontal signals for remembered samples and matching decisions. It provided a comparison with temporal-cortex mechanisms under interference.')
add('basso', 'Basso and Wurtz', 1998,
    'Modulation of neuronal activity in superior colliculus by changes in target probability',
    'https://pubmed.ncbi.nlm.nih.gov/9736670/', 'macaque experiment', 'author abstract',
    'SC buildup activity varied with target probability, including when probability changed while the display remained fixed. More possible targets reduced preparatory activity, linking selection state with uncertainty and saccade latency.')
add('competition', 'Reynolds, Chelazzi and Desimone', 1999,
    'Competitive mechanisms subserve attention in macaque areas V2 and V4',
    'https://pubmed.ncbi.nlm.nih.gov/10024360/', 'macaque experiment and theory', 'author abstract',
    'For two stimuli within a receptive field, attention shifted the response toward that evoked by the attended item alone. The findings supported biased competition between simultaneously represented stimuli.')
add('cohen', 'Cohen and Maunsell', 2009,
    'Attention improves performance primarily by reducing interneuronal correlations',
    'https://pubmed.ncbi.nlm.nih.gov/19915566/', 'macaque experiment', 'author abstract',
    'Population recordings in V4 during an orientation-change task linked much of the attentional improvement in population sensitivity to reduced shared variability, beyond changes in average firing rates.')
add('chelazzi', 'Chelazzi, Miller, Duncan and Desimone', 1993,
    'A neural basis for visual search in inferior temporal cortex',
    'https://pubmed.ncbi.nlm.nih.gov/8497317/', 'macaque experiment', 'author abstract',
    'Monkeys remembered a picture and selected its match from a small array. IT showed sample-related delay activity, followed during search by suppression of nontarget responses and dominance of the target representation.')
add('soto', 'Soto, Heinke, Humphreys and Blanco', 2005,
    'Early, involuntary top-down guidance of attention from working memory',
    'https://www.comp-psych.bham.ac.uk/publications/soto_jep2005.pdf', 'human experiment', 'author-hosted primary paper',
    'A remembered colored shape facilitated line search when it surrounded the target and impaired search when it surrounded a distractor. Effects extended to early responses and the first saccade.')
add('bichot', 'Bichot, Rossi and Desimone', 2005,
    'Parallel and serial neural mechanisms for visual search in macaque area V4',
    'https://doi.org/10.1126/science.1109676', 'macaque experiment', 'primary paper excerpt',
    'During free visual search, V4 responses favored objects sharing target features across the display. The experiment related feature-based parallel bias to selection during search among complex items.')
add('wolfe', 'Wolfe', 2021, 'Guided Search 6.0: An updated model of visual search',
    'https://doi.org/10.3758/s13423-020-01859-9', 'theory', 'author-hosted paper and publisher abstract',
    'Guidance combines stimulus features, target goals, history, reward and scene information into a changing priority map. Selection guides the order in which objects receive further processing and recognition.')
add('schall', 'Schall and Hanes', 1993,
    'Neural basis of saccade target selection in frontal eye field during visual search',
    'https://pubmed.ncbi.nlm.nih.gov/8247155/', 'macaque experiment', 'author abstract',
    'Initial FEF visual responses did not distinguish target from distractor. Activity subsequently evolved to identify the target before the eye movement, separating stimulus onset responses from selection.')
add('zaksas', 'Zaksas and Pasternak', 2006,
    'Directional signals in the prefrontal cortex and in area MT during a working memory for visual motion task',
    'https://pubmed.ncbi.nlm.nih.gov/17093094/', 'macaque experiment', 'author abstract',
    'Monkeys compared two motion directions across a delay. MT and PFC carried direction information with transient delay signals; test responses depended on the remembered sample and related to the eventual decision.')
add('ling', 'Ling, Liu and Carrasco', 2009,
    'How spatial and feature-based attention affect the gain and tuning of population responses',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC2696585/', 'human experiment and theory', 'primary article',
    'External-noise motion experiments separated spatial from feature-based attention. Threshold-versus-noise functions supported different combinations of population gain and tuning, giving behavioral tests of attentional mechanisms.')
add('stokes', 'Stokes and colleagues', 2013, 'Dynamic coding for cognitive control in prefrontal cortex',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC3898895/', 'macaque experiment', 'primary article summary',
    'Population analyses examined how prefrontal coding changed while establishing, maintaining and using task context. The study motivates measuring representational transformations through time rather than only mean persistent activation.')
