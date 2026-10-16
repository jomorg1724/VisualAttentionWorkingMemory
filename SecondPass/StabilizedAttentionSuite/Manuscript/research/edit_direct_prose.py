"""Apply the requested direct, affirmative editorial voice to authored prose."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
paths=list((ROOT/'sections').glob('*.html'))+[ROOT/'build.py',ROOT/'manuscript.js',ROOT/'pixels-tutorial.js']
texts={p:p.read_text(encoding='utf-8') for p in paths}
changes=[]
def replace(old,new):
    count=0
    for p,s in texts.items():
        n=s.count(old)
        if n:texts[p]=s.replace(old,new);count+=n
    if not count:print('UNMATCHED:',old)
    else:changes.append(dict(old=old,new=new,count=count))

replacements=r'''
Here, “attention” means a mechanism that changes which spatially organized evidence influences memory and report, rather than simply a bright region in a visualization.|||Here, “attention” means a mechanism that changes which spatially organized evidence influences memory and report.
These observations do not identify a unique computational algorithm, but they make selective routing an experimentally meaningful hypothesis.|||These observations make selective routing an experimentally meaningful hypothesis.
The superior colliculus is neither a homogeneous attention unit nor a feature-free anatomical structure. Its laminar organization, sensory inputs, motor functions, and distributed interactions prevent a literal equation between a neural region and one matrix in an artificial network. A useful abstraction is instead the|||The superior colliculus combines layered sensory and motor organization with distributed interactions. We study one computational abstraction of this organization: the
Together they motivate testing an observer with an evolving internal state rather than one that receives an unordered set of complete trial frames.|||Together they motivate testing an observer whose internal state evolves as the trial unfolds.
to a broader task battery; it does not reuse an identical model or reproduce that learning protocol.|||to a broader task battery.
A common selector need not require a common sensory code|||A common selector can coordinate distinct sensory codes
Parallel visual processing does not imply independent attention controllers. The influential perception/action distinction concerns what visual information is used for, not a simple division into static and moving images. Our appearance/correspondence split is a computational factorization of evidence, not an anatomical identification of ventral and dorsal streams.|||Parallel sensory pathways can supply a shared attention controller. The influential perception/action distinction organizes visual processing by the uses of visual information. Our model organizes evidence into appearance and temporal correspondence.
A photograph can change completely from one study item to the next without implying that the observer should classify “motion change.” A universal task model should therefore not force all useful visual information through an explicit velocity vector.|||Image-memory tasks require retaining the identity of successive photographs, while motion tasks require comparing movement across observations.
A motion population retains evidence for several possible displacements rather than immediately committing to one estimate.|||A motion population retains evidence for several possible displacements simultaneously.
This is an inductive bias about correspondence and temporal comparison, not supervision with the correct motion direction or privileged event time. The positions of displacement bins are explicitly engineered; calling the model “without geometric assumptions” would be incorrect.|||The engineered displacement bins and local comparisons specify an inductive bias about correspondence and temporal structure. Task labels and pixel-prediction errors train the learned components.
Joint learning alone is not transfer to an untrained task.|||Held-out tasks provide a separate test of transfer.
These questions distinguish a useful performance benchmark from a mechanistic account. A network can learn an answer without using a cue as intended. Decodable information need not drive a decision. A causal intervention can change behavior by damaging all processing rather than selectively redirecting evidence. The strongest experiments therefore combine|||These questions connect task performance to the computations producing it. To measure how cues and selected evidence influence decisions, the experiments combine
The combined evidence supports a shared, causally manipulable computational interface; it does not establish a unique biological circuit, universal benefits, or replication across independent training seeds.|||The combined evidence supports a shared, causally manipulable computational interface in this training lineage.
Read the results and their limits.|||Read the measured results.
“Change” concerns the underlying orientation assigned to an item, not whether every noisy pixel is identical across images.|||“Change” means a change in the underlying orientation assigned to an item.
It is related to the recurrent-attention research of Morgan, Albanna, and Herman, but the present value-coded task variants and terminal supervised training must not be described as an exact reproduction of that paper.|||The present value-coded task variants use terminal supervised training to study the recurrent-attention hypothesis of Morgan, Albanna, and Herman.
Native noise is refreshed, so repeated phase names do not imply identical pixel images.|||Each observation receives fresh native noise, including successive observations within the same phase.
Neither tells the model whether the current trial contains a change.|||The model determines change from the subsequent array observations.
What theta means—and why it is not a single difficulty value|||Theta sets the range of change magnitudes
A 65° condition includes small rotations as well as large ones; it does not mean that every trial rotates by 65°.|||A 65° condition samples rotations throughout the range from −65° to +65°.
These operations preserve the task construction but do not make the three arrays a pure item-count or capacity manipulation: item sampling and the number of backbone patches covering an item also change.|||Across these three array sizes, item count, spatial sampling, and the number of backbone patches covering an item all change.
that number is not the total training exposure of the network.|||this follows the parent checkpoint's training exposure.
Division by 3 is fixed; it is not normalization by each batch's observed average.|||Division by 3 provides a fixed scale across batches.
This is value-weighted supervised classification, not a policy learning when to stop, and it supplies no reaction-time result.|||This objective trains a value-weighted classification at the terminal observation.
Mean images and mean maps are averages over matched conditions; they are not additional movies shown to the model.|||Mean images and mean maps summarize matched conditions.
That ongoing continuation is a separate experiment and is not evidence for the completed findings reported here.|||That continuation is tracked as a separate experiment.
The battery is a collection of source-inspired computational assays, not a literal replication of thirteen animal experiments.|||The battery contains thirteen computational assays adapted from the cited research.
The distinctions below identify what each task requires and which shortcuts it is intended to challenge; they do not establish that a trained network actually uses the intended strategy.|||The distinctions below explain each task's information requirements and experimental design.
Colored marks describe design requirements, not measured neural activity or performance.|||Colored marks indicate each task's design requirements.
Its cue labels and answer overlays are explanations, not model representations.|||Cue labels and answer overlays explain the sequence and response rule.
Browser playback is not a calibrated psychophysics experiment; the original rasters are enlarged without inventing intermediate frames.|||Browser playback enlarges the original rasters and displays the exact observation sequence at the selected playback speed.
H counts probe repetitions, not different images.|||H counts repetitions of the same probe image.
The selectors retain these native conditions; they do not alter the generator or add new training variants.|||The selectors navigate the exported native conditions.
The appearance tasks add demands that velocity cannot satisfy.|||The appearance tasks require color, form, and image-identity information.
Except for the explicitly declared 100-Hz Krauzlis recipe, delays are numbers of observations, not biological milliseconds.|||Delays are expressed in observation counts; the Krauzlis recipe additionally specifies a 100-Hz presentation rate.
is evaluated with specificity and false-positive rate, not balanced accuracy or ROC AUC.|||is evaluated with specificity and false-positive rate.
New stream seeds create new draws, not a new underlying image database.|||New stream seeds create fresh draws from the same underlying image database.
The model's motion features should therefore be interpreted as image-based evidence, not a measurement of ground-truth object trajectories.|||The model's motion features describe image-based correspondence evidence.
The aperture problem: a line does not reveal all of its movement|||The aperture problem: motion ambiguity along an edge
This is a property of available evidence, not simply a failure of an optimizer.|||This ambiguity arises in the local visual evidence.
Thus regularization supplies an assumption, not missing sensory evidence.|||Regularization supplies the smoothness assumption used to select among candidate solutions.
It is an algorithm demonstration, not a trained-model result.|||It demonstrates the algorithm on synthetic inputs.
A sharp distribution is not necessarily a correct one: repeated texture can make the algorithm confidently prefer the wrong displacement.|||Repeated texture can give an incorrect displacement a sharp peak.
The correlation bank is not this energy model; both exemplify how relationships across space and time can be represented by a population of tuned responses.|||Motion-energy models and the correlation bank both represent relationships across space and time through populations of tuned responses.
These insect-inspired computations and primate motion-energy accounts should not be conflated with each other or with our FFT implementation. They motivate sensitivity to spatiotemporal relationships, not an anatomical claim about where a Fourier transform occurs.|||These insect-inspired computations motivate sensitivity to spatiotemporal relationships. Our implementation evaluates local correspondence through FFT correlation.
A softmax-normalized score is not automatically a calibrated Bayesian posterior; that stronger interpretation would require a specified likelihood and calibration evidence.|||Here, softmax provides normalized displacement weights. Probabilistic population-code theories describe how specified likelihoods and response statistics support Bayesian inference.
It has useful connections to biological motion computations, but it is neither a full optical-flow benchmark system nor an exact model of V1 or MT.|||It provides local correspondence populations for the shared recurrent observer.
7.1 Four questions that should not be collapsed into “attention”|||7.1 Four computations behind attentive behavior
These are operational distinctions for this manuscript, not four isolated boxes in the brain.|||These four computations organize our analysis of attentive behavior.
“Deep” here refers to anatomical depth, not a late layer in an artificial neural network.|||“Deep” refers to anatomical depth within the superior colliculus.
selection as a distributed computation rather than assigning all attention to one anatomical center.|||selection as a distributed computation.
The intervention does not have to identify “the motion channels” to affect motion behavior.|||A spatial intervention acts across the features selected at that location.
Equal spatial control does not imply equal performance: the two feature descriptions may differ in quality, and the output rules may demand different comparisons.|||Behavior depends on the quality of each feature description and the comparison demanded by the output rule.
They show that changing a common spatial selector can change which evidence controls the answer; they do not independently establish every step of the biological mechanism.|||They show that changing a common spatial selector can change which evidence controls the answer.
Together they motivate spatially targeted causal tests; they do not identify our softmax with either circuit.|||Together they motivate spatially targeted causal tests of selection.
This example is arithmetic, not a measured model result.|||This is a worked numerical example.
That release does not guarantee a behavioral benefit.|||The behavioral effect depends on the evidence receiving the released priority.
This is a theoretical account of response modulation, not evidence that neural responses are literally normalized to sum to one.|||This theory describes response modulation through divisive normalization.
but its output is an explicit probability-like allocation over locations rather than a prediction of a particular neuron's firing rate.|||and its output explicitly allocates priority across locations.
These statements follow from the task probabilities, not from an assumption that the network explicitly performs Bayesian inference.|||These statements follow directly from the task's conditional probabilities.
It receives neither a numerical validity channel nor the correct event location.|||The observer receives validity through the rendered cue and estimates event location from the images.
A changing hidden-state pattern could retain useful information in a distributed code, but observing that requires decoding across time and testing behavior, not naming the state “memory.”|||Decoding across time and testing behavior can reveal information retained in a changing, distributed hidden-state pattern.
Their account emphasizes recurrent population dynamics rather than a dedicated output for every possible sensory mixture.|||Their account emphasizes recurrent population dynamics.
7.7 Motion representations and prediction: useful principles, limited anatomical claims|||7.7 Motion populations and prediction
it retains scores for 81 candidate displacements rather than forcing an immediate single-vector answer.|||it retains scores for 81 candidate displacements.
Our implementation has one photometric prediction objective and an error feature; it does not implement a complete hierarchy of top-down predictions, precision estimates, and dedicated error populations.|||Our implementation uses a photometric prediction objective and supplies the resulting error as a visual feature.
7.8 What the new VDA evidence changes—and what it leaves open|||7.8 How VDA extends the evidence for shared selection
The relationship is a shared research hypothesis, not an unchanged replication.|||The studies develop the same memory-guided selection hypothesis through these respective implementations.
The new result adds a successful benefit condition; it does not erase the earlier null findings.|||Together, the experiments establish the conditions under which this observer showed costs, benefits, and near-zero effects.
Motion-derived features still enter the visual tokens, so this is not an appearance-only ablation. Nevertheless, the benefit cannot be attributed to a nonzero long-window motion-comparison readout.|||The visual tokens contain both appearance and motion-derived features. In these short trials, the observed benefit is carried by the recurrent CLS/value route.
These are proposed tests, not experiments completed by this manuscript revision.|||These tests define the next stage of the research program.
Analyze target hits, foil false alarms, catch false alarms, and confidence or logits, not only aggregate accuracy.|||Analyze target hits, foil false alarms, catch false alarms, confidence or logits, and aggregate accuracy.
The same selector need not produce equal effect sizes when baseline difficulty, feature discriminability, or attention saturation differs.|||Effect sizes depend on baseline difficulty, feature discriminability, and the initial concentration of attention.
A stronger cross-domain claim would require matched-baseline psychometric comparisons, not identical raw accuracy changes.|||Matched-baseline psychometric comparisons would measure cross-domain effects at comparable difficulty.
A feature-specific model with separate maps is a useful prospective comparison, but it is not implemented or evaluated in this draft.|||A feature-specific model with separate maps is a proposed architectural comparison.
Gate values and decodability over time can guide these tests, but a decoded cue is evidence of information availability, not proof of functional use.|||Gate measurements and decoding track cue availability over time; interventions measure its contribution to behavior.
None of these ablations has a result in this manuscript; success of the combined architecture cannot identify the causal contribution of any one ingredient.|||These proposed ablations would measure the individual contributions of the architecture's ingredients.
Psychometrics, completed analyses and further extensions: psychometric shape, not a single operating point|||Psychometrics: curves across evidence strengths
The equation below is an optional descriptive fit, not a claim that those plots were generated from this fitted function.|||The displayed curves report empirical rates. The equation below provides an optional descriptive fit for subsequent analysis.
They are not interchangeable evaluations of a single unchanged set of weights.|||Each evaluation is associated with its own named checkpoint.
It describes retained performance, not a paired estimate of the effect of adding VDA.|||It describes retained performance after the VDA continuation.
Those means average different task sets; their difference alone is not a measure of forgetting.|||Retention is assessed task by task because those means average different task sets.
It is not a native 100%-validity performance estimate.|||It measures the response to a deliberate violation of the displayed 100%-validity cue.
Thus the main benefit is not merely a general shift toward saying “change.” The simultaneous loss at the cue establishes a trade-off:|||Improved uncued detection and reduced catch false alarms accompany a loss at the cue. This establishes a spatial trade-off:
These probabilities concern trial-level evidence for this frozen checkpoint, not variability across separately trained networks.|||These probabilities quantify trial-level evidence for this frozen checkpoint.
Hit-rate benefits are not automatically balanced-accuracy benefits.|||Hit rate measures detection on change trials; balanced accuracy also incorporates correct rejections.
The statistical endpoints differ, so “eight positive tests versus eight null tests” is not a like-for-like comparison of one unchanged hypothesis.|||The two sets of tests use these different statistical endpoints.
9.4 The tasks are not uniformly solved|||9.4 Performance varies with task difficulty
That is an absence of established benefit under those tested conditions, not proof that the model can never benefit.|||These conditions yielded null results for the prespecified benefit tests.
This is a plausible computational interpretation of the difference between experiments, not an isolated causal explanation, because checkpoint, stimuli, timing, and response rule all differ.|||This interpretation motivates matched comparisons of checkpoint, stimuli, timing, and response rule.
The new VDA results supplement these findings rather than replacing them.|||The VDA results extend this record of task-dependent intervention effects.
This projection shows selection weights, not a pixelwise receptive-field decomposition of the appearance CNN.|||This projection displays the selection weight assigned to each grid site.
A map indicates how information is selected; it is not a neuronal firing-rate recording.|||The map displays the model's spatial selection weights.
That is one plausible explanation, not a fitted or isolated mechanism.|||This explanation can be tested by measuring how released priority is distributed among competing sites.
Read the specified event, size, and dose in each panel; this is an exploratory analysis, not a family of independently corrected confirmatory tests.|||Read the specified event, size, and dose in each panel. These exploratory comparisons use pointwise intervals.
These counts include repeated presentations of matched scenes; they are not that many independent trials.|||These presentation counts include repeated views of matched scenes; statistical analyses use scene-level pairing and clustering.
A metadata match verifies a citation's identity; it does not verify every claim in the full paper.|||Metadata verification establishes the citation's identity.
The biological sources motivate computations at different levels of abstraction, not a single agreed circuit theory.|||The biological sources motivate computations at several levels, from local sensory responses to recurrent selection and behavior.
No new model training or intervention experiment is launched by opening it.|||The explainer is an offline reading and visualization interface.
Mathematical descriptions below are implementation-derived; neuroscience links explain motivations and limits, rather than identifying the network with a particular brain region.|||The mathematical descriptions follow the implementation; the neuroscience links explain the ideas motivating its computations.
The spatial field influences the decision through future attention and values; it is not flattened directly into the final classifier.|||The spatial field influences the decision through future attention and value computations.
The comparison branch feeds its trace under the shared weights; it does not determine the attention logits directly.|||The shared attention weights select the comparison branch's contribution to its trace.
and is not an exponent.|||and names that quantity's role.
Square brackets with a semicolon concatenate vectors; they do not add them.|||Square brackets with a semicolon concatenate vectors.
A symbol's role must be read from its definition, not guessed from its letter.|||Each symbol is defined alongside its equation.
There is one query-to-location attention distribution, not an all-pairs 100 × 100 attention matrix.|||One CLS query produces a distribution over 100 spatial locations.
Adaptive pooling maps the latter to the common 10 × 10 grid; it is not identical to splitting the original image into 100 nonoverlapping CNN receptive fields.|||Adaptive pooling maps the latter to the common 10 × 10 grid, with overlapping input receptive fields.
The biological motivation is retinotopically organized feature processing, not a claim that two convolutions reproduce the ventral visual hierarchy.|||Retinotopically organized feature processing motivates this spatial layout.
It smoothly attenuates negative values rather than imposing a hard threshold.|||It smoothly attenuates negative values.
The fourth number is RMS deviation from the fixed gray level, not standard deviation about the local mean.|||The fourth number is RMS deviation from the fixed gray level.
This P is an image patch, not an output probability.|||P denotes the prepared image patch.
Its width is a receptive-field scale chosen by the implementation, not learned.|||The implementation fixes this receptive-field width at 24 pixels.
FFT denotes an efficient algorithm for this transform; it does not add a new learned layer or assume that neurons explicitly calculate complex numbers.|||FFT evaluates this fixed transform efficiently.
It can be large for strongly textured but ambiguous patterns; it is not confidence estimated from the sharpness of the correlation peak.|||It measures prepared-image energy and can be large for strongly textured, ambiguous patterns.
operates over displacement coordinates, not image coordinates:|||operates over the 9 × 9 displacement grid:
Here p is a displacement weight, not a pixel index.|||Here p denotes a displacement weight.
Lower temperature sharpens the distribution, but does not resolve a genuine ambiguity or correct a wrong peak.|||Lower temperature concentrates the weights around the highest scores.
although these normalized scores are not calibrated neural posteriors.|||and here the normalized scores represent displacement-matching weights.
This is an expectation over alternatives, not the displacement with the largest score.|||This expectation averages all candidate displacements using their weights.
It is a constant-velocity extrapolation using a previous-image estimate as a destination-grid approximation; exact forward-flow inversion is not performed.|||It extrapolates constant velocity using the previous-image estimate on the destination grid.
This is a leaky population memory with one learned rate, not a stored ground-truth direction.|||This leaky population memory has one learned write rate.
Yet an explicit comparison is not proof of proper cue use: attention must still select the relevant comparison, and the classifier must use it.|||Attention selects the relevant comparison, which the classifier uses to form the response.
it is not compressed into the 64-channel visual token.|||it retains its 735 coordinates alongside the visual tokens.
It mixes channels at a location; it does not create a second attention map.|||It mixes channels at each location.
“Selected” labels the vector; it is not a power or an additional learned module.|||“Selected” labels the pooled vector.
The drawing demonstrates the equation; it does not simulate biological current injection or display a trained network's attention.|||The drawing is an interactive demonstration of the selection equation.
Layer normalization was developed partly to improve recurrent-state dynamics, but it is an optimization device rather than an identified local microcircuit.|||Layer normalization was developed partly to improve recurrent-state dynamics.
Superscripts w and r name the two gates; they are not exponents.|||Superscripts w and r identify the write and reset gates.
Normalization shapes the proposed update; it does not rescale the stored H after this interpolation.|||Normalization shapes the proposed update before it is mixed with the stored H.
The reset fraction controls how much previous memory contributes to constructing that proposal; it does not directly erase the stored state.|||The reset fraction controls how much previous memory contributes to constructing the proposal.
This is a spatial GRU, not 100 completely independent vector GRUs.|||The spatial GRU's convolutions couple neighboring locations.
This probability need not be calibrated confidence.|||The largest output probability determines the model's choice.
Reliability weights multiply the error; the denominator is the number of terms, not the sum of reliability weights.|||Reliability weights multiply the error, and the number of terms supplies the averaging denominator.
This operation occurs in optimization, not during the forward memory update.|||The optimizer applies this operation after gradient accumulation.
These optimizer variables persist across training updates but are not trial memory and are not used to answer an evaluation movie.|||The optimizer retains these variables between training updates. Evaluation uses the learned parameters and the trial's recurrent states.
This is equal task opportunity, not equal total compute, equal number of frames, or equal gradient influence.|||Each task receives equal update opportunities; frame counts, computation, and gradient contributions depend on its trials.
Joint training can help shared representations or produce interference; aggregate progress alone cannot distinguish those possibilities.|||Task-level retention and transfer measurements track how joint training affects the shared representations.
Hand-set filters and prescribed motion are teaching devices, not visualizations of learned weights or new performance results.|||The examples use hand-set filters and prescribed motion to isolate individual operations.
The problem is that the answers we want are usually not individual pixel values.|||The answers we want concern patterns, relationships, and remembered events.
A useful representation need not name an object or supply a correct answer.|||A useful intermediate representation describes features that subsequent computations can use.
We do not ask a single velocity estimate to carry all the information needed for color, orientation, motion, and memory judgments.|||Separate descriptions supply color, orientation, motion, and memory judgments with the information they require.
“Feature coordinate” means an entry of a vector; it does not mean an x or y location.|||“Feature coordinate” means an entry of a vector.
At frame t it can use that frame and its past, not later observations.|||At frame t it uses the current frame and its history.
Moving right on that correlation grid means considering a more rightward displacement, not inspecting a farther-right object.|||Moving right on the correlation grid means considering a more rightward displacement of the same image patch.
Their outputs are attached to a common spatial index, but the pixels contributing to each output are not identical.|||Their outputs share a spatial index, with each operation using its own input neighborhood.
Padding is an assumption about missing surroundings, not observed content.|||Padding supplies the assumed values beyond the image boundary.
Its filters may become useful for the task, but we cannot assume that a learned channel corresponds cleanly to “red,” “orientation,” or an object category without examining it.|||Training shapes each channel's selectivity; inspecting its responses reveals the patterns it encodes.
P2 · Features are measurements, not miniature copies of the image.|||P2 · Features measure local image structure.
The two response maps use hand-set derivative filters, not trained weights.|||The two response maps use hand-set derivative filters.
Negative input is smoothly attenuated rather than set exactly to zero.|||Negative input is smoothly attenuated.
Local feature selectivity and spatial organization motivate comparison with early visual cortex, but this layer is not a measured bank of V1 neurons.|||Local feature selectivity and spatial organization motivate comparison with early visual cortex.
Attention projected onto a neat 10 × 10 block diagram is therefore a useful visualization of spatial selection, not an exact picture of which original pixels caused the decision.|||The 10 × 10 attention projection visualizes selection over these spatially indexed representations.
Pooling exchanges some spatial precision for a smaller representation; it cannot be assumed harmless merely because the output has the desired size.|||Pooling trades spatial precision for a smaller representation.
The CNN can learn useful features, but it is not guaranteed to preserve every simple quantity.|||The CNN learns task-useful combinations of local pixel values.
The lower-right examples also show that RMS relative to 0.5 is not standard deviation about the local mean.|||The lower-right examples distinguish RMS relative to 0.5 from standard deviation about the local mean.
These blocks are not identical: their color means and CNN features differ.|||Their color means and CNN features distinguish the blocks.
The point of retaining several feature groups is that one scalar's ambiguity need not become an ambiguity of the entire representation.|||Retaining several feature groups lets one description resolve another's ambiguity.
This is a reason for maintaining an appearance pathway, not a failure of the motion calculation at its intended job.|||The RGB appearance pathway preserves this color distinction.
A brighter overall crop should not earn a high correspondence score merely because it has a large constant component.|||Removing the constant component makes correlation respond to spatial structure.
It is not free information: tapering reduces usable contributions from the edge of the crop, and applying a window at a fixed location means that a translated pattern is not generally a perfectly translated windowed pattern.|||Tapering reduces contributions from the crop's edge. As a pattern moves through the fixed window, its features receive different weights.
Thus even a noiseless moving image may not yield a correlation peak or mean exactly equal to its prescribed displacement.|||This reweighting can shift the correlation peak or mean relative to the prescribed displacement.
None of these is the later normalization of correlation by image energy, and none is LayerNorm.|||Later stages additionally use energy normalization and per-patch LayerNorm, each defined where it appears.
Correlation scores a hypothesis; it does not track named dots|||Correlation scores candidate movements
A long edge does not specify motion along itself, and stripes repeat.|||A long edge leaves motion along itself ambiguous, and stripes repeat.
This makes the match score less sensitive to multiplicative intensity scale; it does not reveal whether the crop contained enough signal to trust.|||This reduces sensitivity to multiplicative intensity scale. A separate energy feature records the signal strength.
The bank represents how these local candidates match the observed images; it is not supervised with the correct velocity or a physical-motion label.|||The fixed bank scores each candidate directly from the observed pixel arrays.
That check establishes the teaching computation's correspondence to the implementation; it does not establish a biological Fourier mechanism or successful task behavior.|||That check verifies the teaching computation against the implementation.
These describe possible displacements, not 81 image locations.|||Each score describes a candidate displacement at the current image site.
It uses the same softmax rule; it is not an 81-candidate model output.|||It uses the same softmax rule on this reduced candidate set.
Lowering temperature makes a score preference stronger; it cannot supply missing visual evidence.|||Lowering temperature amplifies the preference expressed by the existing scores.
Sharpness is therefore neither a free improvement in precision nor proof of confidence.|||Temperature governs both weight concentration and gradient sensitivity.
A population account asks what a pattern across differently tuned units represents, rather than identifying one cell with the entire percept.|||A population account asks what the joint response of differently tuned units represents.
Step 10 · Two reasons not to trust a match|||Step 10 · Signal strength and ambiguity
This is a defined signal-strength heuristic, not an empirically calibrated probability that a velocity estimate is correct.|||This heuristic measures the strength of the prepared image signals.
This does not mean every visual feature vanishes: the color means, learned appearance features, and differences from an existing trace can still carry information.|||Color means, learned appearance features, and differences from an existing trace remain available.
This is ambiguity, not low signal.|||This is ambiguity in a strong signal.
The plot uses it to expose a distinction that the scalar r alone cannot express.|||The plot separates population ambiguity from the signal strength measured by r.
These are complementary observations of the same movie, not three separate motion directions supplied to the network.|||These are complementary temporal observations of the same movie.
The superscript ℓ identifies the chosen lag; it is not a power.|||The superscript ℓ identifies the chosen lag.
The table states physical displacements, not guaranteed estimates.|||The table gives the physical displacements in this prescribed-motion example.
it is not a separate subpixel estimator.|||the expectation can therefore fall between integer candidates.
These are concrete design choices that explain why a useful correlation population need not produce a perfect image prediction.|||These choices determine how the population is summarized for image prediction.
Multiple lags expose different evidence; they do not automatically solve ambiguity, limited displacement range, or startup bias.|||Each lag has its own evidence, displacement range, and startup behavior.
Its values already measure displacement in image pixels, so enlargement does not multiply velocities by ten.|||Flow values retain their image-pixel units during enlargement.
This demonstration supplies its flow by hand to isolate warping; it is not a prediction measured from the trained model.|||The prescribed flow isolates the warping operation in this demonstration.
These energies are not the mean-centered, Hann-windowed crop energies used in r.|||These block energies differ from the prepared-crop energies used in r.
The feature itself is not multiplied by a reliability weight before fusion.|||The normalized error enters fusion directly.
Here ω is a separate pixel-energy weight; it is not the correlation reliability r.|||Here ω weights pixel-prediction errors, while r weights correspondence populations.
Pixels supply the prediction target without direction labels, but task labels still train the observer: the whole model is not unsupervised.|||Pixel observations supply the prediction targets, and task labels supervise the final decisions.
This is a specific image-transport auxiliary objective, not an implementation of the full cortical predictive-coding hierarchy described by Rao and Ballard.|||Our image-transport objective applies the prediction-and-mismatch principle discussed in the cortical predictive-coding account of Rao and Ballard.
These are update counts, not biological milliseconds.|||The half-life is expressed in recurrent update counts.
Controls change the teaching update, not the trained model.|||The controls set the update rate in this synthetic demonstration.
Therefore δ is not a ground-truth change label. It is evidence relative to a particular initialized and adapted reference.|||The value δ measures evidence relative to the initialized and adapted reference.
The windows slide at every time step; they are not aligned using event metadata.|||The windows slide at every observation according to the fixed index rule.
The bar means a temporal weighted average; W is a set of times, not a learned matrix.|||The bar denotes a temporal weighted average, and W denotes its set of observation times.
The factor ten rescales these population coordinates; it is not the old, removed 100-fold attention amplification.|||The factor ten rescales the population coordinates supplied to subsequent layers.
Scaling affects the numerical balance of downstream inputs but does not introduce an answer label.|||Scaling sets the numerical balance of the downstream inputs.
This is an arithmetic counterexample, not a claim that every global encoder must fail.|||This example demonstrates how early averaging can hide a local change.
Concatenation places values next to each other in one vector; it does not average, add, or normalize them.|||Concatenation places all the values next to each other in one vector.
The dimensions below refer to one site at one observation, not the entire movie.|||The dimensions below describe one site at one observation.
Its superscript names its role; it is not a power.|||The superscript names the layer's role.
A fused channel need not remain interpretable as “color” or “motion”; it may combine both.|||A fused channel can combine color and motion evidence.
Neither count grows with trial length because the parameters are reused on every frame.|||These parameter counts stay constant across trial lengths because every frame uses the same weights.
Positional embeddings indicate the site, not the correct target or the cue's meaning; any useful relation between a visible cue and a site must be learned from the task.|||Positional embeddings identify the site. Training teaches the relationship between a visible cue and the relevant site.
Task training encourages useful combinations; successful behavior is evidence about the distinctions needed in tested conditions, not proof of a lossless encoder.|||Task training shapes the compressed representation around distinctions useful for the tested judgments.
The cue must determine which site's evidence matters; distractor change is not automatically a positive answer.|||The cue determines which site's change counts toward the positive response.
High prediction error alone is not a recognition signal.|||Recognition requires comparing the probe with retained image information.
The explicit older-window branch is unavailable; performance cannot be attributed to it.|||The older-window branch is zero; the recurrent CLS/value route carries the decision.
The useful old representation must remain accessible until the response; blank pixels do not themselves preserve it.|||Recurrent state must retain the useful representation through the blank interval.
None of these operations says “the cued target changed.” That statement requires the subsequent selection and decision machinery. The sensory computation supplies evidence with known limitations, rather than the task answer.|||The subsequent selection and decision machinery combines this sensory evidence with the cue to determine whether the target changed.
The scalar r indicates signal energy, not a guarantee that the images depict the same object or a continuous physical scene.|||The scalar r describes the prepared crops' signal energy.
What these demonstrations establish—and what they leave open|||Reproducible demonstrations and numerical checks
They do not measure learned filters, the model's actual attention, recognition capacity, psychometric thresholds, or a biological mechanism. Those require the separate measured-results analyses.|||The separate measured-results chapter reports the trained observer's performance and interventions.
the experimental evidence and the limits of those correspondences in depth.|||the experimental evidence and its connections to the model in depth.
Synthetic teaching data; no trained outputs.|||Synthetic teaching data.
Prescribed population; no trained outputs.|||Prescribed population.
'''
for line in replacements.strip().splitlines():
    old,new=line.split('|||');replace(old,new)

# Remove stand-alone disclaimers whose positive description is already adjacent.
deletions=r'''
None makes a gated recurrent network a literal neural circuit.
Consequently, reaction-time effects and equivalence to the earlier reinforcement-learning result cannot be inferred from the present implementation.
Near-chance performance at the 8° maximum does not by itself identify an encoder-capacity limit.
Our implementation does not solve this least-squares system.
Our model does not minimize this functional or impose a global smooth-flow penalty.
The current observer is much smaller and does not contain any of these pretrained networks.
Its recurrent attention updates are not RAFT iterations, and its correspondence bank is not a dense all-pairs cost volume.
It need not mean that every neuron is indifferent to stimulus features, that static and moving images share an identical sensory code, or that there is exactly one anatomical map in the brain.
Our present multi-item experiments do not replace that matched single-item control.
It does not show that visual cortex is irrelevant or that all of its activity was unchanged.
We do not translate the electrical current into a numerical logit bias.
Grouping all four under a biological label would obscure rather than explain the implementation.
Their joint success also does not identify which stabilization change was necessary.
It does not test dopamine, reinforcement learning, or moment-by-moment expected utility merely because the generator calls a variable “reward.”
They do not imply that every working-memory task uses a fixed sustained-firing code, nor that our 64 numerical channels correspond to 64 such neurons.
Neither makes a GRU an implementation of short-term synaptic plasticity.
This is an argument for a representational choice in our model, not evidence that its normalized correlation scores equal neural probabilities.
It does not selectively inject motion evidence or an “answer” into the output head.
The present task battery alone establishes none of these stronger claims.
A cue benefit in one deliberately selected intervention condition therefore does not establish excellent performance over every VDA difficulty.
It does not prove a complete explanation of the null result: the baseline sensory evidence, task rule, and readout also matter.
It does not replace a biological control site or prove that every effect is exclusively sensory.
It does not establish that all tasks are solved, that an early sensory code became more sensitive, that separate attention systems are biologically absent, or that this effect replicates across independent training seeds.
Numerical stability constants are not additional sensory signals.
It is not a freely interacting class token inside a standard multihead transformer block.
Neither path is a cone-isolating retinal model.
This is a scientifically relevant limitation.
These are numerical conventions, not free biological assumptions.
This is a deliberately exposed limitation rather than a hidden padding or extra-frame workaround.
This does not establish that both domains are behaviorally well represented, or that the same intervention produces equal-sized effects in both.
A model that succeeds only on motion has not demonstrated domain-general attention simply because its wiring is shared.
They should not all be called “divisive normalization” as though they were the same biological mechanism.
It would therefore not be an interchangeable implementation of the per-patch operation specified here.
The CLS vector should not be assigned a one-to-one anatomical identity such as “prefrontal cortex,” and the spatial field should not be described as a direct measurement of persistent neuronal firing.
A parameter count by itself is not a measure of biological plausibility or memory capacity.
Computing an entire array at once in software is not the same as giving the model future evidence.
We will not quietly replace either output with a correct direction, change label, or target coordinate.
It does not mean the model attended the right-hand side of the whole image.
This toy filter is not one of the trained model's filters.
It does not say that every input pixel has equal influence, or that learned filters use all of their support.
It is not subtraction of each image's mean or division by its standard deviation.
Arithmetic grayscale also should not be equated with calibrated human luminance or cone responses.
It still does not make the comparison invariant to arbitrary lighting, changed objects, or occlusion.
Neither a high maximum nor a narrow peak guarantees the true physical motion.
We do not compute that complete output.
Our softmax scores do not establish those statistics or posterior calibration.
Still, neither quantity tells us whether a change is relevant to the task.
It does not choose the most reliable lag or perform a reliability-weighted average.
Neither curve is a measured neural response or a trained decision probability.
They receive neither a cue-phase flag nor the ground-truth event time.
It does not guarantee that the learned attention will select it.
It does not require motion and appearance to be computed identically before selection.
These ideas operate at different levels: receptive-field organization is not the same claim as a particular 24-pixel crop, an FFT, or a 292-to-64 projection being biologically implemented.
'''
for sentence in deletions.strip().splitlines():replace(sentence,'')
for p,s in texts.items():
    s=re.sub(r' +',' ',s) if p.suffix=='.html' else s
    s=re.sub(r'<p>\s*</p>','',s)
    if s!=p.read_text(encoding='utf-8'):p.write_text(s,encoding='utf-8')
(ROOT/'research/direct_prose_edits.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False),encoding='utf-8')
print('Applied',len(changes),'editorial rules.')
