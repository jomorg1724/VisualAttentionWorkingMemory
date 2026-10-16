from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
paths=list((ROOT/'sections').glob('*.html'))+[ROOT/'build.py']
texts={p:p.read_text(encoding='utf-8') for p in paths}
changes=[]
rows=r'''
Technically this is cross-correlation because the kernel is not reversed.|||The library computes cross-correlation, applying the kernel in its stored orientation.
These nonlinearities permit computations that a sequence of purely linear maps cannot express.|||These nonlinearities let the response depend on combinations of feature activation levels.
The distinct symbol \(I_t^{\rm gray}\) always denotes their grayscale average, not the calibrated luminance coordinate used by the chromatic stimulus generator:|||The symbol \(I_t^{\rm gray}\) denotes the arithmetic average of the RGB channels:
This is not the complete 47 × 47 linear correlation of two 24 × 24 patches.|||The bank retains the central ±4-pixel shifts needed for its candidate set.
The prediction of frame \(t\) uses the flow available at \(t-1\), not the flow computed from frame \(t\):|||The prediction of frame \(t\) uses the flow already available at \(t-1\):
Error epsilon here is not the small numerical epsilon used in vector normalization later.|||Here epsilon denotes the normalized prediction error.
Predictive-processing theories motivate sensitivity to mismatch, but our scalar mismatch feature and auxiliary photometric objective are not a complete hierarchical predictive-coding circuit.|||Predictive-processing theories motivate sensitivity to mismatch. Here, a scalar mismatch feature and an auxiliary photometric objective implement that comparison.
W labels a reference window R or a recent window O; it is not a learned weight matrix here.|||W identifies either the reference window R or the recent window O.
this permutation does not change the described information but fixes checkpoint compatibility.|||this ordering fixes the coordinates used by the saved checkpoints.
braces mean include all three lags, not average them.|||braces mean concatenate the entries for all three lags.
The later manipulated map therefore cannot generally be recovered just by multiplying the saved sham map by a bias.|||Computing later manipulated maps requires rerunning the recurrence with the intervention in place.
It does not set attention to −4, subtract four percentage points, or necessarily silence the region completely.|||The softmax converts this logit shift into a change in the region's normalized priority.
Consequently, “inhibition must always impair” is not an architectural prediction.|||The behavioral effect of inhibition depends on which competing evidence gains priority.
The original suite and VDA analyses provide task performance and matched interventions, but not an independent-seed, matched-difficulty cross-domain replication.|||The original suite and VDA analyses provide task performance and matched interventions within the reported checkpoint lineage.
if the manipulation redirects evidence rather than simply increasing a generic “change” response.|||when the manipulation redirects evidence toward different locations.
A common recurrent map may be useful at more than one phase, but a maintenance perturbation is not the same test as preventing initial encoding.|||Phase-specific perturbations test the map's roles in initial encoding, retention, and retrieval separately.
These a and b belong only to this curve, not the intervention mask or model bias vectors.|||Here a and b are the threshold and slope parameters of this curve.
Here \(\gamma\) and \(\lambda\) describe floor and lapse terms; they are not automatically fixed to chance for a yes/no response rate.|||Here \(\gamma\) and \(\lambda\) describe the curve's floor and lapse terms, chosen for the response rate being fitted.
A 75% hit threshold is not a 75% balanced-accuracy threshold.|||A hit threshold specifies detection on positive trials; a balanced-accuracy threshold additionally incorporates correct rejections.
Changes in these summaries do not uniquely localize the affected computation.|||These behavioral summaries guide subsequent tests of the underlying computation.
bootstrap independent scenes or training seeds rather than counting correlated variants as independent observations.|||bootstrap at the level of independent scenes or training seeds.
A task average should not hide a failing delay or load condition.|||Report each delay and load condition alongside the task average.
It is not automatically a velocity in degrees of visual angle per second.|||Conversion to degrees of visual angle per second requires a calibrated spatial scale and frame rate.
The physical motion of objects, the projection of that motion onto the retina, and the measured optical flow need not agree.|||Physical object motion, retinal projection, and measured optical flow describe different stages of image formation and estimation.
It is an assumption about image formation, not an identity that all movies satisfy.|||This image-formation assumption defines the brightness-constancy model.
denote derivatives of intensity with respect to x, y, and time—not three frames or RGB channels.|||denote derivatives of intensity with respect to x, y, and time.
One scalar constraint cannot generally determine both unknown velocity components.|||The scalar constraint leaves one degree of freedom in the two-component velocity.
It does not solve for both components of velocity: one scalar equation contains two unknowns.|||One scalar equation contains two unknown velocity components, leaving a family of possible motions.
Small movements and differentiable intensity are assumptions of the linearization, not guaranteed properties of arbitrary video.|||The linearization assumes small movements and differentiable intensity.
This E is an objective, not a neural energy measurement.|||E denotes the optimization objective.
The inverse formula applies only when G is invertible; otherwise a unique local velocity cannot be recovered by this expression.|||An invertible G gives a unique solution; a singular G leaves ambiguity along one or more directions.
This explains both the utility and the limitations of fitting a common local velocity.|||The spatial distribution of image gradients therefore determines how well a common local velocity can be estimated.
Normalization reduces dependence on multiplicative intensity scale, but does not make correspondence invariant to occlusion or arbitrary appearance changes.|||Normalization reduces dependence on multiplicative intensity scale. Occlusion and appearance changes alter the correspondence evidence itself.
The + and − label directions, not positive and negative pixel intensities.|||The + and − labels identify the two motion directions.
Two equal peaks at \((-3,0)\) and \((3,0)\) average to zero even though neither hypothesis is stationary.|||Two equal peaks at \((-3,0)\) and \((3,0)\) represent opposing motions whose weighted mean is zero.
Earlier motion and cued-orientation tests did not establish their primary balanced-accuracy benefits.|||Earlier motion and cued-orientation tests yielded null results for their primary balanced-accuracy benefit endpoints.
It does not ensure diffuse attention: a large difference between bounded logits can still produce a nearly concentrated softmax.|||Large differences between the bounded logits can still concentrate nearly all priority at a few sites.
This is a property of that trace, not the retention time of the whole network.|||This half-life describes the direct retention in the comparison trace.
The spatial recurrent unit receives the full value field, not \(100\alpha v\).|||The spatial recurrent unit receives the full value field, with attention controlling its write modulation.
for channel j, not attention gains or optimizer moment coefficients.|||for channel j.
Statistics are not pooled across spatial positions, batch examples, or time steps.|||Each site, example, and observation uses its own channel statistics.
; it is not the earlier token \(V_{t,i}\).|||.
It does not prove that the complete recurrent Jacobian is contractive: gate values, candidates, attention, and future keys all depend on the previous state.|||The full recurrent dynamics also include the dependence of gates, candidates, attention, and future keys on previous state.
It teaches the direct retention path; a trained recurrent network has state-dependent gates and cannot be reduced to this curve.|||The demonstration isolates the direct retention path. In the trained network, the gates vary with input and recurrent state.
rather than spatial convolutions.|||to update the global vector.
Its write fraction is not multiplied by \(g\): spatial selection has already occurred in \(z_t\).|||Spatial selection occurs in \(z_t\), which enters the CLS update with its learned write fraction.
The comparison with persistent activity, dynamic coding, or activity-silent memory concerns functional retention, not a claim that one implementation settles the biological debate.|||Persistent activity, dynamic coding, and activity-silent memory provide complementary perspectives on retaining task-relevant information.
is the shared 64-entry decision representation; it is not the spatial memory h.|||is the shared 64-entry decision representation.
Softmax normalizes over answer classes k, not image locations.|||Softmax normalizes over the answer classes indexed by k.
Inactive task heads have no gradient and do not advance their Adam clocks.|||Inactive task heads retain their current parameters and Adam states.
The fixed correlation bank has none.|||The fixed correlation bank contributes zero learned parameters.
It does not show how accurately orientation was encoded there, how long its contents will remain useful, or whether the output head will use them correctly.|||Encoding fidelity, retention, and use by the output head are measured through the corresponding feature and behavioral analyses.
Likewise, recovering a cue from hidden state demonstrates availability to a probe, not that the original decision circuit reads it.|||Decoding measures cue availability in hidden state; interventions test its contribution to the original decision.
even when the eyes do not move.|||during maintained fixation.
Improving its sensory encoder would then address an available signal rather than the failed use of that signal.|||This pattern directs investigation toward the selection and use of the available signal.
Binary signal-detection summaries can describe this behavior, but do not by themselves establish that the intervention improved an early sensory representation.|||Binary signal-detection summaries quantify the behavioral changes in sensitivity and response criterion.
Our VDA benefit is especially useful because detection at the uncued site improves while catch false alarms fall; a uniform increase in willingness to say “change” does not explain that combination.|||Our VDA result combines improved uncued detection with fewer catch false alarms, a pattern of improved discrimination for that location.
A 0% valid cue excludes the cued location when an event occurs; it is not an uninformative cue.|||A 0% valid cue identifies the cued location as excluded whenever an event occurs.
If it cannot be recovered, that could reflect lost information or an inadequate retrieval mechanism.|||A retrieval failure motivates tests of both information retention and the retrieval mechanism.
Averaging equally plausible leftward and rightward alternatives produces zero displacement, which is not the same as evidence for a stationary object.|||Equally plausible leftward and rightward alternatives have a zero mean while retaining two opposing peaks in the population.
Earlier motion and cued-orientation interventions showed large costs but did not establish the predeclared balanced-accuracy benefits.|||Earlier motion and cued-orientation interventions showed large costs and null results on the predeclared balanced-accuracy benefit tests.
Establishing a common biological mechanism would require evidence beyond one architecture and one training lineage.|||Independent training lineages and cross-domain biological experiments provide further tests of the common-mechanism hypothesis.
A separate-map architecture is another useful comparator: if it explains the same data equally well with comparable capacity, the current observations would not uniquely favor a common map.|||A capacity-matched architecture with separate maps would directly compare shared and feature-specific spatial selection.
It does not average all 32 feature channels together.|||Each feature channel is pooled independently.
A correspondence calculation using only their grayscale images cannot recover which was redder.|||The two patches produce identical grayscale correspondence inputs.
A sharp rectangular crop introduces an edge that belongs to our measurement window rather than the moving object.|||A sharp rectangular crop introduces its own boundary edge into the measured image.
At shift zero, the product sum is zero because the nonzero entries do not overlap.|||At shift zero, the nonzero entries occupy different positions, giving a product sum of zero.
This convolution acts on neighboring <em>displacement hypotheses</em>; it does not move information between neighboring image patches.|||This convolution combines neighboring <em>displacement hypotheses</em> within each image site's matching grid.
The actual encoder does not compute this entropy as its reliability feature.|||Entropy is a teaching statistic in this plot; the encoder uses the energy-based reliability r.
The implementation does not mask these startup estimates or replace the divisor with the elapsed interval.|||These startup estimates enter the computation with the nominal lag divisor.
This cannot reconstruct newly revealed surfaces, objects that appeared from outside the view, or genuinely novel images.|||Newly revealed surfaces and novel images contribute residuals wherever they differ from the transported image.
It does <em>not</em> bypass attention: the same spatial weights select it before it enters its decaying memory.|||The same spatial attention weights select it before it enters its decaying memory.
For photographs, high crop energy does not make cross-image correspondence meaningful.|||For photographs, crop energy and cross-image correspondence describe different aspects of the input.
AUC is the area under the receiver-operating-characteristic curve: it measures ranking of positive versus negative evidence across thresholds, rather than performance at the actual argmax choice.|||AUC is the area under the receiver-operating-characteristic curve: it measures ranking of positive versus negative evidence across decision thresholds.
Empty-list recognition cells have only negative labels and do not have a meaningful binary AUC or balanced accuracy.|||Empty-list recognition cells contain only negative labels and are summarized by specificity and false-positive rate.
A percentage point is a difference between percentages: 68.0% minus 48.2% is about 19.8 points, not a 19.8% relative increase.|||A percentage-point change subtracts the two percentages: 68.0% minus 48.2% is about 19.8 points.
A larger array also changes item sampling after adaptation to 100 × 100, so these curves do not isolate network capacity.|||Increasing array size changes both item count and item sampling after adaptation to 100 × 100.
None of its eight primary one-sided balanced-accuracy benefit tests passed Holm correction.|||All eight primary one-sided balanced-accuracy benefit tests fell below the criterion for significance after Holm correction.
Averaging over its one query leaves the weight unchanged; dividing that row by 100 again would not be the corresponding query average.|||With one query, the column average equals that query's weight for each patch.
It is a separate sample from the 384-scene primary test, which explains why its VDA4 estimate is not identical.|||Its separate scene sample produces its own VDA4 estimate alongside the 384-scene primary test.
A larger array gives released priority more places to go; the unexpected changing item need not receive all of it.|||In a larger array, released priority can spread among more competing items.
They do not cleanly separate encoding from storage or readout, because the intervention changes both reading and writing and its recurrent consequences can persist after the manipulation ends.|||The intervention affects reading and writing, with recurrent consequences that can persist after the manipulated phase.
It does not identify an optimal biological stimulation dose, because logit units have no conversion to current or drug concentration.|||Dose is expressed in the model's priority-logit units.
Their plots are useful for developing hypotheses, but the primary eight-test correction does not confer confirmatory status on all those comparisons.|||These exploratory plots develop hypotheses through pointwise comparisons; the confirmatory analysis is the prespecified eight-test family.
It is not a direction-change judgment and it is not semantic recognition of object categories.|||It tests exact image membership in the presented list.
A repeated probe contributes another recurrent update even if its pixels do not change.|||Each repetition of the probe contributes another recurrent update using the same pixels.
Blank intervals can contain native task glyphs; the atlas shows them rather than silently removing them.|||The atlas displays the native glyphs present during blank intervals.
'''
for row in rows.strip().splitlines():
    old,new=row.split('|||');count=0
    for p,s in texts.items():
        n=s.count(old)
        if n:texts[p]=s.replace(old,new);count+=n
    if not count:print('UNMATCHED',old)
    else:changes.append(dict(old=old,new=new,count=count))
for p,s in texts.items():
    if s!=p.read_text(encoding='utf-8'):p.write_text(s,encoding='utf-8')
(ROOT/'research/direct_prose_followup_edits.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False),encoding='utf-8')
print('Applied',len(changes),'follow-up rules.')
