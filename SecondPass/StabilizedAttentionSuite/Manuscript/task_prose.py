"""Direct explanatory wording for atlas descriptions embedded in the manuscript.

The exported atlas records and stimulus assets retain their original provenance.
"""
REPLACEMENTS = r'''
A website replay starts another viewing of the same episode; the jump from frame 1 back to frame 0 is not a second valid motion sample.|||Replay restarts the same two-observation episode.
Although the historical raster rationale mapped an 85-pixel aperture to a source-inspired visual angle, this stream does not assign an elapsed biological time to the two arrays.|||The historical raster rationale maps the 85-pixel aperture to a source-inspired visual angle; this stream expresses motion in pixels per observation.
Demonstration milliseconds therefore cannot be converted into a claimed experimental speed.|||The playback control sets the viewing speed of the demonstration.
The answer names the forward displacement that best explains common structure across the two observations, not the direction from the second frame back to the first.|||The answer identifies the forward displacement from the first observation to the second.
That paper is a source for the stimulus family, not evidence that a two-frame classification exercise measures the same selective-attention process.|||The two-frame assay adopts this stimulus family for a direction-classification task.
Website slowing and looping do not reproduce a human two-interval presentation schedule and introduce no native retention condition.|||The viewer's playback controls slow or repeat the original two-observation sequence.
Crossing the axial boundary is not a large rotation or a reversal of class, and phase displacement is not an angle sign.|||Orientation follows axial periodicity: angles separated by 180° describe the same stripe axis. The signed angular change determines the answer.
The stored signed_orientation_degrees is the authoritative sampled change; the probe angle need not be explicitly reduced modulo 180 inside the renderer for the raster to preserve axial periodicity.|||The stored signed_orientation_degrees specifies the sampled change, and the grating renderer preserves axial periodicity.
There is no blank, cue, adaptation field or separate report image, and the preview clock is not the original psychophysical timing.|||The model receives the two grating observations consecutively. The preview clock controls their display speed.
A side-by-side filmstrip is an analysis aid; it is not an extra simultaneous observation supplied by the native generator.|||The side-by-side filmstrip helps compare the two sequential observations.
The response is a comparison of amplitude, not the sign of mean luminance or the amount of apparent phase motion.|||The response compares the two grating amplitudes.
The added independent pixel noise means the two full rasters are not exact scalar multiples even though their noiseless grating components are.|||The grating components differ by an amplitude multiplier, and each rendered image additionally contains independent pixel noise.
Their experiments measured thresholds over masker contrasts rather than assuming that detection and discrimination are the same operation.|||Their experiments measured thresholds across masker contrasts, separating detection from discrimination.
A replay repeats the same decision problem, not a longer sequence with more independent evidence.|||Replay presents the same episode again.
In the ideal noiseless construction both amplitudes are equal, although independently peak-normalized finite Gabor rasters need not have exactly equal RMS.|||The noiseless construction assigns equal amplitudes; finite rasterization and independent peak normalization can produce small RMS differences.
The historical starting point is Campbell, Nachmias and Jukes’s Spatial-Frequency Discrimination in Human Vision, whose bibliographic record was verified, but its publisher page was blocked and its methods or ratio-specific conclusions were not verified in this pass.|||The historical starting point is Campbell, Nachmias and Jukes’s Spatial-Frequency Discrimination in Human Vision.
No detailed result is attributed to that inaccessible body.|||
The question is not which disc has the greater arithmetic mean of its RGB components, nor which appears subjectively brighter on a particular screen.|||The response follows the specified chromatic increment in the generator's color coordinates.
The frame_luminances metadata are computed from those ideal colors, before edge blending and noise, rather than from every rendered pixel.|||The frame_luminances metadata describe the ideal colors before edge blending and noise.
A tiny float-rounding difference in that field is not a separate answer cue.|||The metadata retain ordinary floating-point rounding.
This motivates stating the color coordinate system and considering luminance rather than treating an arbitrary RGB distance as a perceptual distance.|||This motivates specifying the color coordinate system and its luminance constraint.
There is no spatial cue, fixation phase, retention blank or separate probe; a side-by-side comparison is an explanatory view rather than an added native input.|||The model receives the two arrays consecutively; the side-by-side view supports visual comparison of those observations.
The rule concerns the known generative construction, not a guarantee of a unique consciously visible contour.|||The label identifies the array generated with the prescribed contour structure.
Both versions are generated from one crop, not successive views of a changing scene.|||Both versions are generated from the same photograph crop.
Looping is an atlas convenience and does not turn this comparison into an image-recognition trial.|||The atlas can replay the same two-observation spectral comparison.
The answer is therefore not simply whichever frame has a greater overall standard deviation.|||The answer follows the spectral manipulation under the shared RMS scaling.
This supports asking about scale-dependent image statistics rather than scene category judgments.|||This motivates testing scale-dependent image statistics.
The two sample frames repeat the orientation evidence, not the exact pixel raster.|||The two sample frames share the underlying orientations and contain fresh noise.
The viewer’s loop restart is another presentation of the illustration, not a fifth frame of the native trial.|||The native trial has four frames, and looping restarts that sequence.
That directly supports the relevance of spatial selection among competing local signals, but their stimuli were moving dots, not four Gabor arrays.|||Their moving-dot experiments motivate the role of spatial selection among competing local signals; this assay uses Gabor arrays.
The cue is not merely a one-frame flash: its three visible presentations include both samples.|||The cue remains visible for three observations, including both samples.
The delay is counted in native frames, not milliseconds, and its “blank” field is not entirely blank because the corner vocabulary remains.|||The delay is counted in native frames. The corner vocabulary remains visible during this interval.
At D0 the last sample is immediately followed by the probe; choosing another delay cell selects an independently generated episode, not an automatically matched extension of the same evidence.|||At D0 the probe immediately follows the last sample. Each delay cell contains independently generated episodes.
An unchanged target is therefore a genuine negative, not a third response class.|||An unchanged target belongs to the negative response class.
The metadata’s rotation_magnitude records the nonzero step size used in the trial, so it must not be described as the target’s actual rotation when that target stayed unchanged.|||The metadata's rotation_magnitude records the trial's nonzero step size; the target's actual rotation is determined separately by whether it changes.
Their verified task description motivates the distinction between physical evidence and context-dependent relevance, not the specific use of a plus/minus glyph or orientation-change judgment.|||Their task motivates the distinction between physical evidence and context-dependent relevance. This assay expresses context through a signed cue and an orientation judgment.
Frame 1 is the initial dot-position reference, which does not yet contain a motion transition.|||Frame 1 supplies the initial dot positions for subsequent motion comparisons.
The report frame does not show dots or a target ring.|||The final report uses the retained motion evidence after the dot array disappears.
Even D0 still asks about earlier motion transitions rather than only the final raster.|||At D0, the response still depends on the preceding motion transitions.
“Longest duration” means greatest total count, not the longest uninterrupted run.|||“Longest duration” means the greatest total count of transitions in a direction.
The advancing count bars are analysis displays computed from this metadata, not a decoded network state.|||The advancing count bars display the direction counts recorded in the trial metadata.
The motivating question is how a decision can depend on evidence distributed over time rather than on one terminal snapshot.|||The task asks how a decision integrates evidence distributed over time.
These are B12/B20/B28 conditions, not D12/D20/D28 retention blanks.|||B12/B20/B28 specify the baseline motion duration.
Browser playback is a labeled slowdown, not calibrated 100-Hz stimulation.|||Browser playback displays a labeled slowdown of the source's 100-Hz sequence.
A catch draws nuisance sign/magnitude values internally but its signed_change_degrees is zero and changed_patch is null; those unused draws must not be displayed as a physical catch event.|||For catch trials, signed_change_degrees is zero and changed_patch is null.
The event line, target/foil tags and change sign are explanation-only: showing them in an unassisted observer view would reveal information not explicitly announced in the raster.|||The event line, target/foil tags, and change sign are explanatory overlays. The unassisted observer view presents the original raster sequence.
These facts were verified in the primary full text, not inferred from another Krauzlis-laboratory experiment.|||The primary full text documents these task details.
The queried location is selected internally before rendering, but it is not visually disclosed until after the sample and delay.|||The queried location is selected before rendering and disclosed to the observer after the sample and delay.
There are no inserted postquery delay frames, and the query does not itself show the remembered orientation.|||The spatial query immediately follows the retention interval and identifies the location to report from memory.
“D0” removes the retention blanks but does not remove the query frame or turn it into a precue.|||At D0 the retrospective query immediately follows the sample.
Luck and Vogel’s primary feature/conjunction memory study motivates the broader importance of object-level combinations, but it is not a retrospective exchange experiment.|||Luck and Vogel's feature/conjunction memory study motivates the importance of retaining object-level combinations.
Only its abstract was verified here, and no quantitative capacity result or detailed methods claim is transferred to this assay.|||
H is a condition control, not an independently resampled holding time within that cell.|||H fixes the probe-hold duration for the selected condition.
A longer hold repeats one query image, not new recognition questions.|||A longer hold presents additional observations of the same query image.
Frame counts here do not have an assigned biological time scale; the viewer must preserve the full N24 list rather than abbreviate it to improve the animation.|||Timing is expressed in observation counts, and the viewer presents the complete list, including every image in N24 trials.
'''

def direct_task_prose(text):
    for row in REPLACEMENTS.strip().splitlines():
        old,new=row.split('|||')
        text=text.replace(old,new)
    return text
