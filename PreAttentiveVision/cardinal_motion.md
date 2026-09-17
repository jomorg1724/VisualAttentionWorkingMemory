# Two-frame cardinal motion: first corrected PAV task

The user explicitly selected **exactly two 100×100×3 images**, with a population of dots moving right, up, left or down, and the network classifying that direction. This resolves the pending temporal choice in [krauzlis_stimulus.md](krauzlis_stimulus.md). The target is motion-direction discrimination, not a change in motion direction. No attention cue, foil, delay or future component is implemented here.

## Source and adaptation

[Lovejoy and Krauzlis (2010), Online Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC3412590/) describes circular stochastic-motion patches of radius 4.25° with two-refresh dot lifetimes. A dot either appears at a random position or displaces four screen pixels, approximately 0.2°. Motion coherence controls the fraction moving in a common direction; other dots move in random directions. The original task contains 160-ms pulses, four diagonal directions, cues and distractors. This PAV task retains limited-lifetime population correspondence but replaces the temporal trial with a single observed transition and uses the user's four **cardinal** directions. It is an adaptation, not a replication.

## Explicit raster choices

| Quantity | Version-2 implementation | Status |
|---|---|---|
| Input | Two separate 100×100 RGB images | User requirement |
| Labels | 0 right, 1 up, 2 left, 3 down | User requirement; coordinate convention explicit |
| Aperture | Diameter 85 pixels, centered at (49.5,49.5) | 4.25° source radius mapped at 10 pixels/degree |
| Displacement | Uniformly 1, 2 or 3 pixels | Raster robustness variation around source-derived 2-pixel anchor |
| Dot population | 256 on the full 100×100 periodic domain; approximately145 visible on average | Chosen raster density, not a reported source dot count |
| Lifetime | Two frames, with exactly128 identities surviving and128 reborn | Balanced initial ages; finite-lifetime adaptation |
| Direction spread | None for surviving dots | User's cardinal-direction requirement |
| Coherence | 1.0 among surviving dots | Fixed initial task, no coherence curriculum |
| Dot rendering | Bilinear subpixel deposition, Gaussian sigma0.7 pixels | Anti-aliased raster adaptation |
| Intensity | Background0.2, positive dot increment0.7, clipped at1 | Numerical intensity choice; not calibrated luminance |
| Channels | Same grayscale image in R,G,B | Three-channel interface; no color direction cue |

Two source refreshes do not require assigning a physical speed to these arrays. The anchor displacement is `0.2 degrees × 10 pixels/degree = 2 pixels`. A claim in degrees/second would additionally require a specified time between images. The1/3-pixel values are explicit modifications, not source measurements. There is no added angular jitter, arbitrary color coding or changed-only sensory augmentation.

## Preventing single-frame boundary cues

Initial positions are uniform on a periodic square. To obtain the second frame, translate all surviving positions in the chosen direction modulo100 and place the reborn identities independently and uniformly. Apply the **same fixed circular aperture only after rendering**. Translation preserves the uniform position distribution, and the independent rebirth law also preserves it. Therefore either frame alone has the same distribution for every class. A fixed aperture boundary cannot reveal direction through systematic dot loss or an accumulating density edge.

Both image frames contain approximately the same expected number of visible dots; instantaneous counts fluctuate naturally. A tracked dot can cross the visible aperture boundary, so surviving identity count is reported for the full domain, not falsely claimed to be128 visible correspondences. The term coherence here refers to surviving dots: half of all domain identities are refreshed, which supplies correspondence noise even though every survivor follows the selected cardinal direction.

## API and evidence

`neuroscience_stimuli.py` provides `TASK_CLASSES={'motion_direction':4}` and `TaskStream(seed,split).batch(n,task='motion_direction')`. It returns CPU tensors of shape `[n,2,3,100,100]`, integer labels and audit metadata. Metadata is never model input. Multiples-of-four batches have exactly equal class counts. `state_dict`/`load_state_dict` preserve local RNG, stream position and protocol identity.

The focused check verifies labels, image shape/range, exact stream restoration and that a simple rendered cross-correlation observer recovers the intended motion. Its accuracy is a **stimulus-observability diagnostic**, not any encoder's result. Evidence is in `cardinal_motion_check.json`. `cardinal_motion_examples.png` displays both frames of each direction. The accompanying GIF slows each frame to300ms and inserts a blank before looping, avoiding a spurious reverse-motion transition; that preview timing is not scientific acquisition timing.

The previous version-1 source remains separate. This task does not recover or continue the deleted historical architecture, and it does not choose implementations for visual attention, visual working memory or integration.
