# Primary sources and explicit task adaptations

Checked2026-09-14UTC. This battery is a set of computational task adaptations. Only the additional cued dot-change task follows the named Krauzlis recipe directly; the other four are declared extensions of the user's task requests and existing project generators.

## Selected dot-change source

**Arcizet, F. and Krauzlis, R. J. (2018), “Covert spatial selection in primate basal ganglia.”** [PLOS Biology primary article](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005930), DOI10.1371/journal.pbio.2005930. Methods subsection “Motion-direction CD task” and Figure1 define the selected paradigm.

The source presents a0.2s peripheral ring cue, then0.5s blank, then two opposite-location patches whose means differ90°. Dots have16°direction SD,10-frame/100ms lifetime,15°/s speed,3–3.75°aperture radius and10–13°eccentricity. Printed density is25dots/deg²/s. Events occur1.0–4.3s after onset:57%target,29%foil,14%catch. Median changes are26°/28°, with both rotation signs. Releasing within1s of a target change counts as a hit; foil/catch require withholding.

Our [protocol](PROTOCOL.md) retains target/foil semantics but compresses durations, fixes geometry/counts, replaces a continuous joystick response with final classification, and uses a100pixel raster. Its100Hz conversion follows10frames/100ms. The source does not license calling every frame100ms. Dot-density conversion, subpixel rendering and boundary rules are explicitly implementation choices.

The existing [Krauzlis research note](../../PreAttentiveVision/krauzlis_stimulus.md) also discusses Zénon2012 and Lovejoy2010. They are **not mixed into this recipe**: the former's8-frame lifetime/opposite means and the latter's coherence pulse define different details/objectives.

## Orientation and binding

The old [sequence generator](../stimuli.py) implements plus/minus relevance glyphs and cumulative signed evidence. Here the user confirmed **single sample-to-probe signed rotation** while reusing the glyph/sign meaning. The old cumulative threshold bar is not copied into the new task. Localizing that cue across four regions is a new spatial demand.

[Luck and Vogel (1997), “The capacity of visual working memory for features and conjunctions,” Nature](https://www.nature.com/articles/36846), DOI10.1038/36846, is primary background for controlled visual feature/conjunction memory comparisons. Our four-region retrospective target-versus-foil swap is not a reproduction of its stimulus set or a basis for borrowing its capacity estimate. Exact all-trial swap counterbalancing is specified in this project's protocol to prevent a global-change shortcut.

The prior project [spatial comparison](../SpatialComparison/report.md) showed why task construction matters: a whole-array two-item swap could be detected by remembering one location. The new target query occurs after the sample delay, and both labels contain one exchange; this is a designed task change, not a retrospectively relabeled old result.

## Image source and recognition boundary

[Berkeley's official BSDS500 resource](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) provides the natural-photo dataset originally used for segmentation and boundary research. Its requested reference is Arbelaez, Maire, Fowlkes and Malik (2011), “Contour Detection and Hierarchical Image Segmentation.” We use photographs only, not its segmentation annotations or published benchmark scores.

The [existing local provenance](../../PreAttentiveVision/data/bsds500/README.md) and [manifest](../../PreAttentiveVision/data/bsds500/manifest.json) record200train/100validation/200test source identities and archive/file hashes. Recognition preserves these disjoint source pools. Repeated episodes and crops do not become independent photographs. No ownership or new source-image license is asserted.

The list lengths0/4/12/24, consecutive one-frame study images, three blank frames after the full list and repeated exact probe are the user's requested computational recognition task, not an attributed reproduction of a named human timing experiment. L0has no positive examples: report specificity separately rather than inventing a binary AUC.

## Evidence boundary

Primary neuroscience motivates sensory/task structure; model learning, Adam/BPTT, exact raster sizes and compressed timing are engineering decisions. Successful task performance will be measured from the new run's saved predictions. These source notes are neither results nor evidence that the implementation already learned spatial attention or item retention.
