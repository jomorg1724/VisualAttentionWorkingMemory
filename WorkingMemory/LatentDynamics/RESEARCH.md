# Reading the working-memory state

Research and analysis rationale, 2026-09-13. This is an analysis of frozen trained networks, not a new memory architecture or training objective. The implemented scope and measured results belong in this directory's protocol and report.

## Questions that the pictures should answer

We want to see whether visual features, stimulus locations, momentary motion, accumulated directional evidence, and eventual choices have identifiable population representations. We also want to see how those representations evolve when the screen becomes blank and when the query or comparison probe arrives. The sensory field H, memory firing rates R, and adaptation A are different stages of this computation and should be examined separately.

For the spatial attention model, H, R, and A each have shape [B,64,13,13]. Flattening a field for analysis preserves its indexed spatial information; global averaging does not. A small spatial pooling grid offers a manageable first visualization, but discards within-bin detail. Spatial selectivity maps provide a complementary view. Access to A is an analysis privilege: the ordinary classifier does not directly read it.

There are two distinct meanings of segmentation: grouping recurring population states, and grouping channels/sites by their tuning. Both are useful exploratory summaries. Neither establishes anatomically distinct brain regions or biological cell types.

## Why changing activity need not mean forgetting

Murray and colleagues found a stable working-memory population subspace in primate prefrontal cortex despite strong dynamics in individual neurons. This motivates examining stimulus information separately from the overall movement of the state. A network may travel a long distance in state space while preserving a usable orientation code. [Murray et al., 2017, PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC5240715/).

For an actual rendered axial orientation theta, use the target

\[
y=(\cos 2\theta,\sin 2\theta).
\]

A linear decoder trained at time t is evaluated at time s on independent held-out episodes. Its reconstructed angle is half the atan2 of the predicted vector. The axial error is

\[
e=\frac12\left|\operatorname{atan2}\bigl(\sin 2(\hat\theta-\theta),\cos 2(\hat\theta-\theta)\bigr)\right|.
\]

Plot the train-time by test-time error matrix. Strong transfer through blanks supports a stable accessible code. Good separately fitted late decoding with poor early-to-late transfer is consistent with a changed readout relationship, but can also reflect nuisance shifts, scaling, or limited fitting data. Failure of all tested decoders does not establish information erasure. Norms, trajectories, and decoding should be read together.

## Motion: what happened versus what the model decides

Mante and colleagues analyzed primate prefrontal activity and trained recurrent networks in a context-dependent sensory integration task. Population-level trajectories helped explain selection and integration despite complex single-neuron responses. Our motion task differs, but their approach motivates separating evidence, context, and choice axes. [Mante et al., 2013, Nature](https://www.nature.com/articles/nature12742).

For the existing equal-duration motion transitions, a descriptive evidence target is

\[
c_t(d)=\sum_{k\leq t}\mathbf 1[d_k=d],\qquad d\in\{\mathrm{right},\mathrm{up},\mathrm{left},\mathrm{down}\}.
\]

These counts are analysis labels, never model inputs. Signed contrasts such as c(right)-c(left) and c(up)-c(down), the full count vector, current direction, leading direction, and the model's reported class answer different questions. A leading direction is undefined at ties unless the analysis explicitly handles them. The final winner alone is insufficient to demonstrate accumulation.

Evidence increases with elapsed time and correlates with recent direction. Therefore, analyze at matched transition indices and compare with recency/time-only predictors where samples permit. During a blank, elapsed blank time increases but the true motion evidence remains fixed. Distinguish cumulative counts from normalized fractions; each has different trivial time dependencies. A latent evidence correlation without these distinctions can be misleading.

## Geometry, rotations, and biological interpretation

Libby and Buschman found sensory and recent-memory representations occupying different population subspaces, with stable and switching selectivity contributing to their transformation. Their experiment involved auditory sequence processing in mice, not our visual-memory task. It motivates a geometric hypothesis, not a claim that our model implements the same circuit. [Libby and Buschman, 2021, Nature Neuroscience](https://www.nature.com/articles/s41593-021-00821-9).

A more recent primate study found some prefrontal subspace rotations even before working-memory task training, while other geometry changes accompanied training and distinguished correct from error trials. Thus a rotation by itself is not evidence of deliberate maintenance or successful task computation. Our useful question is whether geometry relates to feature accessibility and errors. [Pu et al., 2024, Nature Communications](https://www.nature.com/articles/s41467-024-50717-y).

## Visualization and identification are complementary

PCA provides a common linear reference with a stated fraction of variance explained. Fit its scaler and basis using the analysis training split, then project held-out episodes with that same basis. Do not fit separate plots at every frame and interpret their arbitrary relative rotations as network dynamics.

UMAP and t-SNE help expose local organization. Show the same coordinates colored by orientation, direction, phase, elapsed time, and correctness rather than selecting a projection because its labels look attractive. A two-dimensional island is not automatically a discrete memory state. Distance, density, and apparent cluster size depend on the embedding procedure. t-SNE without an out-of-sample transform should be labeled a descriptive embedding of its supplied points, not a held-out decoder. [Kobak and Berens, 2019, Nature Communications](https://doi.org/10.1038/s41467-019-13056-x).

Task-dependent linear axes are useful alongside unsupervised plots. Demixed PCA explicitly separates dependencies on task variables in population data and addresses mixed selectivity. Generic label regression is not dPCA and should not be named as such. Our first pass can use simpler held-out linear decoding and tuning profiles; full demixing is an option if the available factorial sampling supports it. [Kobak et al., 2016, eLife](https://elifesciences.org/articles/10989).

CEBRA is a modern contrastive embedding method that can use behavior labels or temporal relationships to shape latent geometry. It is relevant for subsequent analysis, but an orientation-separated embedding explicitly trained with orientation labels would not independently demonstrate that an unsupervised plot discovered orientation structure. We prioritize unlabelled embeddings and separately evaluated decoders for the first atlas. [Schneider, Lee and Mathis, 2023, Nature](https://www.nature.com/articles/s41586-023-06031-6).

## What the existing tasks can and cannot identify

- **Orientation:** use the actual raster angle, including random context rotation, not the renderer's abstract stimulus level. Decode each bound item separately where appropriate.
- **Changes:** distinguish sample angle, probe angle, their relation, the ground-truth answer and the model's choice. The change label ordinarily becomes answerable only after the probe. Apparent pre-probe prediction needs a stimulus-correlation check.
- **Cues:** the current subset has cue images tied to task/phase. Their visible footprint or phase can be decoded, but this does not independently identify a general rule representation. Document this limitation instead of introducing new teaching.
- **Location binding:** use item/location-aware summaries so a successful global orientation readout is not confused with preserving which item was where.
- **Errors:** compare correct and incorrect trials without treating outcome-dependent groupings as causal interventions. Feature difficulty and stimulus imbalance may also differ between those groups.

All train/validation/test assignments are by base episode. Frames and paired delays from one episode remain in the same split. Analysis-only readouts never update the deployed network. The first atlas describes one trained checkpoint and its existing task distribution; it does not establish a universal neural code or a biological correspondence.
