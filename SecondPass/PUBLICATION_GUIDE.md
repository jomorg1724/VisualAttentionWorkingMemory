# Explainers, papers and recordings

Start with the [guided HTML explainer](StabilizedAttentionSuite/Manuscript/index.html). Its task selector introduces the trials, performance, attention over time, and neuroscience results in stages. The [full reference](StabilizedAttentionSuite/Manuscript/reference.html) contains the longer mathematical and implementation explanations.

## Reading

- [18-page neuroscience results synthesis](../output/pdf/neuroscience_results/Shared_Priority_Neuroscience_Results.pdf), with [LaTeX source](../output/pdf/neuroscience_results/main.tex).
- [Individual task papers and analysis gallery](TaskNeuroscience/Results/index.html).
- [Complete task-paper volume](TaskNeuroscience/Results/papers/complete_volume.pdf).
- [Complete optic-flow walkthrough](../output/pdf/optic_flow_complete_walkthrough.pdf), with [LaTeX source](../output/pdf/optic_flow_complete_walkthrough.tex).
- [Two padding stages](../output/pdf/optic_flow_two_padding_stages.pdf).

## Watching and exploring

- [Narrated architecture animation](ArchitectureExplainer/index.html), [MP4](ArchitectureExplainer/architecture-explainer.mp4), and [subtitles](ArchitectureExplainer/captions.srt).
- [Attention maps across conditions and time](TaskNeuroscience/Results/attention/index.html).
- [Original task atlas and human-playable trials](StabilizedAttentionSuite/Manuscript/task-atlas/index.html).
- [Earlier suite attention and intervention gallery](StabilizedAttentionSuite/Analysis/run01/gallery/index.html).
- [VDA attention and intervention gallery](StabilizedAttentionVDA/Analysis/run01/gallery/index.html).
- Doom playback: [continuation 1](DeadlyCorridor/CloudRuntime/continuation01/playback/index.html) and [continuation 2](DeadlyCorridor/CloudRuntime/continuation02/playback/index.html).

GitHub displays HTML source. To use the interactive pages, clone the repository, run `python -m http.server 8877` from its root, and open `http://127.0.0.1:8877/`.

## Included material

The publication includes finished PDFs, LaTeX sources, plots, GIFs, videos, playback pages, selected trial examples, aggregate result tables, and the display assets used by the explainers. Trial ZIP downloads contain the atlas examples. The paper ZIP collects the papers, figures, summaries and analysis source.

Model checkpoints, raw training datasets and large per-trial analysis arrays remain local. The earlier suite gallery's two JSONL downloads refer to local raw analysis exports; those downloads are available only in a research checkout containing the original exports. Its plots and interactive attention viewer use the included display assets.

The architecture animation illustrates the spatial-memory-query architecture. The neuroscience papers identify their analyzed checkpoint and experimental conditions; the animation's schematic activations are separate from those measured results.
