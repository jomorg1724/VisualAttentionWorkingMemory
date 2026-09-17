# Second pass

Everything needed to pick the project up again after the 2026-09-16 reset and the 2026-09-17 results, in one folder.

| Item | Where | What it is |
|---|---|---|
| Papers we pulled and cited | [papers/BIBLIOGRAPHY.md](papers/BIBLIOGRAPHY.md) | 65 references extracted from every research note in the repository, grouped by theme, each with where it was cited and what claim it supported; unverified locators are flagged. `papers/fetch_papers.py` downloads the open-access ones (arXiv, PMC) into `papers/pdf/`, which is git-ignored. |
| How we work on pods | [PODS.md](PODS.md) | Rules, accounts, the Windows gotchas that cost hours, the scripts, the procedure, the dated failures and the cost log. |
| The KDA paper | [KDA_paper/KDA_paper.md](KDA_paper/KDA_paper.md) (also `.tex`, `.docx`, `.html`) | Tutorial-style NeurIPS-format paper: background (linear attention, delta rule, gated decay), the spatial KDA module as implemented with shapes, why it works on cued retention (matrix form, nonexpansive transition, implicit attention), measured results, and the attention and psychometric analysis programmes. |
| Analysis tools | `../WorkingMemory/PlainBaseline/analysis/` | `kda_probe.py` (gate maps, exact implicit attention weights, state probes, interventions), `psychometric.py` (magnitude, delay, cue and distraction sweeps with cumulative-Gaussian and exponential fits), `psych_stream.py` (the parametrised generator they use). |
| Where the results live | `../LabJournal/experiments/25-*.md`, `26-*.md`, `27-*.md` | Audit, plain baseline, accumulator comparison. |

## State of the project in one paragraph

The five-task battery is sound (experiment 25). The recipe the old lineage used collapses any model to a constant output within 150 updates; with a centred input, three stacked frames and Adam at 1e-4 a plain CNN+GRU learns the two-way rungs of the orientation family from scratch and the real task by curriculum, and learns retention across blanks only by a delay curriculum, to 0.85-0.88 (experiment 26 and 27). A gated spatial state inside the conv stack, ConvGRU or the KDA accumulator, reaches 0.994-1.000 at every delay on two seeds under the same rules (experiment 27). The KDA model is the working baseline for the second pass. A local re-run (`WorkingMemory/PlainBaseline/runs/local_kda_program_20260917/`) reproduced the pod result with checkpoints kept, and both analyses have been run on it (paper Sections 6.7 and 7.4): the memory is necessary, precise, persistent to 48 blanks and site-local; its only fragility is cue contrast and position, which points the next question at attention rather than memory.

## Next steps, in order

1. Done: `kda_probe.py` and `psychometric.py` on the local re-run's final model; remaining probes are site swap and readout attribution (paper 6.4, 6.5).
2. The same gate, curriculum and ladder program on motion, binding, recognition and Krauzlis, ladder rungs built per family, plain control beside it.
3. The gating-versus-key-addressing ablation (Section 8 of the paper).
4. Second seed of everything that is a claim.
5. Only then any of the old lineage's attention, E/I or priority-readout components, one at a time against this baseline.
