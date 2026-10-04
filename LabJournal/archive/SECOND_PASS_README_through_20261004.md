> Historical snapshot through October 4, 2026. Earlier present-tense launch statements are not current status. Relative links have been adjusted for this archive location.

# Second pass

**Current: fresh whole-model final spatial ConvGRU is training on RunPod.**
[Fresh-only harness](../../SecondPass/SpatialReadout/FreshRun/README.md),
[verified live record](../CURRENT_STATUS.md). No trained parameters
transferred. The original global-GRU baseline remains preserved; the cancelled
transformer and historical warm starts have not been resumed.

**Current: user-cancelled cloud run; original CNN + spatial KDA + global GRU
restored as the active architecture.** [Exact implementation, provenance and
fresh-only rule](../../SecondPass/ACTIVE_BASELINE.md). No new training or checkpoint transfer.
Historical launch authorizations below are superseded.

**New authorization / running, 2026-09-23:** the user explicitly requested “go ahead and set up a 8 hour training run for more trsining and more updates”. The versioned v3 continuation resumes **terminal 715**, not historical selected 117. It pins **2,145 additional updates / 68,640 additional episodes** (5,280/task), reaching **2,860 total updates / 91,520 episodes** (7,040/task). One local MPS worker; no architecture, loss, stimuli, sampling, optimizer or batch changes.

Verified live update733 and checkpoint728; hard deadline **2026-09-23 09:08:50.062308 UTC**. Parent owns guardian `proc_5cbdb07600dd` (sole MPS worker51433). Details and automatic final-report location: [joint-training record](../joint-suite-training.md). Earlier completion/no-further-authorization statements below are historical and superseded by this explicit new allowance.


Everything needed to pick the project up again after the 2026-09-16 reset and the 2026-09-17 results, in one folder.

| Item | Where | What it is |
|---|---|---|
| Papers we pulled and cited | [papers/BIBLIOGRAPHY.md](../../SecondPass/papers/BIBLIOGRAPHY.md) | 65 references extracted from every research note in the repository, grouped by theme, each with where it was cited and what claim it supported; unverified locators are flagged. `papers/fetch_papers.py` downloads the open-access ones (arXiv, PMC) into `papers/pdf/`, which is git-ignored. |
| How we work on pods | [PODS.md](../../SecondPass/PODS.md) | Rules, accounts, the Windows gotchas that cost hours, the scripts, the procedure, the dated failures and the cost log. |
| The KDA paper | [KDA_paper/KDA_paper.md](../../SecondPass/KDA_paper/KDA_paper.md) (also `.tex`, `.docx`, `.html`) | Tutorial-style NeurIPS-format paper: background (linear attention, delta rule, gated decay), the spatial KDA module as implemented with shapes, why it works on cued retention (matrix form, nonexpansive transition, implicit attention), measured results, and the attention and psychometric analysis programmes. |
| Analysis tools | `../WorkingMemory/PlainBaseline/analysis/` | `kda_probe.py` (gate maps, exact implicit attention weights, state probes, interventions), `psychometric.py` (magnitude, delay, cue and distraction sweeps with cumulative-Gaussian and exponential fits), `psych_stream.py` (the parametrised generator they use). |
| Where the results live | `../LabJournal/experiments/25-*.md`, `26-*.md`, `27-*.md` | Audit, plain baseline, accumulator comparison. |

## State of the project in one paragraph

The five-task battery is sound (experiment 25). The recipe the old lineage used collapses any model to a constant output within 150 updates; with a centred input, three stacked frames and Adam at 1e-4 a plain CNN+GRU learns the two-way rungs of the orientation family from scratch and the real task by curriculum, and learns retention across blanks only by a delay curriculum, to 0.85-0.88 (experiment 26 and 27). A gated spatial state inside the conv stack, ConvGRU or the KDA accumulator, reaches 0.994-1.000 at every delay on two seeds under the same rules (experiment 27). The KDA model is the working baseline for the second pass. A local re-run (`WorkingMemory/PlainBaseline/runs/local_kda_program_20260917/`) reproduced the pod result with checkpoints kept, and both analyses have been run on it (paper Sections 6.7 and 7.4): the memory is necessary, precise, persistent to 48 blanks and site-local; its only fragility is cue contrast and position, which points the next question at attention rather than memory.

## Next steps, in order

**Completed local joint run (2026-09-22):** 715 updates / 22,880 episodes,
with selected and terminal evaluations complete on all 35 cells, inside the
original four-hour cap. [Results](../../SecondPass/JointTraining/RESULTS.md) report terminal
715's strong contrast/chromatic/spectral learning and near-chance motion and
spatial/sequence performance. The official validation-selected checkpoint is
117; both artifacts are preserved. No further run is active or automatically
authorized. Launch descriptions below are historical.

**Latest authorization (2026-09-22):** the user now requests a fresh local KDA
joint-training run across the complete suite. See the
[run contract](../joint-suite-training.md). The user subsequently
kept the ORIGINAL four-hour deadline and authorized correcting the
evaluation-heavy allocation. The [v2 amendment](../../SecondPass/JointTraining/AMENDMENT_V2.md)
pins 715 total updates / 22,880 episodes (1,760/task), preserving the full
step-117 model/Adam/streams/RNG state. Resumed progress reached 122 updates /
3,904 episodes at 07:49:04 UTC; checkpoint 118 was hash-verified and loaded.
Parent owns `proc_e7bdb8db0d3f`; deadline stays 10:47:56.585638 UTC.
Final selected/terminal scores across all 35 cells are pending. This is
more substantive acquisition, not established convergence or an overnight run.
No cloud, second arm, training reset or automatic cap extension is authorized.

**Current preparation (2026-09-22):** [Unified task suite](../../SecondPass/TaskSuite/README.md)
assembles all 13 discussed tasks in 35 primary cells for a **later fresh-weight
joint-training run**. CPU rendering/replay checks and fresh-KDA forward checks
pass; no training was launched. The older per-family/curriculum roadmap below
is historical, not an implicit schedule or current training authorization.

1. Done: `kda_probe.py` and `psychometric.py` on the local re-run's final model; remaining probes are site swap and readout attribution (paper 6.4, 6.5).
2. The same gate, curriculum and ladder program on motion, binding, recognition and Krauzlis, ladder rungs built per family, plain control beside it.
3. The gating-versus-key-addressing ablation (Section 8 of the paper).
4. Second seed of everything that is a claim.
5. Only then any of the old lineage's attention, E/I or priority-readout components, one at a time against this baseline.
