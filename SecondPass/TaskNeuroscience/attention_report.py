"""Paper structure and a portable frame-by-frame attention viewer."""
import json,re,time
from pathlib import Path
import numpy as np
from .design import OUT,TASKS,write

def read(path):return json.loads(path.read_text(encoding='utf8'))

CONTEXT={
 'motion_direction':'Rows separate the four motion directions and pool the tested dot speeds. The four line traces refer to fixed image quadrants.',
 'orientation':'Rows separate absolute orientation differences and pool rotation signs and nuisance orientations. Traces refer to fixed image quadrants.',
 'contrast':'Rows separate contrast increments and pool pedestal levels and interval order. Traces refer to fixed image quadrants.',
 'spatial_frequency':'Rows separate octave increments and pool interval order and nuisance frequencies. Traces refer to fixed image quadrants.',
 'chromatic_increment':'Rows separate chromatic increments and pool interval order. Traces refer to fixed image quadrants.',
 'contour':'Rows separate contour-alignment jitter and pool interval order. Traces refer to fixed image quadrants.',
 'natural_spectrum':'Rows separate spectral-exponent differences and pool interval order and photographs. The map average describes image-coordinate allocation across different photographs; each viewer overlay belongs to one replayed photograph.',
 'orientation_ring':'The target is fixed at top left. Rows compare clockwise and counterclockwise changes in the zero-delay native condition.',
 'orientation_cued':'The target is fixed at top left with a 24-frame delay. Rows separate target changes matching the instruction, unchanged targets, and target changes opposing the instruction. Clockwise and counterclockwise instructions are pooled.',
 'motion_duration_cued':'The cued patch is fixed at top left. Rows compare zero-, twelve- and twenty-four-frame retention delays. Motion onset defines frame zero; the later retention interval and report remain on the same time axis.',
 'krauzlis_cued_motion':'The cue is fixed at the right patch, with 28 baseline transitions. Rows separate a target change, a left-foil change, and a catch trial. The shared time origin is the first postchange frame, including its scheduled counterpart on catches.',
 'spatial_binding':'The retrocued target is fixed at top left with a 24-frame retention interval. Rows separate swaps involving that target from swaps confined to other locations.',
 'image_recognition':'Rows cross four versus twenty-four study images with seen versus novel probes, using five probe frames. Probe onset defines the common time origin. The central region is the middle 4-by-4 patch square; the second trace sums the remaining 84 sites. An average over different photographs measures a consistent spatial preference; the viewer separately preserves the exact photograph and its own attention.',
 'vda4':'All rows are no-change trials with the cue fixed at bottom right. Rows compare the four displayed cue validities, pooling the three native difficulty cells and reward colors. This most directly follows the condition-by-time organization of Figure 4 in Morgan, Albanna and Herman.',
 'vda9':'All rows are no-change trials with the cue fixed at bottom right. Rows compare zero, 25, 50, 75 and 100 percent displayed cue validity, pooling the three native difficulty cells and reward colors.',
 'vda16':'The cue is fixed at row 2, column 3 of the four-by-four array. No-change trials are pooled into lower (25/50 percent) and higher (75/100 percent) displayed-validity groups because separate fixed-site validity cells have few observations. Difficulty and reward colors are pooled; the displayed sample counts quantify the available evidence.',
 'memory_guided_search':'The correct array location is fixed at top left. Rows cross feature, irregular-object and fractal families with distinct versus similar candidates. Sample duration, blank delay and array exposure are fixed at 8, 12 and 8 frames. This holds timing and spatial geometry constant while changing the demands on remembered-content matching.',
 'motion_memory_search':'The matching candidate is fixed at top left. Rows compare 20-, 45- and 90-degree candidate separation at twelve sample frames, twelve delay frames and twelve array frames.',
 'popout_search':'The singleton is fixed at top left and array exposure is eight frames. Rows compare weak and strong color differences (12 and 72 renderer units) and weak and strong orientation differences (5 and 45 degrees).',
}

def section(task,data):
    from .papers import esc,fig
    rows=data['groups'];fs=data['figures']
    s='% BEGIN CONDITION ATTENTION\n'+r'\clearpage\subsection{Attention across conditions and time}'+'\n'
    s+='The analysis is organized around a direct visual question: where does the shared priority signal go as the trial unfolds, and how does that trajectory depend on the task condition? Rows retain fixed experimental conditions and physical locations. Columns follow time. The arrangement follows the heatmap-plus-time-course logic of Morgan, Albanna and Herman (2025), Figure 4 (\\url{https://arxiv.org/html/2502.10955v1}).\n\n'
    s+=esc(CONTEXT[task])+'\n\n'
    s+=r'\paragraph{What each plotted quantity means.}'+ '\n'
    s+=r'For trial $i$, frame $t$ and image patch $p$, $\alpha_{itp}$ is the attention probability assigned to that patch; its sum over the 100 patches is one. For a condition containing $n$ trials, the map shows $\bar\alpha_{tp}=n^{-1}\sum_i\alpha_{itp}$. The 100 entries are placed back into their original 10-by-10 image grid without spatial rotation. A line for region $R$ shows the trial-average mass $M_R(t)=n^{-1}\sum_i\sum_{p\in R}\alpha_{itp}$. A line labeled ``other regions (mean)'' averages their separate masses rather than summing all alternatives. A dotted horizontal line marks the mass the same region would receive under uniform attention, its patch count divided by 100.'+'\n\n'
    s+='Target apertures use the same patch-center geometry throughout a task: circular masks with a 16-pixel radius for search arrays and a 12-pixel radius for the other localized tasks. VDA uses its native stimulus-cell boundaries mapped onto the 100-site grid. The central sample region is a 16-pixel-radius circle excluding candidate apertures. The source tables retain the exact region weights.\n\n'
    s+='Every curve includes all selected trials, independently of whether the model answered correctly. Shaded bands are pointwise 95 percent intervals from 2,000 whole-trial bootstrap resamples. Repeated photograph identities are resampled together. These bands describe each condition separately. Phase-colored backgrounds mark encoding, retention and comparison; frame zero on the horizontal axis is the task-specific evidence onset, or the first motion transition in the duration task. Original frame numbers remain visible on the maps.\n\n'
    s+='Map colors share one scale across every condition and time point within this task. A square-root color transform makes small weights visible; the colorbar retains probability units. The overview uses named temporal landmarks, with explicit original frame numbers when trial lengths differ. The appendix supplies every recorded frame, and the accompanying HTML viewer pairs an exact trial movie with its own attention overlay and the condition-average map.\n\n'
    s+=r'\begin{center}\small\begin{tabular}{p{.68\textwidth}rr}\toprule Condition & Trials & Clusters\\\midrule'+'\n'
    for r in rows:s+=f"{esc(r['title'])} & {r['n']} & {r['clusters']}"+r'\\'+'\n'
    s+=r'\bottomrule\end{tabular}\normalsize\end{center}'+'\n'
    maps=fs['maps'] if max(len(r['time']) for r in rows)<=7 else fs['landmarks']
    for name in maps:
        s+=r'\clearpage'+'\n'+fig(task,name,'Condition-by-time spatial allocation. Each row averages its stated trial group at a fixed physical location. The common square-root color scale is labeled in probability per patch. Original frame t and phase identify each observation; on full-series panels the signed number additionally gives time relative to evidence onset.')
    for name in fs['curves']:
        s+=r'\clearpage'+'\n'+fig(task,name,'Regional attention through the entire trial, separately for each condition. Solid traces are mean probability masses; shading gives pointwise 95 percent cluster-bootstrap intervals. Dotted lines show uniform allocation for each region. Background bands and labels identify trial phases. The vertical dashed line is the alignment event.')
    s+=r'\clearpage'+'\n'+fig(task,fs['comparison'],'Direct comparison of conditions on the same axes. Left: attention at the first named region, with pointwise uncertainty. Right: concentration of the condition-average map, measured as the exponential of its entropy. A value near 100 means a broad mean map; smaller values mean concentration on fewer sites. This summary is computed after averaging maps and is distinct from averaging each trial\'s concentration.')
    s+=r'\paragraph{Reading the observed trajectories.}'+'\n'
    for r in rows:
        a=np.array(r['mean_mass'])[:,0];k=int(a.argmax())
        s+=esc(f"{r['title']}: {r['region_names'][0].lower()} mass peaks at {a[k]:.3f} at original frame {k} ({r['phases'][k]}), and is {a[-1]:.3f} on the final frame. ")
    s+='These summaries locate the observed allocation in time; the neighboring intervention experiments test its causal contribution to behavior.\n\n% END CONDITION ATTENTION\n'
    return s

def appendix(task,data):
    from .papers import fig
    s='% BEGIN FULL CONDITION ATTENTION\n'+r'\clearpage\section{Condition-by-time attention: all frames}'+'\n'
    s+='Each panel retains the same condition definitions and color scale as the main attention section. Frames are aligned to the task-specific evidence onset. Gray cells mark times outside a particular trial, rather than missing observations. All observed frames are displayed.\n'
    for name in data['figures']['maps']:
        s+=r'\clearpage'+'\n'+fig(task,name,'Complete condition-average spatial trajectory. Rows are conditions, columns are aligned frames, and each title also records the original frame and task phase. Sample counts and a shared probability scale accompany every panel.')
    return s+'% END FULL CONDITION ATTENTION\n'

def enhance_source(task,source):
    path=OUT/task/'attention_dynamics.json'
    if not path.exists():return source
    data=read(path);main=section(task,data)
    if '% BEGIN CONDITION ATTENTION' in source:
        source=re.sub(r'% BEGIN CONDITION ATTENTION.*?% END CONDITION ATTENTION\n',lambda _:main,source,flags=re.S)
    else:
        a=source.index(r'\subsection{Allocation of shared spatial attention}')
        b=source.index(r'\subsection{Causal sensitivity to spatial and mnemonic interventions}',a)
        source=source[:a]+main+source[b:]
    extra=appendix(task,data)
    if '% BEGIN FULL CONDITION ATTENTION' in source:
        source=re.sub(r'% BEGIN FULL CONDITION ATTENTION.*?% END FULL CONDITION ATTENTION\n',lambda _:extra,source,flags=re.S)
    else:source=source.replace(r'\clearpage\section{Reproducibility record}',extra+r'\clearpage\section{Reproducibility record}')
    return source

def gallery():
    from matplotlib import colormaps
    folder=OUT/'attention';folder.mkdir(exist_ok=True)
    data={t:read(OUT/t/'attention_dynamics.json') for t in TASKS}
    lut=(colormaps['magma'](np.linspace(0,1,256))[:,:3]*255).astype(int).tolist()
    source=Path(__file__).with_name('attention_viewer.html').read_text(encoding='utf8')
    source=source.replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/')).replace('__LUT__',json.dumps(lut)).replace('__CONTEXT__',json.dumps(CONTEXT))
    (folder/'index.html').write_text(source,encoding='utf8')
    write(folder/'manifest.json',dict(tasks=19,conditions=sum(len(d['groups']) for d in data.values()),
        frames=sum(len(r['time']) for d in data.values() for r in d['groups']),
        selected_presentations=sum(r['n'] for d in data.values() for r in d['groups']),
        inference_rerun=False,time=time.time(),reference='https://arxiv.org/html/2502.10955v1'))

