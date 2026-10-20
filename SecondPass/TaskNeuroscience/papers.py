"""Build one evidence-linked LaTeX paper per task and an integrated volume."""
import argparse,csv,hashlib,html,json,os,subprocess,time
from pathlib import Path
import numpy as np
from .design import *
from .paper_content import CONTENT,REFERENCES,NEUROSCIENCE
from .interpretation import DISCUSSION
from .figures import figures

TOOLS=Path(__file__).with_name('Tools')
PAPERS=OUT/'papers'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def esc(s):
    s=str(s);return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak{}','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)
def pct(v):return 'unavailable' if v is None else f'{100*v:.1f}\\%'
def num(v):return 'unavailable' if v is None else f'{v:.3f}'
def ci(v):return f'[{pct(v[0])}, {pct(v[1])}]'
def para(s):return esc(s)+'\n\n'

def trial_protocol(task,cell):
    root=OUT/task;ms=read(root/'native'/f'{cell}.json');a=np.load(root/'native'/f'{cell}.npz');p=phases(task,ms[0],a['temporal'].shape[1]);runs=[];start=0
    for i in range(1,len(p)+1):
        if i==len(p) or p[i]!=p[start]:
            runs.append(f"frame {start}: {p[start]}" if i-start==1 else f"frames {start}--{i-1}: {p[start]}");start=i
    s=r'\subsection{Stimulus sequence and spatial layout}'+'\n'
    s+=f"For the representative native condition {esc(cell)}, the sequence has {len(p)} frames, indexed from zero: "+'; '.join(esc(v) for v in runs)+'. The complete condition table specifies the other tested timing and difficulty settings.\n\n'
    names={'displacement_pixels':'Dot displacement (pixels)','signed_orientation_degrees':'Signed grating rotation (degrees)',
        'contrast_increment':'Contrast increment','pedestal':'Contrast pedestal','frequency_octave_increment':'Frequency increment (octaves)',
        'chromatic_increment':'RGB chromatic increment','alignment_jitter_degrees':'Contour orientation jitter (degrees)',
        'path_elements':'Contour-path elements','total_elements':'Total elements','beta_delta':'Spectral-exponent difference',
        'rotation_magnitude':'Rotation magnitude (degrees)','motion_transitions':'Motion transitions','dots_per_patch':'Dots per aperture',
        'step_pixels':'Displacement per transition (pixels)','aperture_radius_pixels':'Aperture radius (pixels)',
        'study_load':'Study images','probe_hold':'Probe frames','theta':'Native rotation parameter (degrees)',
        'noise_degrees':'Native angular noise parameter (degrees)','displayed_validity':'Displayed cue validity','value':'Native trial reward value',
        'native_size':'Original raster width (pixels)','family':'Stimulus family','similarity':'Similarity condition',
        'spacing':'Candidate direction spacing (degrees)','difference':'Singleton feature difference'}
    rows=[]
    for key,title in names.items():
        vv=[m[key] for m in ms if key in m and isinstance(m[key],(str,int,float))]
        if not vv:continue
        unique=sorted(set(vv));value=', '.join(str(v) for v in unique) if len(unique)<=8 else f'{min(unique):.4g} to {max(unique):.4g}'
        rows.append((title,value))
    if rows:
        s+='The following values are observed in the 128 native trials of this representative condition. Continuous nuisance ranges summarize those draws.\n'
        s+=r'\begin{center}\small\begin{tabular}{p{.56\textwidth}p{.34\textwidth}}\toprule Stimulus property & Observed values\\\midrule'+'\n'
        for title,value in rows:s+=esc(title)+' & '+esc(value)+r'\\'+'\n'
        s+=r'\bottomrule\end{tabular}\normalsize\end{center}'+'\n'
    centers=ms[0].get('positions_xy',ms[0].get('centers_xy'))
    if centers is not None:s+='Stimulus centers in the 100 by 100 input are '+esc(str(centers))+'. Coordinate x increases rightward and y downward.\n\n'
    if task=='memory_guided_search':s+='Each candidate is a 28 by 28 raster. The sample appears centrally and the array contains the exact same sample raster translated to one of four locations. Feature stimuli combine color, shape and texture; irregular objects use radial Fourier contours; fractal stimuli use spectral fractional-noise textures with matched within-support intensity histograms. Similar feature arrays contain repeated individual feature values, requiring conjunction information to select a unique object.\n\n'
    if task=='spatial_binding':s+='The native orientation inventory consists of four angles spaced 45 degrees apart around one random base angle, randomly assigned to locations. The probe exchanges two locations while preserving that inventory.\n\n'
    return s
def fig(task,name,caption,width='.98'):
    return '\\begin{figure}[htbp]\n\\centering\n'+f'\\includegraphics[width={width}\\textwidth,height=.65\\textheight,keepaspectratio]{{../{task}/figures/{name}.pdf}}\n'+r'\caption{'+caption+'}\n\\end{figure}\n'

PREAMBLE=r'''\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,xcolor,hyperref,fancyhdr,microtype}
\definecolor{ink}{HTML}{164D70}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=ink,citecolor=ink}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\headheight}{15pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Computational neurophysiology: frozen suite}
\fancyhead[R]{\small 18 October 2026}\fancyfoot[C]{\thepage}
\setcounter{topnumber}{2}\setcounter{bottomnumber}{2}
\renewcommand{\floatpagefraction}{0.65}
\newcommand{\R}{\mathbb{R}}
\begin{document}
'''

METHODS=r'''
\section{Methods}
\subsection{Computational subject and experimental units}
The experimental subject is the previously trained 19-task model, with the main analysis fixed at update 204,289. This checkpoint was selected by the existing equal-task monitoring criterion before the present analysis. Its SHA-256 identifier is recorded in the reproducibility table. The final saved checkpoint at update 220,637 provides a matched robustness comparison. Model parameters are frozen throughout these experiments. A trial is a freshly generated native movie, its prescribed response, and the model's output. Independent procedural draws provide the sampling unit; resampling intervals quantify uncertainty for this trained subject under the specified stimulus distribution.

The 19-task training stage inherited the 16-task backbone and output heads, added three search-task heads, and continued optimization with Adam at learning rate $3\times10^{-4}$, moment coefficients $(0.9,0.999)$, numerical epsilon $10^{-8}$ and no weight decay. Effective batches contained 32 trials, accumulated in microbatches of four. The objective combined terminal cross-entropy with a 0.05-weighted pixel-prediction loss; VDA terminal cross-entropy additionally used the native reward value divided by three. Active-parameter gradient norm was clipped at 100. The selected checkpoint was saved during two-times static-memory-search sampling; the final checkpoint includes later four-times sampling. Checkpoint indices refer to this continuation stage, which itself inherited earlier learning.

Each native condition receives 128 new test trials. The final-checkpoint comparison replays the first 32 of those movies under the same generator namespace. Probe fitting uses 192 separate validation-split trials per selected condition and evaluation uses 128 test-split trials. Natural-image probe fitting and evaluation retain separate source-image splits. The native four-trial generator calls are preserved even when their tensors are combined into eight-trial GPU forwards. Inference uses float32; dense appearance, fusion and spatial recordings are stored in float16, while global memory, comparison memory and probabilities retain float32. Stimulus metadata defines analyses and interventions; only native pixels and the requested task head enter the observer's ordinary forward computation.

\subsection{Architecture and recording sites}
At each model frame, a 100 by 100 RGB (red, green and blue) image is represented at 100 sites arranged on a 10 by 10 grid. Each site carries 32 learned appearance channels, local RGB and contrast statistics, and temporal correspondence-derived features. A learned linear fusion maps these to 64 channels. The spatial recurrent state has 64 channels at each grid site; the global recurrent state, denoted $c_t$ and called CLS memory in the code, has 64 channels. CLS denotes the global classification-token pathway. A separate 735-dimensional comparison-memory vector accumulates selected local temporal-comparison evidence. The task head reads a learned 64-dimensional transformation of the concatenated global and comparison memories.

Let $v_{t,s}$ be the fused visual vector at site $s$, $h_{t-1,s}$ the preceding spatial memory, and $c_{t-1}$ the preceding global memory. The key and query are
\begin{align}
 k_{t,s}&=\operatorname{unit}\{W_k[\operatorname{LN}_v(v_{t,s})+\operatorname{LN}_h(h_{t-1,s})]\},\\
 q_t&=\operatorname{unit}\{W_q\operatorname{LN}_q(c_{t-1}+e_{\rm CLS})\},\\
 \ell_{t,s}&=\kappa\,k_{t,s}^{\mathsf T}q_t,\qquad
 \alpha_{t,s}=\frac{\exp\ell_{t,s}}{\sum_{u=1}^{100}\exp\ell_{t,u}}.
\end{align}
Here $W_k,W_q$ are learned matrices, $e_{\rm CLS}$ is a learned token vector, $\operatorname{LN}$ denotes layer normalization across channels, and $\operatorname{unit}$ divides a vector by its Euclidean norm with a numerical floor of $10^{-6}$. The expression $k^{\mathsf T}q$ is the key-query dot product. The scale is $\kappa=1+15\sigma(\rho)$, where $\rho$ is a learned scalar and $\sigma$ is the sigmoid function. Its value in the selected checkpoint is approximately 7.7525. The distribution $\alpha$ supplies the actual spatial-selection weights. There is one global query, so averaging the attention matrix over queries returns this same vector. The same spatial weights route learned visual values and temporal-comparison channels and influence spatial-memory writes.

Recordings are taken from appearance features, fused visual tokens, the spatial recurrent field, the global recurrent state, comparison memory and the attention distribution. We verify exact final-output and attention parity between the recording implementation and the authoritative model on all 19 tasks. Intermediate readouts are verified against forwards truncated at the corresponding frame. These controls establish that recording preserves the observer's original computation.

\subsection{Behavioral and signal-detection measures}
The model's response is the largest terminal class probability. Accuracy is the fraction of prescribed responses selected. Balanced accuracy averages recall across response classes. Macro AUC averages one-versus-rest receiver-operating-characteristic areas using continuous probabilities. Each native condition receives equal weight in the task summary. Wilson intervals accompany individual-condition accuracy; task intervals resample stimulus clusters within the fixed condition set. Matched procedural scenes remain together across timing conditions, and repeated photographs are grouped by image identity. A Wilson boundary interval is used when every observed response is correct or incorrect.

For binary judgements with both classes present, let $H$ and $F$ be hit and false-positive rates. With $h,f$ observed positive reports and $n_1,n_0$ opportunities, the finite-sample rates are $\widetilde H=(h+1/2)/(n_1+1)$ and $\widetilde F=(f+1/2)/(n_0+1)$. We report
\begin{equation}
 d'=\Phi^{-1}(\widetilde H)-\Phi^{-1}(\widetilde F),\qquad
 c=-\tfrac12[\Phi^{-1}(\widetilde H)+\Phi^{-1}(\widetilde F)].
\end{equation}
$\Phi^{-1}$ is the inverse standard-normal cumulative distribution. Sensitivity $d'$ measures separation under this signal-detection convention; criterion $c$ summarizes positive-response tendency. This scalar criterion is distinct from the time-indexed global-memory vector $c_t$. Labels and the meaning of a positive report are specified separately for each task. The Brier score in the condition tables is the trial-averaged sum $\sum_k[p(k)-\mathbf 1(y=k)]^2$, where $y$ is the prescribed class and $\mathbf 1(y=k)$ is one for that class and zero otherwise; smaller values indicate better probability predictions. Difficulty challenges use the existing audited renderers. Zero-strength observations retain their scheduled labels as an ambiguity control. The complete native factorial, empirical difficulty points and response probabilities are retained in machine-readable files.

\subsection{Threshold-defined chronometry}
The unchanged terminal readout is applied to each recorded prefix state. If $p_t(k)$ is its probability for class $k$, evidence becomes available at frame $t_0$, and $\theta$ is a fixed confidence threshold, then
\begin{equation}
 \tau_\theta=\min\{t-t_0+1:t\geq t_0,\ \max_k p_t(k)\geq\theta\}.
\end{equation}
The chosen class is the argmax at that first crossing. We use thresholds 0.6, 0.7, 0.8 and 0.9, fixed before inspection of outcomes. Trials without a crossing are right-censored at movie end. We report the crossing fraction, median frames among crossings, accuracy of crossing decisions and the fraction already above threshold before task evidence. Time is measured in model frames; two-observation tasks have only one post-evidence frame. This defines a reproducible computational commitment time for the trained readout.

\subsection{Spatial priority and causal manipulations}
Attention mass in a region $S$ is $A_t(S)=\sum_{s\in S}\alpha_{t,s}$. Spatial concentration is summarized by effective patch count, $N_{\rm eff}=\exp[-\sum_s\alpha_s\log\alpha_s]$, which equals 100 for a uniform distribution and approaches one for selection of a single patch. Spatial masks use model-site centers at pixel coordinates $4.5+10j$, where $j=0,\ldots,9$ on either axis. For point arrays, the designated mask surrounds the task-defined target center; the alternative is the next center in the renderer's location ordering, with wraparound. The radius is 16 pixels for search-array centers and 12 pixels for the other point arrays. Thus the alternative mask is a fixed geometric comparison location and need not coincide with the particular foil that changed. For VDA, masks contain the model-site centers mapped into each native cell by the exact nearest-neighbor resize; the alternative is the cell obtained by reversing the row-major cell index. Native VDA9 cues in this implementation occupy the two opposite corners, so these two masks are distinct. Full-field tasks use the top-left and bottom-right image quadrants as designated and alternative regions. Their designated sites are spatial test regions rather than inferred target objects. Off-site controls use a single site outside the two regional masks; the VDA control additionally minimizes Gabor occupancy.

Computational stimulation and inhibition add a signed dose $\delta$ to the priority logits at the selected sites and frames:
\begin{equation}
 \alpha^{(\delta)}_{t,s}=\operatorname{softmax}_s\{\ell_{t,s}+\delta\,\mathbf 1[s\in S]\mathbf 1[t\in E]\}.
\end{equation}
$E$ identifies encoding, delay or evidence frames. The indicator $\mathbf 1[\cdot]$ equals one when its condition holds and zero otherwise. Positive doses raise relative priority and negative doses lower it. The dose is measured in logit units. If a selected region has attention mass $A$ before adding a uniform regional dose, its immediate new mass is $A'=e^\delta A/(1-A+e^\delta A)$. Thus its odds against the remaining image are multiplied by $e^\delta$; a dose of four multiplies these odds by approximately 54.6. Later frames can additionally change through recurrent feedback. Evidence-epoch doses are $\pm2,\pm4,\pm8$ at designated and alternative sites; encoding and delay use $\pm4$. A missing epoch is explicitly recorded as zero active intervention trials. Off-site and global shifts provide controls. Every arm reuses identical input pixels. Adding the same constant to all logits must leave the softmax unchanged, and this invariance is checked during collection.

Memory manipulations replace one recurrent state by $g$ times that state after processing the middle delay frame, or after the frame immediately before evidence when no blank delay exists. The next frame is then processed from that modified state. Gains $g=0$ and $g=1.5$ respectively reset and amplify the global state, spatial state or comparison memory, one component at a time. The movie then continues normally. Paired accuracy differences compare each arm with the same-trial sham. Intervals use 3,000 paired resamples; exact discordant-pair binomial tests assess changes in correctness for independent procedural trials. For natural photographs, repeated probe or base-image identities remain grouped in bootstrap resamples, and 9,999 paired source-image sign flips test the accuracy contrast. The saved test-method field distinguishes those Monte Carlo cluster tests from the exact trial-level tests. Holm adjustment is applied within task, and the four designated $\pm4$ evidence comparisons also enter one family across tasks and selected conditions.

\subsection{Independent-trial decoding and population geometry}
Each region is transformed using probe-fitting trials only. Representations wider than 256 coordinates undergo a seeded, label-independent random projection to 256 dimensions. A fixed training-derived principal-component basis then retains at most 32 dimensions and is shared across time. Coordinates are standardized from the fitting data. This capacity-matched representation is used for external linear probes; the original agent remains frozen.

For a design matrix $X$, target matrix $Y$ and fixed ridge penalty $\lambda_{\rm ridge}=10$, the fitted mapping minimizes
\begin{equation}
 \|Y-XW-\mathbf 1 b^{\mathsf T}\|_F^2+\lambda_{\rm ridge}\|W\|_F^2.
\end{equation}
Each row of $X$ represents one trial and each column one retained activation coordinate. Each row of $Y$ contains that trial's desired decoding target. The matrix $W$ contains fitted weights and $b$ contains output intercepts; $\mathbf 1$ is a column of ones. The squared Frobenius norm $\|\cdot\|_F^2$ sums squared entries of a matrix. The first term therefore measures prediction error, while the second penalizes large fitted weights. Classification targets are one-hot answer labels, the model's eventual choice, and cue or target location where available. Continuous targets include an 8 by 8 RGB descriptor of the last sample/study image and circular orientation or motion variables when provided by the task. Continuous reconstruction is scored against a predictor using the fitting-set mean. The resulting $R^2$ can be negative when the fitted mapping predicts new trials worse than that baseline.

For the spatial and global recurrent states, a decoder fitted at each time is evaluated at every time on independent test trials. The vertical axis is training time and the horizontal axis is test time. Diagonal performance measures available information at the same phase; off-diagonal performance tests transfer of the fitted linear readout between phases. A final-time label-permutation test uses 199 fitting-label shuffles, with unchanged held-out labels. Population trajectories use the same training-derived axes, and their class means are computed from held-out trials. Participation ratio is $(\sum_j\lambda_j)^2/\sum_j\lambda_j^2$ for covariance eigenvalues $\lambda_j$ within the retained subspace. Frame-to-frame root-mean-square (RMS) displacement measures how rapidly the state evolves in those coordinates.
'''

def build(task):
    root=OUT/task;b=read(root/'behavior.json');ca=read(root/'causal.json');neural=read(root/'neural_summary.json');fm=read(root/'figure_manifest.json');att=read(root/'attention_summary.json');content=CONTENT[task]
    PAPERS.mkdir(exist_ok=True);cell=neural_cells(task)[0];classes=TASKS[task]['classes']
    regions=neural[0]['regions'];top=max([(name,r) for name,r in regions.items() if not r.get('constant_zero')],key=lambda z:z[1]['final_auc'])
    causal=[r for r in ca['interventions'] if r['cell']==cell and r['active_trials'] and r['arm']!='sham' and 'global' not in r['arm']]
    largest=min(causal,key=lambda r:r['delta_accuracy']);best=max(causal,key=lambda r:r['delta_accuracy'])
    robust=float(np.mean([r['delta_accuracy'] for r in b['latest_comparison']]))
    s=PREAMBLE+f'\\title{{\\color{{ink}}{esc(content["title"])}}}\n'+r'\author{Visual Attention and Working Memory Project\\Computational experiment series: previous trained model}\date{18 October 2026}\maketitle'+'\n'
    s+=r'\begin{abstract}'+'\n'
    s+=f"We characterized {esc(task.replace('_',' '))} in a frozen recurrent visual observer using behavioral challenges, spatial-priority interventions, memory perturbations and population decoding. Across {b['conditions']} native conditions and {b['trials']:,} fresh trials, equal-condition accuracy was {pct(b['equal_cell_accuracy'])} (95\\% interval {ci(b['equal_cell_ci95'])}). "
    s+=f"For the representative condition {esc(cell)}, the strongest final answer readout among the tested regions was {esc(top[0])}, with held-out macro AUC {num(top[1]['final_auc'])}. "
    s+=f"The largest observed impairment in that condition was {esc(largest['arm'])}, changing accuracy by {100*largest['delta_accuracy']:+.1f} percentage points. "
    s+='The analyses relate task competence to information availability, spatial selection and state dependence, with paired interventions and independent probe trials.\n'+r'\end{abstract}'+'\n'
    s+=r'\section{Introduction and scientific questions}'+'\n'+para(content['intro'])+para(content['rationale'])
    s+='The biological motivation is developed through '+', '.join(r'\cite{'+k+'}' for k in content['refs'])+'. The present experiments use the model as an accessible computational subject: its choices, internal population coordinates and interventions can be measured on exactly matched movies.\n\n'
    s+=r'\subsection{Neuroscientific rationale}'+'\n'+para(NEUROSCIENCE[task])
    s+=r'\subsection{Task definition and prediction structure}'+'\n'+para(TASKS[task]['question'])
    s+='Response labels are '+'; '.join(f'{i}: {esc(label)}' for i,label in enumerate(TASKS[task]['labels']))+'. '
    s+=f"The native suite contains {b['conditions']} conditions for this task. A complete condition table follows the main results. Detailed neural and intervention measurements use "+', '.join(esc(c) for c in neural_cells(task))+'.\n\n'
    s+=trial_protocol(task,cell)+para(content['interpretation'])+METHODS
    s+=r'''\subsection{Full-state controls and unit-level selectivity}
As a control for dimensionality reduction, external ridge classifiers are also fitted directly to every recorded coordinate at the final frame and at the retention boundary. The ridge penalty is selected from 0.001, 0.01, 0.1, 1 and 10 using a class-stratified 75/25 split inside the probe-fitting set. Standardization is fitted on the inner fitting subset during selection and on the full fitting set for the final held-out evaluation. Each standardized full-coordinate vector is additionally divided by the square root of its coordinate count; these penalty values refer to that normalized feature scale. The full-crop reconstruction and candidate-reconstruction regressions use the same scaling convention. These full-state probes supplement the fixed-capacity temporal analysis. Global-memory unit selectivity is assessed by response-class differences on held-out trials, with Holm correction across the 64 units. Cross-validated class-centroid distances multiply independently estimated fitting and test centroid differences, standardized by fitting-set channel dispersion; negative estimates are retained.

For native factorial summaries, bootstrap clusters retain the same procedural scene across matched timing conditions. Photo repeats are grouped by probe or base image identity within condition. This preserves the dependence generated by reusing stimuli. The saved condition-level results allow every marginal to be reconstructed and interactions to be examined directly.
'''
    s+=r'\section{Results}'+'\n'+r'\subsection{Native behavior, difficulty and checkpoint robustness}'+'\n'
    s+=f"Native equal-condition accuracy was {pct(b['equal_cell_accuracy'])}, with macro AUC {num(b['equal_cell_auc'])}. The matched final checkpoint changed accuracy by {100*robust:+.2f} percentage points on average across conditions, using 32 same-movie comparisons per condition. "
    worst=min(b['cells'],key=lambda r:r['accuracy']);highest=max(b['cells'],key=lambda r:r['accuracy'])
    s+=f"The lowest native cell was {esc(worst['cell'])} at {pct(worst['accuracy'])}; the highest was {esc(highest['cell'])} at {pct(highest['accuracy'])}. These comparisons retain the native sampling design rather than selecting trials after observing success.\n\n"
    if b['psychometric']:
        points=[r for r in b['psychometric'] if r['strength']>0 and r['event'] not in ('catch','foil','uncued')]
        if points:
            low=min(points,key=lambda r:r['strength']);high=max(points,key=lambda r:r['strength'])
            s+=f"In the empirical difficulty challenge, the smallest positive tested strength ({low['strength']:g}) produced {pct(low['accuracy'])} correct responses, compared with {pct(high['accuracy'])} at the largest strength ({high['strength']:g}). Each point uses {PSYCH_N} trials. The full curve, including intermediate levels and event-specific responses, is shown in Figure~\\ref{{fig:behavior}}.\n\n"
    else:
        for dim in ('family','similarity','delay','sample_frames','exposure','load','spacing','difference'):
            rows=[r for r in b['factors'] if r['factor']=='condition_'+dim]
            if not rows:continue
            vals='; '.join(f"{esc(r['value'])}: {pct(r['accuracy'])}" for r in rows)
            s+=f"Marginal accuracy by {esc(dim.replace('_',' '))} was {vals}. These marginal summaries average over the other native factors; the condition table preserves their interactions.\n\n"
    s+=fig(task,'behavior',r'\label{fig:behavior}Behavioral phenotype. Left: accuracy and Wilson intervals for every native condition. Middle: empirical signal-strength functions or native factorial margins. Right: observed accuracy within confidence bins, against the identity line. Zero-strength challenges assess responses to ambiguous physical evidence. Native cells contain 128 trials each.')
    s+=r'\subsection{Time course and commitment thresholds}'+'\n'
    cr=[r for r in b['chronometry'] if r['cell']==cell]
    s+=r'\begin{center}\begin{tabular}{rrrrr}\toprule Threshold & Crossed & Median frames & Crossing accuracy & Premature\\\midrule'+'\n'
    for r in cr:s+=f"{r['threshold']:.1f} & {pct(r['crossing_fraction'])} & {r['conditional_median_frames'] if r['conditional_median_frames'] is not None else '--'} & {pct(r['conditional_accuracy'])} & {pct(r['premature_fraction'])} \\\n"
    # Escape a trailing table row with two literal backslashes.
    s=s.replace(' \\\n',' \\\\\n')
    s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
    s+=f"The representative condition {esc(cell)} provides {cr[0]['available_frames']} post-evidence frames. Median latency is conditional on crossing; the uncommitted-fraction plot retains all trials. The threshold sweep makes the speed--coverage tradeoff explicit. A rise in correct-answer probability over time can be compared with the development of location and answer decodability.\n\n"
    s+=fig(task,'chronometry','Threshold-defined timing. Left: the fraction reaching each confidence threshold and accuracy among those crossings. Middle: the fraction still uncommitted after each available evidence frame. Right: probability assigned to the correct response and accuracy of the unchanged head applied to prefix states. The dashed vertical line marks evidence onset.')
    s+=r'\subsection{Allocation of shared spatial attention}'+'\n'
    s+=f"At the final frame of {esc(cell)}, the designated region received mean attention mass {att['mean_final_site_mass']:.3f}, compared with {att['mean_final_alternative_mass']:.3f} at its designated alternative. The designated region occupies {pct(att['mean_site_area_fraction'])} of model sites. Mean effective patch count was {att['mean_final_effective_patches']:.2f}. "
    s+=f"Outcome-conditioned traces contain {att['correct_trials']} correct and {att['error_trials']} error trials. Their separation describes an association between allocation and outcome; the matched interventions below test causal effects.\n\n"
    s+=fig(task,'attention','Shared-priority dynamics. Location masses sum the native attention probabilities within the specified regions. Effective patch count translates entropy into the size of an equally weighted population. Correct/error traces use the same predeclared geometry. The supplementary atlas shows every frame of a representative movie and its spatial projections.')
    s+=r'\subsection{Causal sensitivity to spatial and mnemonic interventions}'+'\n'
    for r in (largest,best):
        s+=f"For {esc(r['arm'])}, paired accuracy changed by {100*r['delta_accuracy']:+.1f} percentage points (paired 95\\% interval {ci(r['ci95'])}; paired p={r.get('p_value',r['p_exact']):.4g}; within-task Holm p={r['holm_within_task']:.4g}). There were {r['gained']} corrected trials and {r['lost']} newly incorrect trials. "
    s+='These extrema describe the observed exploratory sweep. The four predesignated spatial comparisons at dose magnitude four are evaluated jointly across the full series in the integrated synthesis.\n\n'
    s+=fig(task,'causal','Matched causal effects in the representative condition. Left: changes from sham across signed spatial-priority doses during available evidence. Right: resetting or amplifying one memory component at the specified frame. Intervals resample matched trials, preserving sham--intervention pairing. Target and distractor labels denote the task geometry; full-field tasks use fixed quadrants.')
    if ca['counterfactual']:
        rr=[r for r in ca['counterfactual'] if r['arm']=='cued_-4_change']
        s+=r'\subsection{Bottom-right cue and top-left counterfactual change}'+'\n'
        for r in rr:s+=f"For {esc(r['event'])} events, bottom-right inhibition at dose -4 changed correct-report frequency by {100*r['delta_accuracy']:+.1f} percentage points, with paired interval {ci(r['ci95'])}. "
        s+='The 100 percent cue and uncued-change combination is deliberately outside the native cue-validity rule. A change at top left still has a correct positive label because VDA requires a report for a change anywhere. Catch trials have no physical change and provide the false-alarm control.\n\n'
        s+=fig(task,'counterfactual','Requested counterfactual experiment. Every movie has a 100 percent bottom-right cue. Change reports are plotted separately for changes at the cue, changes at top left and catches. Priority perturbations occur only on the two change frames. Each event type contains 256 matched nuisance draws per intervention arm.')
    s+=r'\subsection{Readable information across representation levels}'+'\n'
    s+=r'\begin{center}\begin{tabular}{lrrr}\toprule Region & Final AUC & Balanced accuracy & Permutation p\\\midrule'+'\n'
    for name,r in regions.items():
        if r.get('constant_zero'):s+=esc(name)+r' & \multicolumn{3}{l}{Constant zero on this condition}\\'+'\n'
        else:s+=f"{esc(name)} & {r['final_auc']:.3f} & {pct(r['final_balanced_accuracy'])} & {r['final_permutation_p']:.3f}"+r'\\'+'\n'
    s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
    s+='All table entries are evaluated on held-out trials. Final-time permutation p-values quantify how unusual the observed decoding is relative to shuffled fitting labels; the full regional panel is exploratory. The capacity-matched probes quantify readable information in the specified projected subspaces. Choice, location and continuous-feature targets are retained in the accompanying JSON results.\n\n'
    s+=fig(task,'decoding','Information across six recording sites. The answer-decoding curves report held-out macro AUC. Sample-content curves report reconstruction of a coarse last-sample or last-study image descriptor relative to a fitting-set-mean predictor. Each region uses an independently fitted training-only projection, shared across time; the agent itself is unchanged.')
    s+=r'\subsection{Mnemonic population geometry and code stability}'+'\n'
    for region in ('cls','spatial'):
        r=regions[region];s+=f"In {esc(region)} memory, the participation ratio ranged from {min(r['participation_ratio']):.2f} to {max(r['participation_ratio']):.2f} within the retained subspace. The final step displacement was {r['state_step_rms'][-1]:.3f} standardized RMS units. "
    s+='The time-generalization matrices distinguish a stable readout from information that is readable only by a phase-specific decoder. Broad off-diagonal AUC supports persistence of a common linear discriminant; a strong diagonal with weaker transfer supports changing code geometry. Trajectories use one set of axes fitted to independent fitting trials, so apparent movement is not caused by refitting the axes at each frame.\n\n'
    nmeta=read(root/'neural'/f'{cell}_test.json');ntime=len(regions['cls']['diagonal']);boundary=retention_boundary(task,nmeta[0],ntime)
    continuous=[k for k in regions['cls'] if k.startswith('remembered_') and k.endswith('_r2')]
    if continuous:
        key=continuous[0];description='the sample motion direction, represented by its cosine and sine' if 'direction' in key else 'the sample orientation at the instructed location, represented by the cosine and sine of twice its angle'
        if task=='spatial_binding':description='the original orientation at the location that will be cued after retention'
        s+=f"At retention-boundary frame {boundary}, reconstruction of {description} had held-out R-squared {regions['cls'][key][boundary]:.3f} in CLS memory and {regions['spatial'][key][boundary]:.3f} in spatial memory. This continuous target describes stimulus content, while the final-answer decoder tests the response required after the subsequent evidence.\n\n"
        if task=='spatial_binding':s+='Before the retrospective cue, the model has not been told which location this single-target regression will query. The all-four-location inventory analysis below directly tests the stored orientation-location bindings without requiring advance knowledge of the future cue.\n\n'
    else:
        vals=[regions[r]['sample_content_r2'][boundary] for r in ('cls','spatial')]
        if all(v is not None for v in vals):s+=f"At retention-boundary frame {boundary}, coarse sample-content reconstruction had R-squared {vals[0]:.3f} from CLS memory and {vals[1]:.3f} from spatial memory. These values quantify reconstruction of the specified 8-by-8 descriptor on unseen trials.\n\n"
    reset=[r for r in ca['interventions'] if r['cell']==cell and 'gain0_' in r['arm']]
    s+='At the same intervention boundary, resetting each component changed paired accuracy as follows: '+', '.join(f"{esc(r['arm'].split('_')[0])}, {100*r['delta_accuracy']:+.1f} percentage points" for r in reset)+'. Comparing these causal effects with the decoding curves distinguishes information that is recoverable from a state from information whose removal at that instant affects the subsequent choice.\n\n'
    s+=fig(task,'population','Temporal generalization and held-out population trajectories. Top: decoders fitted separately at each frame and tested across all frames; chance AUC is 0.5. Bottom: class-mean trajectories in a shared training-derived coordinate system, with squares marking starts and stars marking ends. Spatial and CLS populations are analyzed separately.')
    s+=fig(task,'memory','Population dynamics and mnemonic variables. Left: covariance participation ratio within the retained subspace. Middle: frame-to-frame RMS state displacement. Right: continuous remembered-feature reconstruction when supplied by the task, otherwise location decoding or response-class centroid variance as stated by the legend. Each quantity is tied to the original sequence clock.')
    if fm.get('extra'):
        s+=r'\subsection{Condition-resolved mechanisms}'+'\n'
        s+='The additional panels retain differences between selected stimulus families and timing conditions. Family averages summarize a balanced native factorial; cells remain available individually in the appendix and source data. The decoding comparison uses the same external fitting protocol for every listed condition.\n\n'
        for name in fm['extra']:
            caption='Held-out final answer decoding across selected conditions and recording sites. A constant-zero comparison representation is displayed at chance AUC.' if name=='condition_decoding' else 'Event-specific causal response functions. Curves show positive reports after evidence-epoch target or alternative-site intervention; the horizontal line is paired sham. Event counts are stated above each panel. A positive report has the task-specific meaning defined in the text.' if name=='event_causality' else 'Memory-search factorial. Each cell shows accuracy averaged over the other exposure or sample-duration factor, within the indicated family/similarity or motion-spacing group. Numbers are percent correct; all contributing native conditions contain 128 trials. Delay, encoding time and search time remain distinct factors.'
            s+=fig(task,name,caption)
    discussion=r'\clearpage\section{Discussion}'+'\n'+para(DISCUSSION[task])
    discussion+=f"The behavioral result establishes {pct(b['equal_cell_accuracy'])} native accuracy for this trained subject across the tested condition distribution. The information result identifies {esc(top[0])} as the strongest final regional readout in the representative condition under the fixed probe procedure. The causal result identifies {esc(largest['arm'])} as the largest observed impairment in that condition. These three findings concern distinct levels of explanation: successful behavior, recoverable information and dependence on a manipulated component. Their conjunction is more informative than any one alone.\n\n"
    discussion+='The most useful interpretation of a null manipulation is determined by its interval and by whether it altered an active component at a relevant time. Zero-active-frame interventions are identified in the data, and comparison memory can be exactly zero in short sequences. The shared softmax also makes priority changes competitive: increasing one region necessarily changes the relative mass elsewhere. Thus the causal object is the complete redistribution induced by the specified intervention.\n\n'
    discussion+=para(content['next'])
    deep_path=root/'deep_analysis.json'
    if deep_path.exists():
        deep=read(deep_path);dc=deep['full_state_probes'][0]
        s+=r'\subsection{Full-state probe control and population selectivity}'+'\n'
        s+=r'\begin{center}\begin{tabular}{lrrr}\toprule Region & Full final AUC & Full retention AUC & Coordinates\\\midrule'+'\n'
        for region,rr in dc['regions'].items():
            final=rr['final'];ret=rr['retention_boundary']
            s+=esc(region)+f" & {num(final.get('auc'))} & {num(ret.get('auc'))} & {final.get('dimensions','--')}"+r'\\'+'\n'
        s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
        sig=sum(r['holm_p']<.05 for r in dc['cls_unit_selectivity'])
        s+=f"The full-state control uses the original coordinate width, while the main temporal figures retain at most 32 components. Both evaluations use independent held-out trials. At the final frame, {sig} of 64 global-memory units had response-class selectivity at within-population Holm p below 0.05. This is selectivity for the task answer in a recurrent activation coordinate. The complete unit effects and cross-validated centroid-distance trajectories are stored in \\texttt{{deep\\_analysis.json}}.\n\n"
        if deep['feature_errors']:
            er=deep['feature_errors'];s+=f"For similar feature-conjunction arrays, there were {er.get('errors',0):,} errors among {er.get('trials',0):,} evaluated trials across the timing factorial. Among errors, the selected alternative shared the target color {er.get('wrong_shared_color',0):,} times, shape {er.get('wrong_shared_shape',0):,} times and texture {er.get('wrong_shared_texture',0):,} times. These are descriptive counts across repeated-scene timing conditions; the scene-clustered analyses supply the uncertainty framework.\n\n"
    s+=discussion
    s+=r'\subsection{Scope of inference}'+'\n'
    s+='This series studies one trained parameter realization. Its uncertainty intervals describe stimulus sampling for that realization. Threshold-defined latencies are in model frames and reflect the existing classifier applied to prefix states. Spatial doses are changes to computational priority logits, and memory perturbations act on recurrent tensors. Independent training seeds, calibrated behavioral clocks and biological intervention models are the next levels of replication. The empirical results, code and task definitions are sufficient to reproduce the computational experiments reported here.\n\n'
    refs=list(dict.fromkeys(content['refs']+['king','stokes']))
    s+=r'\begin{thebibliography}{99}'+'\n'
    for key in refs:
        title,url=REFERENCES[key];s+=r'\bibitem{'+key+'} '+esc(title)+r' \url{'+url+'}.\n'
    s+=r'\end{thebibliography}\clearpage\appendix\section{Complete native condition results}'+'\n'
    s+=r'\small\begin{longtable}{p{.44\textwidth}rrrr}\toprule Condition & n & Accuracy & AUC & Brier\\\midrule\endhead'+'\n'
    for r in b['cells']:s+=f"{esc(r['cell'])} & {r['n']} & {pct(r['accuracy'])} & {num(r['auc'])} & {r['brier']:.3f}"+r'\\'+'\n'
    s+=r'\bottomrule\end{longtable}\normalsize\section{Every-frame attention atlas}'+'\n'
    s+=f"Condition {esc(cell)}. The first two rows show one exact native trial and its own attention overlay. The third row averages attention over {fm['atlas_group_n']} trials sharing the selected location/cue and response label. Every model frame is included. Averaging unrelated object identities is restricted to the attention map; the displayed RGB movie remains one real trial.\n"
    for name in fm['atlas']:s+=fig(task,name,'Native frames, own attention overlay and condition-group spatial projection. Shared color scaling within each panel represents attention probability per patch.')
    s+=r'\clearpage\section{Reproducibility record}'+'\n'
    s+='Main checkpoint: update 204,289. Final robustness checkpoint: update 220,637.\n\n'
    for name,h in EXPECTED.items():s+=esc(name)+r': \texttt{'+h[:32]+r'}\par\texttt{'+h[32:]+'}\n\n'
    s+=f"Data namespace {NS}; probe fitting {NS+1}; probe testing {NS+2}; interventions {NS+3}; difficulty challenges {NS+4}; VDA counterfactuals {NS+5}. The analysis package is \\texttt{{SecondPass/TaskNeuroscience}}. The protocol, source hashes, parity checks, per-trial NPZ files, metadata, statistical JSON files and editable LaTeX sources are retained beside this paper. Figures and text are generated directly from completed analysis results.\n"
    s+=r'\end{document}'
    path=PAPERS/f'{task}.tex';path.write_text(s,encoding='utf8')
    return path

def compile_paper(path):
    # Flush queued figures before long-table sections. Otherwise TeX's longtable
    # output routine can place deferred floats over the following text page.
    source=path.read_text(encoding='utf8')
    if path.stem in TASKS:
        from .attention_report import enhance_source
        source=enhance_source(path.stem,source)
        from .neuroscience_report import enhance_source as add_neuroscience
        source=add_neuroscience(path.stem,source)
    for heading in ('Matched-scene effects of sample, delay and search time',
                    'Can the decoded memory identify the matching object?',
                    'Matching a decoded memory to decoded visual candidates',
                    'Identical-array, changed-memory experiment'):
        token=r'\subsection{'+heading+'}'
        if token in source and r'\clearpage'+token not in source:
            source=source.replace(token,r'\clearpage'+token)
    if r'\emergencystretch=1em' not in source:
        source=source.replace(r'\begin{document}',r'\emergencystretch=1em'+'\n'+r'\begin{document}')
    if r'\captionof' in source and r'\usepackage{capt-of}' not in source:
        source=source.replace(r'\begin{document}',r'\usepackage{capt-of}'+'\n'+r'\begin{document}')
    path.write_text(source,encoding='utf8')
    command=[str(TOOLS/'tectonic.exe'),'--keep-logs','--outdir',str(PAPERS),str(path)]
    r=subprocess.run(command,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=600)
    (PAPERS/(path.stem+'_compile.txt')).write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
    if r.returncode:raise RuntimeError(f'LaTeX failed: {path.name}: {r.stderr[-1500:]}')

def index():
    rows=[]
    for task in TASKS:
        pdf=PAPERS/f'{task}.pdf'
        if not pdf.exists():continue
        b=read(OUT/task/'behavior.json');rows.append(dict(task=task,title=CONTENT[task]['title'],accuracy=b['equal_cell_accuracy'],auc=b['equal_cell_auc'],trials=b['trials']))
    if not rows:return
    page='''<!doctype html><meta charset="utf-8"><title>Computational neurophysiology: 19 task papers</title><style>body{font:17px/1.65 system-ui;color:#18354b;background:#eef3f6;max-width:1100px;margin:55px auto;padding:20px}h1{font-size:42px;line-height:1.15}article{background:white;padding:24px;margin:18px 0;border-radius:12px}a{color:#12617c}small{color:#526672}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #d8e2e8}img{width:100%}</style><h1>Behavior, attention and memory<br>in a shared visual observer</h1><p>One scientific paper per task. Previous model: selected checkpoint 204,289; final-checkpoint robustness at 220,637. New frozen-model experiments collected 18 October 2026.</p>'''
    page+='<p><strong><a href="papers/synthesis.pdf">Read the synthesis</a> · <a href="papers/complete_volume.pdf">Complete volume</a> · <a href="nineteen_task_papers_and_source.zip">Download papers, LaTeX and source results</a></strong></p><p><a href="protocol.json">Experiment protocol</a> · <a href="primary_family.json">Across-task corrected causal tests</a> · <a href="data_audit.json">Data audit</a> · <a href="pdf_audit.json">PDF audit</a></p>'
    if (OUT/'attention/index.html').exists():
        page+='<article><h2>Attention across conditions and time</h2><p><a href="attention/index.html">Open the interactive attention atlas</a>: compare conditions, play exact trial inputs with their own attention overlays, and inspect condition-average maps and regional time courses for all nineteen tasks.</p></article>'
    if (PAPERS/'neuroscience_comparison.pdf').exists():
        page+='<article><h2>Biological experiments and theory</h2><p><a href="papers/neuroscience_comparison.pdf">Read the focused neuroscience companion</a>. Every task is compared with multiple human or animal experiments, with detailed interpretation of agreement, differences and the next discriminating test. The same comparisons are integrated into the individual task papers.</p><p><a href="literature_sources.json">43-source evidence ledger</a> · <a href="literature_coverage.json">Task-to-study mapping</a> · <a href="papers/neuroscience_comparison.tex">LaTeX source</a></p><p>Additional saved-trial analyses test temporal shortcuts in motion-duration judgments and separate recognition sensitivity from response criterion. These analyses use the previous frozen model and add no new training.</p></article>'
    page+='<nav style="columns:2;background:white;padding:20px;border-radius:12px"><h2>Choose a task</h2>'+''.join(f'<div><a href="#task-{r["task"]}">{html.escape(r["task"].replace("_"," "))}</a> — {r["accuracy"]:.1%}</div>' for r in rows)+'</nav>'
    for r in rows:
        t=r['task'];page+=f'<article id="task-{t}"><h2>{html.escape(r["title"])}</h2><p>{r["trials"]:,} native trials · Accuracy {r["accuracy"]:.1%} · AUC {r["auc"]:.3f}</p><p><a href="papers/{t}.pdf">Read paper (PDF)</a> · <a href="papers/{t}.tex">LaTeX source</a> · <a href="{t}/behavior.json">Behavioral results</a> · <a href="{t}/causal.json">Causal results</a> · <a href="{t}/neural_summary.json">Population results</a></p><p><a href="{t}/figures/attention.png">Attention plot</a> · <a href="{t}/figures/causal.png">Intervention plot</a> · <a href="{t}/figures/decoding.png">Decoding plot</a> · <a href="{t}/figures/full_state_geometry.png">State-space geometry</a></p><img loading="lazy" src="{t}/figures/behavior.png" alt="Behavioral analysis"></article>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    write(OUT/'paper_index.json',rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--task');p.add_argument('--no-compile',action='store_true');p.add_argument('--rebuild',action='store_true');a=p.parse_args()
    wanted=[a.task] if a.task else list(TASKS);done=set()
    while True:
        for task in wanted:
            if task in done:continue
            if not (OUT/task/'statistics_complete.json').exists():continue
            if a.rebuild or not (PAPERS/f'{task}.pdf').exists():
                from .statistics import behavioral
                behavioral(task)
                write(OUT/'paper_status.json',dict(task=task,phase='figures_and_paper',pid=os.getpid(),time=time.time()))
                figures(task);path=build(task)
                if not a.no_compile:compile_paper(path)
            done.add(task);index()
        if len(done)==len(wanted) or not a.watch:break
        time.sleep(20)
    if len(done)==len(TASKS):write(OUT/'papers_complete.json',dict(tasks=list(done),time=time.time()))

if __name__=='__main__':main()
