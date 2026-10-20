"""Insert literature comparisons into all papers and build a focused companion."""
import json,re,html,time
from .design import OUT,TASKS
from .literature_sources import SOURCES
from .neuroscience_content import CHAPTERS

BEGIN='% BEGIN BIOLOGICAL COMPARISON'
END='% END BIOLOGICAL COMPARISON'
REFBEGIN='% BEGIN BIOLOGICAL REFERENCES'
REFEND='% END BIOLOGICAL REFERENCES'
RESULTBEGIN='% BEGIN BIOLOGICAL RESULTS'
RESULTEND='% END BIOLOGICAL RESULTS'
DISCBegin='% BEGIN BIOLOGICAL DISCUSSION'
DISCEnd='% END BIOLOGICAL DISCUSSION'
THEORY={'contrast':['normalization'],'memory_guided_search':['wolfe'],
        'popout_search':['wolfe'],'motion_memory_search':['wolfe']}

def read(p):return json.loads(p.read_text(encoding='utf8'))
def references(task):return list(dict.fromkeys([s[0] for s in CHAPTERS[task]['studies']]+THEORY.get(task,[])))
def bib(keys):
    from .papers import esc
    return ''.join(r'\bibitem{bio_'+k+'} '+esc(f"{SOURCES[k]['authors']} ({SOURCES[k]['year']}). {SOURCES[k]['title']}.")+' '+r'\href{'+SOURCES[k]['url']+r'}{Primary source}.'+'\n' for k in keys)

def reanalysis(task):
    from .papers import esc,fig
    p=OUT/task/'literature_analysis.json'
    if not p.exists():return ''
    rows=read(p)['rows'];s=r'\subsection{Additional analysis prompted by the biological comparison}'+'\n'
    if task=='motion_duration_cued':
        s+='The full eight-transition count winner is compared with three fixed shortcuts: report the first direction, report the last direction, or report the most frequent direction in the last three transitions (ties favor the most recent tied direction). These rules use the cued stream and receive perfect access to its native metadata. The trained model receives pixels. Thus this comparison gives each shortcut an especially favorable sensory input.\n\n'
        s+='Within each native condition, there are 128 independent procedural draws. The conflict subset consists of trials on which the shortcut disagrees with the correct count winner. Reported intervals are Wilson 95 percent intervals within that subset. Conditions remain separate; no additional model forwards or fitted parameters are used.\n\n'
        s+=fig(task,'literature_temporal_shortcuts','Saved-trial temporal-strategy comparison. Left: trained-model and shortcut accuracies. Right: model accuracy restricted to trials where each shortcut is wrong. Error bars are within-condition Wilson intervals. Dotted line marks four-choice chance. The plotted delay affects retention after the same number of motion transitions.')
        s+=r'\begingroup\small\renewcommand{\arraystretch}{.95}\begin{longtable}{rrlrrr}\toprule Delay & $n$ & Shortcut & Rule correct & Conflict $n$ & Model on conflict\\\midrule\endhead'+'\n'
        for r in sorted(rows,key=lambda r:(r['delay'],r['strategy'])):
            name={'first_transition':'First','last_transition':'Last','last_three_transitions':'Last three'}[r['strategy']]
            s+=f"{r['delay']} & {r['n']} & {name} & {100*r['heuristic_accuracy']:.1f}\\% & {r['conflict_trials']} & {100*r['model_conflict_accuracy']:.1f}\\%"+r'\\'+'\n'
        s+=r'\bottomrule\end{longtable}\endgroup'+'\n'
        s+='The model remains accurate when these short-history rules fail. This directly rejects the tested shortcut descriptions for these movies. It establishes use of a richer temporal summary; order-controlled evidence is the next test for distinguishing exact counts from other recurrent summaries.\n'
    else:
        s+='The table re-expresses the existing native recognition decisions as repeat hits, novel-image false alarms, sensitivity and criterion. Positive means ``previously studied.'' Sensitivity is '+r"$d'=\Phi^{-1}(\widetilde H)-\Phi^{-1}(\widetilde F)$"+' and criterion is '+r"$c=-[\Phi^{-1}(\widetilde H)+\Phi^{-1}(\widetilde F)]/2$."+' Here '+r'$\Phi^{-1}$'+' converts a cumulative probability to a standard-normal quantile; '+r'$\widetilde H=(h+0.5)/(n_1+1)$'+' and '+r'$\widetilde F=(f+0.5)/(n_0+1)$'+' use counts of hits '+r'$h$'+', false alarms '+r'$f$'+', repeat opportunities '+r'$n_1$'+' and novel opportunities '+r'$n_0$'+'. The correction keeps perfect observed rates finite. Negative criterion indicates a tendency toward positive reports in this descriptive signal-detection coordinate system.\n\n'
        s+=fig(task,'literature_recognition_sdt','Recognition reanalysis of the saved native trials. Curves retain the three probe-hold conditions. Hits stay high while false recognition grows with study load; sensitivity generally decreases. These are descriptive condition estimates. There are 64 repeats and 64 novel probes per nonempty-list condition. Source-image clustering used elsewhere in the paper remains necessary for inferential comparisons across repeated photographs.')
        s+=r'\begin{longtable}{rrrrrr}\toprule Load & Hold & Hits/repeats & False/novel & $d^{\prime}$ & $c$\\\midrule\endhead'+'\n'
        for r in sorted(rows,key=lambda r:(r['load'],r['probe_hold'])):
            dp='--' if r['dprime'] is None else f"{r['dprime']:.2f}";cc='--' if r['criterion'] is None else f"{r['criterion']:.2f}"
            s+=f"{r['load']} & {r['probe_hold']} & {r['hits']}/{r['repeat_trials']} & {r['false_alarms']}/{r['novel_trials']} & {dp} & {cc}"+r'\\'+'\n'
        s+=r'\bottomrule\end{longtable}'+'\n'
        s+='The empty-list condition has no repeat opportunities, so sensitivity and criterion are undefined there. At higher loads, reduced sensitivity and a positive-report tendency coexist. A criterion-only explanation would fail to capture the observed decrease in separation under this convention; an item-similarity analysis can identify what makes the novel probes seem familiar.\n'
    # Keep the two compact analysis figures in document order before longtables.
    s=s.replace(r'\begin{figure}[htbp]',r'\begin{center}\begin{minipage}{\textwidth}')
    s=s.replace(r'\caption{',r'\captionof{figure}{')
    s=s.replace(r'\end{figure}',r'\end{minipage}\end{center}')
    return s+'\n'

def body(task,companion=False):
    from .papers import esc
    from .paper_content import CONTENT
    from .interpretation import DISCUSSION
    c=CHAPTERS[task];b=read(OUT/task/'behavior.json')
    title=CONTENT[task]['title'] if companion else 'Comparison with biological experiments and theory'
    s=r'\clearpage\section{'+esc(title)+'}\n'
    s+=r'\textbf{Scientific question.} '+esc(c['question'])+'\n\n'
    s+=f"The frozen checkpoint at update 204,289 achieved an equal-condition mean accuracy of {100*b['equal_cell_accuracy']:.2f}\\% across {b['conditions']} native conditions ({b['trials']:,} trials). The following comparisons use that same checkpoint and the measured interventions and population recordings in this paper.\n\n"
    if companion:
        s+=r'\subsection{What the model measurements establish}'+'\n'+esc(DISCUSSION[task])+'\n\n'
    for key,comparison in c['studies']:
        ref=SOURCES[key]
        s+=r'\subsection{'+esc(f"{ref['authors']} ({ref['year']}): {ref['kind']}")+'}\n'
        s+=esc(ref['finding'])+r' \cite{bio_'+key+'}\n\n'
        s+=esc(comparison)+'\n\n'
    s+=r'\subsection{Theory and the model computation}'+'\n'+esc(c['theory'])
    if THEORY.get(task):s+=r' \cite{'+','.join('bio_'+k for k in THEORY[task])+'}'
    s+='\n\n'+reanalysis(task)
    s+=r'\subsection{Agreement, difference and scientific interpretation}'+'\n'+esc(c['conclusion'])+'\n\n'
    s+=r'\subsection{The next discriminating experiment}'+'\n'+esc(c['next_test'])+'\n\n'
    s+='This last experiment is a proposed test; the results above are the completed frozen-model measurements.\n'
    return s

def enhance_source(task,source):
    from .papers import esc
    for begin,end in [(BEGIN,END),(REFBEGIN,REFEND),(RESULTBEGIN,RESULTEND),(DISCBegin,DISCEnd)]:
        source=re.sub(re.escape(begin)+r'.*?'+re.escape(end)+'\n?', '',source,flags=re.S)
    c=CHAPTERS[task]
    # Put the biological experiments before the operational task definition.
    # Their model-specific interpretations belong in Discussion, alongside the
    # measured results, rather than in a self-contained companion chapter.
    introduction=r'\subsection{Biological experiments motivating this task}'+'\n'
    introduction+=r'\textbf{Experimental question.} '+esc(c['question'])+'\n\n'
    for key,_ in c['studies']:
        ref=SOURCES[key]
        introduction+=r'\paragraph{'+esc(f"{ref['authors']} ({ref['year']})")+'} '
        introduction+=esc(ref['finding'])+r' \cite{bio_'+key+'}\n\n'
    introduction+='These experiments motivate the behavioral, spatial-selection and population measurements reported below. The discussion returns to each study to compare its findings with the measured behavior and internal states of this observer.\n'
    anchor=r'\subsection{Task definition and prediction structure}'
    assert source.count(anchor)==1,(task,'Introduction anchor')
    source=source.replace(anchor,BEGIN+'\n'+introduction+END+'\n'+anchor)
    # Saved-trial reanalyses are empirical Results, not literature commentary.
    results=reanalysis(task)
    anchor=r'\section{Discussion}'
    assert source.count(anchor)==1,(task,'Discussion anchor')
    if results:
        source=source.replace(anchor,RESULTBEGIN+'\n'+results+RESULTEND+'\n'+anchor)
    discussion=r'\subsection{Interpreting these results alongside biological experiments}'+'\n'
    for key,comparison in c['studies']:
        ref=SOURCES[key]
        discussion+=r'\subsubsection{'+esc(f"{ref['authors']} ({ref['year']})")+'}\n'
        discussion+=esc(comparison)+r' \cite{bio_'+key+'}\n\n'
    discussion+=r'\subsection{A computational account of the observed behavior}'+'\n'+esc(c['theory'])
    if THEORY.get(task):discussion+=r' \cite{'+','.join('bio_'+k for k in THEORY[task])+'}'
    discussion+='\n\n'+r'\subsection{Where this task agrees with the biological evidence}'+'\n'+esc(c['conclusion'])+'\n\n'
    discussion+=r'\subsection{A follow-up experiment motivated by these results}'+'\n'+esc(c['next_test'])+'\n'
    anchor=r'\subsection{Scope of inference}'
    assert source.count(anchor)==1,(task,'Discussion end anchor')
    source=source.replace(anchor,DISCBegin+'\n'+discussion+DISCEnd+'\n'+anchor)
    source=source.replace(r'\end{thebibliography}',REFBEGIN+'\n'+bib(references(task))+REFEND+'\n'+r'\end{thebibliography}')
    return source

INTRO=r'''
\title{Biological experiments, theory and the 19-task observer}
\author{Literature-grounded analysis of the frozen previous model}
\date{18 October 2026}\maketitle
This companion connects each task to multiple human or animal experiments, explains the proposed computation, and compares that proposal with the model's behavior, attention, state interventions and decoding results. Each chapter ends with a discriminating experiment motivated by the actual agreement or difference. The full task papers contain the complete psychometrics, time-resolved attention figures, population analyses and intervention sweeps.

The computational subject is the previous architecture at update 204,289. Its final saved state at 220,637 is a separate robustness comparison in the task papers. The fresh spatial-memory-query cloud experiment is a separate subject. The present comparison uses 248,352 completed model evaluations and adds descriptive reanalyses of existing motion-duration and recognition trials, with zero new model forwards.

\clearpage\setcounter{tocdepth}{1}\tableofcontents\clearpage
\section{A common framework for comparing the experiments}
\subsection{From sensory content to selection, memory and report}
Four questions organize the evidence. What information reaches the visual representation? Which information receives priority? What remains available after the input disappears? Which available information actually controls the report? A successful answer to one question does not settle the others. The static object-search result makes this separation especially concrete: a decoded memory can support a strong external matcher while the trained agent chooses poorly. Motion search provides a successful comparison in the same broad architecture.

Each model site contains a vector of visual attributes. Spatial attention assigns a coefficient to that vector. If $v_s$ is the vector at site $s$ and $\alpha_s$ its attention weight, a selected summary has the form $u=\sum_s\alpha_s v_s$. Here $u$ is the pooled representation, the sum runs over spatial sites, and all components of a selected site's vector contribute through the same spatial coefficient. This is a concrete shared-selection hypothesis for static and motion information. Different sensory feature transformations can feed that common selection mechanism.

\subsection{What a local attention perturbation predicts mathematically}
Let $A$ be the total attention weight inside a selected region before a uniform logit dose $\delta$. Softmax normalization gives
\[
A'=\frac{e^\delta A}{1-A+e^\delta A}.
\]
Positive $\delta$ increases the region's odds of selection and negative $\delta$ decreases them. The unperturbed regions inherit the remaining total weight. This redistribution is a property of the intervention. A behavioral benefit elsewhere additionally requires usable evidence and a readout that can exploit it. The VDA experiments measure those two levels separately: the map changes, but the full reciprocal behavioral benefit is absent at the tested weak strengths.

\subsection{Biased competition, normalization and dynamic memory}
Biased competition proposes that attention favors one of several represented stimuli. Divisive-normalization theory implements a related interaction between excitatory drive and pooled suppression. These ideas motivate testing target-alone and target-with-competitor responses under controlled attention. Our softmax has a shared spatial denominator; the sensory and recurrent layer-normalization operations have different roles. The physiological theory is best tested through response functions across stimulus and attention conditions, rather than through the name of a normalization operation. \cite{bio_competition,bio_normalization}

Memory theories also make different predictions about what remains stable. A persistent population pattern, a changing code with preserved information, and a reconstruction from another state can all support a later correct response. Cross-time decoding measures transfer of a fixed readout; state perturbation measures dependence at a particular moment. Combining them with content reconstruction connects representation and behavior. Dynamic prefrontal coding motivates this temporal approach. \cite{bio_stokes}

\subsection{Interpreting agreement at the right experimental level}
The chapters distinguish agreement in behavior, agreement in the represented variable, and agreement in causal dependency. A spatial logit dose is specified in model units; electrical stimulation and pharmacological inactivation act on biological tissue. Comparisons therefore focus on experimentally matched patterns: which location improves or worsens, which event is falsely reported, which content is recoverable, and when a perturbation matters. Model frame counts measure computational updates. A conversion to milliseconds requires a separate timing model.

Published biological studies are summarized from the verified primary sources listed in the references. The source ledger records whether the evidence inspected was an author abstract, figure legends or article text. The proposed follow-up experiments are explicitly separated from completed results. The chapters give multiple experimental anchors for every task and use theory to explain how those anchors constrain the computation.
'''

def companion():
    from .papers import PAPERS,PREAMBLE,compile_paper,esc
    assert set(CHAPTERS)==set(TASKS)
    keys=list(dict.fromkeys(['competition','normalization','stokes']+[k for t in TASKS for k in references(t)]))
    source=PREAMBLE+INTRO
    for t in TASKS:source+=body(t,True)
    source+=r'\clearpage\section{Source access and reproducibility}'+'\n'
    source+=r'The source ledger contains bibliographic metadata, a concise finding and the level of primary-source access used. Task chapters and their references are generated from neuroscience\_content.py and literature\_sources.py. New diagnostic tables and plots are reproduced by literature\_analysis.py from the saved native trials. Their input hashes are recorded in literature\_reanalysis\_provenance.json. Full native-trial and checkpoint provenance remains in the accompanying analysis package.'+'\n'
    source+=r'\begin{thebibliography}{99}'+'\n'+bib(keys)+r'\end{thebibliography}\end{document}'
    p=PAPERS/'neuroscience_comparison.tex';p.write_text(source,encoding='utf8');compile_paper(p)
    (OUT/'literature_sources.json').write_text(json.dumps(SOURCES,indent=2),encoding='utf8')
    coverage={t:dict(experiments=[k for k in references(t) if SOURCES[k]['kind']!='theory'],sources=references(t),
        question=CHAPTERS[t]['question'],new_saved_trial_analysis=(OUT/t/'literature_analysis.json').exists()) for t in TASKS}
    (OUT/'literature_coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf8')
    assert all(len(v['experiments'])>=2 for v in coverage.values())
    return p

if __name__=='__main__':
    companion()
