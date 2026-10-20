"""Recompute final statistics, assemble papers, audit and package the experiment."""
import argparse,csv,hashlib,json,os,platform,subprocess,time,zipfile,sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import fitz
from .design import *
from .statistics import behavioral,causal,global_correction,paired
from .figures import figures
from .papers import build,compile_paper,index,PAPERS,PREAMBLE,esc,pct,num,ci,fig
from .paper_content import CONTENT,REFERENCES
from .psychometric_fit import analyze as psychometric_fit
from .population_panels import units as unit_panels,query as query_analysis,weak_vda,recognition,orientation_inventory,condition_maps,memory_examples,geometry
from .timing import analyze as timing_effects
from .value_attention import analyze as value_attention
from .commitment import analyze as commitment
from .checkpoint_comparison import analyze as checkpoint_comparison

def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def attention_report(task):
    import matplotlib.pyplot as plt
    from .figures import COLORS,save
    root=OUT/task;rows=[]
    for cell in neural_cells(task):
        p=root/'attention_controls'/f'{cell}.npz';a=np.load(p);m=read(p.with_suffix('.json'));maps=a['attention'].astype(np.float32);mass=[]
        for i,meta in enumerate(m['metadata']):
            ss=sites(task,meta);mass.append(np.stack([(maps[i].reshape(maps.shape[1],maps.shape[2],100)*mask).sum(-1) for mask in ss],-1))
        mass=np.array(mass);r=dict(cell=cell,n=len(a['y']),arms=m['arms'],mean_region_mass=mass.mean(0).tolist(),central_sample=[])
        if len(a['central_sample_p']):
            for i,dose in enumerate((-4,4)):r['central_sample'].append(dict(dose=dose,**paired(a['y'],a['p'][:,0],a['central_sample_p'][:,i])))
        rows.append(r)
    write(root/'attention_control_summary.json',rows)
    r=rows[0];mass=np.array(r['mean_region_mass']);fig0,ax=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for i,arm in enumerate(r['arms']):
        if arm.startswith('global'):continue
        for j in range(3):ax[j].plot(mass[i,:,j],label=arm.replace('_evidence',''))
    for j,title in enumerate(('Designated-region mass','Alternative-region mass','Off-site mass')):
        ax[j].set(xlabel='Model frame',ylabel='Attention probability mass',title=title);ax[j].legend(fontsize=7)
    save(fig0,root/'figures/intervention_maps')
    if len(rows)>1:
        names=[v for v in rows[0]['arms'] if not v.startswith('global')]
        allocation=np.array([[np.array(r['mean_region_mass'])[r['arms'].index(name),-1,0] for name in names] for r in rows])
        causal_rows=read(root/'causal.json')['interventions']
        accuracy=np.array([[next(z['accuracy'] for z in causal_rows if z['cell']==r['cell'] and z['arm']==name) for name in names] for r in rows])
        fig0,axes=plt.subplots(1,2,figsize=(11,3+.5*len(rows)),constrained_layout=True)
        labels=[r['cell'].replace('_S8_D12_H8','').replace('_S12_D12_H12','').replace('_',' ') for r in rows]
        for ax,values,title in zip(axes,(allocation,accuracy),('Final target attention mass (32 replays)','Accuracy (128 paired trials)')):
            im=ax.imshow(values,cmap='viridis',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(len(rows)),labels,fontsize=9);ax.set_xticks(range(len(names)),[n.replace('_evidence','').replace('distractor','alternative').replace('_',' ') for n in names],rotation=35,ha='right',fontsize=9);ax.set_title(title,fontsize=11)
            for i in range(len(rows)):
                for j in range(len(names)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<.6 else '#173344')
        fig0.colorbar(im,ax=list(axes),label='Fraction',shrink=.7);save(fig0,root/'figures/intervention_allocation_by_condition')

def audit_data():
    # Include the executed recording/model/renderer dependency chain in provenance,
    # even when this reporting process itself consumes already collected arrays.
    from . import collect,attention_controls,sample_reconstruction,joint_readout,deep_analysis
    rows=[];native_trials=0;units=0
    for task,spec in TASKS.items():
        root=OUT/task
        assert (root/'collection_complete.json').exists() and (root/'statistics_complete.json').exists() and (root/'deep_analysis.json').exists()
        for c in spec['conditions']:
            a=np.load(root/'native'/f"{c['id']}.npz");assert len(a['y'])==NATIVE_N;assert a['p'].shape==(NATIVE_N,spec['classes'])
            assert np.isfinite(a['p']).all() and np.isfinite(a['attention']).all()
            assert np.allclose(a['p'].sum(-1),1,atol=1e-6);assert np.allclose(a['attention'].sum((2,3)),1,atol=1e-5)
            assert np.allclose(a['temporal'][:,-1],a['p'],atol=1e-7)
            metadata=read(root/'native'/f"{c['id']}.json");assert len(metadata)==NATIVE_N
            q=np.load(root/'native'/f"{c['id']}_latest.npz");assert np.array_equal(a['y'][:32],q['y'])
            native_trials+=len(a['y']);units+=1
        rows.append(dict(task=task,native_conditions=len(spec['conditions']),neural_conditions=len(neural_cells(task)),finite_probabilities=True,normalized_attention=True,terminal_prefix_equal=True))
    assert units==437 and native_trials==55936
    assert all(digest(p)==EXPECTED[k] for k,p in CHECKPOINTS.items())
    counts={}
    for task in TASKS:
        for category in ('native','neural','interventions','psychometric','counterfactual','attention_controls','query_switch','weak_counterfactual'):
            for path in (OUT/task/category).glob('*.npz'):
                a=np.load(path);p=a['p'];assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-5)
                counts[category]=counts.get(category,0)+int(np.prod(p.shape[:-1]))
                for name in ('central_sample_p','blank_p'):
                    if name in a and len(a[name]):counts[category]+=int(np.prod(a[name].shape[:-1]))
    source=Path(__file__).parent
    modules={}
    for name,module in sorted(sys.modules.items()):
        path=Path(getattr(module,'__file__','') or '')
        if path.is_file() and path.suffix=='.py':
            try:relative=str(path.resolve().relative_to(ROOT)).replace('\\','/')
            except ValueError:continue
            if any(part in ('.venv','venv','Tools','Results') for part in Path(relative).parts):continue
            modules[relative]=digest(path)
    write(OUT/'source_provenance.json',dict(imported_repository_sources=modules,packages={name:version(name) for name in ('torch','numpy','scipy','scikit-learn','matplotlib','Pillow','PyMuPDF','threadpoolctl')},python=platform.python_version(),time=time.time(),instructions='Repository Python dependencies used by the analysis are included under repository_sources in the archive. Place those files at their original paths in the project; run the analysis modules from the repository root. Checkpoint and natural-image dataset prerequisites remain external and are identified by the protocol.'))
    write(OUT/'data_audit.json',dict(tasks=rows,native_conditions=units,native_trials=native_trials,checkpoint_hashes_unchanged=True,
        forward_presentations_by_category=counts,total_saved_forward_presentations=sum(counts.values()),
        final_analysis_source_sha256={p.name:digest(p) for p in source.glob('*.py')},python_version=platform.python_version(),numpy_version=np.__version__,time=time.time()))

def export_csv(task):
    root=OUT/task;folder=root/'source_tables';folder.mkdir(exist_ok=True)
    data={'native':read(root/'behavior.json')['cells'],'factor_marginals':read(root/'behavior.json')['factors'],
          'chronometry':read(root/'behavior.json')['chronometry'],'psychometric':read(root/'behavior.json')['psychometric'],
          'interventions':read(root/'causal.json')['interventions'],'event_interventions':read(root/'causal.json')['subgroups'],
          'counterfactual':read(root/'causal.json')['counterfactual']}
    for name,rows in data.items():
        if not rows:continue
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (folder/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def synthesis():
    import matplotlib.pyplot as plt
    from .figures import save
    bs={t:read(OUT/t/'behavior.json') for t in TASKS};ca={t:read(OUT/t/'causal.json') for t in TASKS};pf=read(OUT/'primary_family.json')
    fig0,ax=plt.subplots(figsize=(10,7),constrained_layout=True);vals=[bs[t]['equal_cell_accuracy'] for t in TASKS];lo=[bs[t]['equal_cell_ci95'][0] for t in TASKS];hi=[bs[t]['equal_cell_ci95'][1] for t in TASKS]
    ax.barh(range(19),vals,color='#164d70');ax.errorbar(vals,range(19),xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',color='#172f40',capsize=2);ax.set_yticks(range(19),[t.replace('_',' ') for t in TASKS],fontsize=10);ax.invert_yaxis();ax.set(xlim=(0,1.04),xlabel='Equal-native-condition accuracy',title='Previous model: 55,936 fresh native trials');save(fig0,PAPERS/'suite_behavior')
    measures=['target_-4_evidence','target_+4_evidence','distractor_-4_evidence','distractor_+4_evidence','cls_gain0','spatial_gain0','comparison_gain0'];matrix=[]
    for t in TASKS:
        rr=ca[t]['interventions'];matrix.append([np.mean([r['delta_accuracy'] for r in rr if r['arm'].startswith(a)]) for a in measures])
    matrix=np.array(matrix);fig0,ax=plt.subplots(figsize=(10,8),constrained_layout=True);im=ax.imshow(matrix,cmap='RdBu_r',vmin=-.6,vmax=.6,aspect='auto');ax.set_yticks(range(19),[t.replace('_',' ') for t in TASKS],fontsize=9);ax.set_xticks(range(7),[x.replace('_evidence','').replace('_gain0',' reset') for x in measures],rotation=40,ha='right',fontsize=9);fig0.colorbar(im,ax=ax,label='Paired accuracy change');ax.set_title('Causal sensitivity averaged over selected neural conditions');save(fig0,PAPERS/'suite_causal')
    title='Shared priority, distributed memory and task-dependent readout: synthesis of nineteen computational experiments'
    s=PREAMBLE+'\\title{'+esc(title)+r'}\author{Visual Attention and Working Memory Project}\date{18 October 2026}\maketitle'+'\n'
    s+=r'\section{The experimental program}'+'\n'
    s+='This volume treats the previously trained visual model as one computational subject and applies a common experimental logic to every task. It combines fresh psychophysics, threshold-defined commitment times, spatial-priority perturbations, memory resets, external decoders and population-state analyses. The following nineteen papers contain task-specific hypotheses, methods, results, figures, native condition tables and every-frame representative attention atlases.\n\n'
    s+='The main checkpoint is 204,289, selected under the prior monitoring criterion. The final checkpoint at 220,637 supplies same-movie robustness measurements. Native behavior covers all 437 conditions with 128 trials per condition (55,936 presentations). Additional experiments examine weaker signals, causal manipulations, independent fitting/test populations and explicit VDA counterfactuals. The source data distinguish presentations from unique procedural scenes. Timing manipulations that reuse a scene are clustered together in uncertainty calculations.\n\n'
    s+=r'\section{Behavior across the entire suite}'+'\n'
    s+=r'\small\begin{longtable}{p{.43\textwidth}rrrr}\toprule Task & Conditions & Trials & Accuracy & AUC\\\midrule\endhead'+'\n'
    for t,b in bs.items():s+=f"{esc(t.replace('_',' '))} & {b['conditions']} & {b['trials']} & {pct(b['equal_cell_accuracy'])} & {num(b['equal_cell_auc'])}"+r'\\'+'\n'
    s+=r'\bottomrule\end{longtable}\normalsize'+'\n'
    s+=r'\begin{figure}[htbp]\centering\includegraphics[width=.95\textwidth,height=.7\textheight,keepaspectratio]{suite_behavior.pdf}\caption{Accuracy of the selected previous model on fresh native trials. Each task averages its native conditions equally; the number of conditions differs across tasks. Intervals preserve procedural scene clusters across timing conditions.}\end{figure}'+'\n'
    s+=r'\section{Causal organization}'+'\n'
    sig=[r for r in pf if r['holm_global']<.05];benefit=[r for r in sig if r['delta_accuracy']>0];cost=[r for r in sig if r['delta_accuracy']<0]
    s+=f"Across {len(pf)} predesignated dose-four evidence comparisons, {len(sig)} survived the cross-task Holm correction at 0.05: {len(benefit)} benefits and {len(cost)} costs. This family includes target and alternative-site stimulation and inhibition in each selected condition. The full exploratory dose and epoch sweeps are reported separately in the task papers.\n\n"
    s+=r'\begin{figure}[htbp]\centering\includegraphics[width=.98\textwidth,height=.74\textheight,keepaspectratio]{suite_causal.pdf}\caption{Causal effects of four spatial interventions and three memory resets. Values average the task-specific selected neural conditions, whose full identities are retained in the protocol. Positive values indicate improved paired accuracy; negative values indicate impairment.}\end{figure}'+'\n'
    s+=r'\subsection{Spatial memory and global memory support distinct tests}'+'\n'
    for t in ('orientation_cued','spatial_binding','image_recognition','memory_guided_search','motion_memory_search'):
        rows=ca[t]['interventions'];rr=[]
        for state in ('cls','spatial','comparison'):
            z=[r['delta_accuracy'] for r in rows if r['arm'].startswith(state+'_gain0')];rr.append(f"{state}: {100*np.mean(z):+.1f} percentage points")
        s+=esc(t.replace('_',' '))+': '+', '.join(rr)+'.\n\n'
    s+='These resets measure dependence at a specified instant, followed by normal processing of the remaining movie. A state may contain readable content and still be recoverable from other states after a reset. Conversely, a reset can disrupt subsequent computation even when a restricted linear probe reads the state poorly. Pairing causal and representational results addresses this distinction directly.\n\n'
    s+=r'\subsection{The VDA counterfactual}'+'\n'
    for t in ('vda4','vda9','vda16'):
        rr=[r for r in ca[t]['counterfactual'] if r['arm']=='cued_-4_change'];s+=esc(t.upper())+': '
        s+='; '.join(f"{r['event']} accuracy change {100*r['delta_accuracy']:+.1f} points, paired interval {ci(r['ci95'])}" for r in rr)+'.\n\n'
    s+='These trials pair a certain bottom-right cue with a controlled event at the cue, at top left, or nowhere. The event sequence deliberately tests an expectation violation. All three VDA variants are short enough that the delayed comparison memory is zero; the shared visual-attention and recurrent pathways therefore carry the measured effect. The paired catch condition distinguishes improved detection from increased positive-response tendency.\n\n'
    from .statistics import holm
    weak=[]
    for t in ('vda4','vda9','vda16'):
        weak.extend(dict(task=t,**r) for r in read(OUT/t/'weak_counterfactual_summary.json'))
    for r,p in zip(weak,holm([r['p_exact'] for r in weak])):r['holm_across_vda_followups']=p
    write(OUT/'vda_followup_family.json',weak)
    dual=[]
    for t in ('vda4','vda9','vda16'):
        for strength in (2,4,8):
            rows=[r for r in weak if r['task']==t and r['strength']==strength and r['arm']=='cued_-4_change'];target=next(r for r in rows if r['event']=='cued');foil=next(r for r in rows if r['event']=='uncued')
            if target['delta_accuracy']<0 and foil['delta_accuracy']>0 and max(target['holm_across_vda_followups'],foil['holm_across_vda_followups'])<.05:dual.append((t,strength))
    s+=r'\subsection{Testing the dual effect away from ceiling}'+'\n'
    s+=f"The exploratory weak-signal follow-up tested 2, 4 and 8 degrees in every VDA size. A joint dual-effect criterion requires both a cued detection cost and an uncued detection benefit under cue inhibition, with both paired tests surviving Holm correction across all {len(weak)} follow-up comparisons from the three VDA tasks. This criterion was met in {len(dual)} of nine task-strength settings"
    s+=(': '+', '.join(f'{t.upper()} at {v} degrees' for t,v in dual) if dual else '')+'. The complete corrected family is supplied as \\texttt{vda\\_followup\\_family.json}.\n\n'
    s+=r'\subsection{Holding the array fixed while changing the remembered target}'+'\n'
    for t in ('memory_guided_search','motion_memory_search'):
        rr=read(OUT/t/'query_switch_summary.json');s+=esc(t.replace('_',' '))+': '
        s+='; '.join(f"{esc(r['cell'])}: {pct(r['accuracy'])} balanced query accuracy, template probability gain {r['template_probability_gain']:.3f}" for r in rr)+'.\n\n'
    s+='The four target queries share identical search pixels within each scene. A query-dependent change therefore reflects the influence of the preceding sample and resulting recurrent state on the same current evidence. The accompanying attention contrast measures whether that influence also changes spatial allocation toward the newly queried location.\n\n'
    s+=r'\subsection{Usable object information can exceed the learned choice}'+'\n'
    joint=read(OUT/'memory_guided_search/joint_readout.json')
    s+='Separate linear probes reconstruct the remembered sample from middle-delay spatial memory and the current candidates from local appearance features at the first array frame. An external squared-distance comparison of those reconstructions then chooses a location. Candidate fitting preserves whole scenes in inner validation, and evaluation uses 128 independent scenes per condition. Probe pairs are fitted separately within each family/similarity condition; the original shared agent remains frozen.\n\n'
    s+=r'\small\begin{longtable}{p{.39\textwidth}rrr}\toprule Family & Agent & Joint readout & Paired Holm p\\\midrule\endhead'+'\n'
    for r in joint['results']:
        sp=r['readouts']['spatial'];s+=f"{esc(r['cell'].replace('_S8_D12_H8',''))} & {pct(r['agent_accuracy'])} & {pct(sp['accuracy'])} & {sp['paired_vs_agent']['holm_joint_family']:.4g}"+r'\\'+'\n'
    s+=r'\bottomrule\end{longtable}\normalsize'+'\n'
    s+='The two object conditions show especially large improvements: the available recurrent and visual representations support nearly perfect external matching, while the learned agent makes many errors. This localizes a major component of the failure to how available information is used for comparison and selection. Similar fractals remain more difficult, even though their joint-readout performance also improves. The scientific distinction is between storing information, representing current candidates, and implementing a successful comparison between them.\n\n'
    s+=r'\section{Neuroscientific interpretation}'+'\n'
    s+='Causal attention experiments motivate manipulating a spatial selection signal while measuring perception. Reversible superior-colliculus and frontal-eye-field inactivation can impair cued motion-change performance (Bollimunta, Bogadhi and Krauzlis, 2018). The model permits a corresponding computational question: how does changing priority at a retinotopic location alter the contribution of evidence there and elsewhere? Because the weights form a normalized distribution, this intervention intrinsically redistributes relative influence. The separate location, event and catch measurements make that redistribution behaviorally interpretable.\n\n'
    s+='The memory analyses distinguish content from use. Visual working-memory experiments motivate measuring a sample representation during stimulus absence (Miller, Erickson and Desimone, 1996). Memory-guided search additionally asks how that representation selects a new target among alternatives (Chelazzi and colleagues, 1993). In this architecture, storage can occur across the spatial and global recurrent states, while the query that selects current evidence is computed from the preceding global state. Testing preserved sample information, subsequent target allocation and memory-reset effects therefore provides three complementary observations of the storage-to-selection process.\n\n'
    s+='Population geometry supplies the temporal link. Dynamic prefrontal coding shows why a changing activity pattern can support ongoing computation (Stokes and colleagues, 2013). Temporal generalization asks whether the same fitted linear rule reads information at different times (King and Dehaene, 2014). Here, a shared training-derived projection prevents the axes from changing merely because each time point was analyzed separately. Cross-time readout, state displacement and continuous remembered-feature reconstruction then jointly characterize what persists and how its code evolves.\n\n'
    s+=r'\section{Using the papers and source data}'+'\n'
    s+='Each task paper is self-contained. The HTML index links its PDF, editable LaTeX source, behavioral data, causal data and neural summaries. CSV source tables expose individual native conditions, psychometric points, threshold statistics and intervention effects. Raw trial arrays and metadata remain in the local Results directory. The portable paper archive includes reports, figures, code and summary source data; it excludes model checkpoints and raw stimulus/video datasets.\n\n'
    s+='The analysis is conditional on one trained model. The next replication axis is independently trained models. The fixed threshold procedure supplies computational commitment times, and the intervention definitions specify priority-logit and memory-tensor operations precisely. Those definitions support reproducible comparisons to subsequent architectures without requiring a change in the task suite.\n\n'
    s+=r'\begin{thebibliography}{99}'+'\n'
    for k in ('sc','wm','it','stokes','king','wolfe'):
        title,url=REFERENCES[k];s+=r'\bibitem{'+k+'} '+esc(title)+r' \url{'+url+'}.\n'
    s+=r'\end{thebibliography}\end{document}'
    path=PAPERS/'synthesis.tex';path.write_text(s,encoding='utf8');compile_paper(path)

def audit_pdfs():
    review=OUT/'visual_review';review.mkdir(exist_ok=True);rows=[];allthumbs=[]
    for task in ['synthesis']+list(TASKS):
        path=PAPERS/f'{task}.pdf';doc=fitz.open(path);text='\n'.join(p.get_text() for p in doc)
        assert len(doc)>3 and len(text)>3000
        assert '??' not in text and '\ufffd' not in text
        bounds=[]
        for i,page in enumerate(doc):
            for block in page.get_text('dict')['blocks']:
                if block['type']!=0:continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if x0<0 or y0<0 or x1>page.rect.width+.5 or y1>page.rect.height+.5:bounds.append((i+1,span['text']))
            # Render every page; retained thumbnails make complete visual inspection possible.
            png=review/f'{task}_{i+1:03d}.png';page.get_pixmap(matrix=fitz.Matrix(.45,.45)).save(png);allthumbs.append(png)
        assert not bounds,(task,bounds)
        log=(PAPERS/f'{task}.log').read_text(errors='replace')
        warnings=[line for line in log.splitlines() if 'Overfull' in line or 'Float too large' in line or ('undefined' in line and 'undefined control' in line)]
        rows.append(dict(task=task,pages=len(doc),text_characters=len(text),bounds_errors=bounds,layout_warnings=warnings,sha256=digest(path)))
    # Contact sheets in page order, labeled by file and page.
    from PIL import Image,ImageOps,ImageDraw
    for start in range(0,len(allthumbs),30):
        batch=allthumbs[start:start+30];sheet=Image.new('RGB',(5*300,6*410),'#dfe7eb');draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(batch):
            im=Image.open(path).convert('RGB');im.thumbnail((290,380));x=(i%5)*300+(300-im.width)//2;y=(i//5)*410;sheet.paste(im,(x,y));draw.text(((i%5)*300+5,y+382),path.stem[:42],fill='#173547')
        sheet.save(review/f'contact_{start//30:02d}.jpg',quality=88)
    write(OUT/'pdf_audit.json',dict(papers=rows,total_pages=sum(r['pages'] for r in rows),all_pages_rendered=True,visual_inspection='pending',time=time.time()))

def package():
    audit=read(OUT/'pdf_audit.json')
    assert audit['visual_inspection']=='passed','Review rendered pages before packaging'
    for row in audit['papers']:
        assert not row['bounds_errors'] and not row['layout_warnings'],row['task']
        assert digest(PAPERS/f"{row['task']}.pdf")==row['sha256'],'PDF changed after review'
    doc=fitz.open();toc=[];offset=0
    for task in ['synthesis']+list(TASKS):
        source=fitz.open(PAPERS/f'{task}.pdf');toc.append([1,'Synthesis' if task=='synthesis' else CONTENT[task]['title'],offset+1]);doc.insert_pdf(source);offset+=len(source)
    doc.set_toc(toc);doc.save(PAPERS/'complete_volume.pdf',garbage=4,deflate=True);doc.close()
    index()
    files=[]
    for p in PAPERS.iterdir():
        if p.suffix in ('.pdf','.tex'):files.append(p)
    for task in TASKS:
        files.extend((OUT/task/'figures').glob('*.pdf'));files.extend((OUT/task/'figures').glob('*.png'));files.extend((OUT/task/'source_tables').glob('*.csv'))
        files.extend(OUT/task/name for name in ('behavior.json','causal.json','neural_summary.json','deep_analysis.json','attention_summary.json','attention_control_summary.json','psychometric_fit.json'))
        files.append(OUT/task/'unit_panels.json')
        if (OUT/task/'query_switch_summary.json').exists():files.append(OUT/task/'query_switch_summary.json')
        if (OUT/task/'weak_counterfactual_summary.json').exists():files.append(OUT/task/'weak_counterfactual_summary.json')
        if (OUT/task/'recognition_summary.json').exists():files.append(OUT/task/'recognition_summary.json')
        if (OUT/task/'orientation_inventory.json').exists():files.append(OUT/task/'orientation_inventory.json')
        if (OUT/task/'condition_maps.json').exists():files.append(OUT/task/'condition_maps.json')
        if (OUT/task/'timing_effects.json').exists():files.append(OUT/task/'timing_effects.json')
        if (OUT/task/'memory_examples.json').exists():files.append(OUT/task/'memory_examples.json')
        if (OUT/task/'full_crop_reconstruction.json').exists():files.append(OUT/task/'full_crop_reconstruction.json')
        if (OUT/task/'value_attention.json').exists():files.append(OUT/task/'value_attention.json')
        if (OUT/task/'commitment.json').exists():files.append(OUT/task/'commitment.json')
        if (OUT/task/'checkpoint_family_comparison.json').exists():files.append(OUT/task/'checkpoint_family_comparison.json')
        if (OUT/task/'joint_readout.json').exists():files.append(OUT/task/'joint_readout.json')
        if (OUT/task/'attention_dynamics.json').exists():files.append(OUT/task/'attention_dynamics.json')
        if (OUT/task/'literature_analysis.json').exists():files.append(OUT/task/'literature_analysis.json')
    if (OUT/'attention/index.html').exists():
        files.extend((OUT/'attention/index.html',OUT/'attention/manifest.json',OUT/'attention/verification.json'))
    for name in ('literature_sources.json','literature_coverage.json','literature_reanalysis_provenance.json','literature_verification.json','literature_metadata_audit.json'):
        if (OUT/name).exists():files.append(OUT/name)
    files.extend(OUT/name for name in ('protocol.json','data_audit.json','pdf_audit.json','primary_family.json','vda_followup_family.json','checks.json','renderer_checks.json','statistical_identity_audit.json','audited_forward.py','source_provenance.json','index.html'))
    target=OUT/'nineteen_task_papers_and_source.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            if p==OUT/'index.html':z.writestr('index.html',p.read_text(encoding='utf8').replace('<a href="nineteen_task_papers_and_source.zip">Download papers, LaTeX and source results</a>','Papers, LaTeX and source results are included in this folder'))
            else:z.write(p,p.relative_to(OUT))
        for p in Path(__file__).parent.glob('*.py'):z.write(p,'analysis_code/'+p.name)
        z.write(Path(__file__).with_name('README.md'),'analysis_code/README.md')
        z.write(Path(__file__).with_name('attention_viewer.html'),'analysis_code/attention_viewer.html')
        z.write(Path(__file__).with_name('test_attention_viewer.cjs'),'analysis_code/test_attention_viewer.cjs')
        for relative,expected in read(OUT/'source_provenance.json')['imported_repository_sources'].items():
            p=ROOT/relative;assert digest(p)==expected,(relative,'source changed after audit')
            z.write(p,'repository_sources/'+relative)
    write(OUT/'delivery.json',dict(time=time.time(),papers=19,volume=str(PAPERS/'complete_volume.pdf'),archive=str(target),archive_bytes=target.stat().st_size,archive_sha256=digest(target),full_volume_pages=offset))

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    required=['collection_complete.json','statistics_complete.json','deep_complete.json','papers_complete.json','memory_guided_search/full_crop_reconstruction.json','memory_guided_search/joint_readout.json','motion_memory_search/attention_controls/motion_A90_S12_D12_H12.json']
    deadline=time.time()+18*3600
    while not all((OUT/name).exists() for name in required):
        if not a.wait:return
        if (OUT/'failure.json').exists():raise RuntimeError('Collector failed')
        if time.time()>deadline:raise TimeoutError('Publication wait limit')
        time.sleep(20)
    for task in TASKS:causal(task)
    global_correction()
    primary_family=read(OUT/'primary_family.json')
    for task in TASKS:
        if task=='vda4':
            while not (OUT/'attention_controls_complete.json').exists():
                write(OUT/'publication_status.json',dict(phase='waiting_for_completed_vda_followups',time=time.time(),pid=os.getpid()))
                if time.time()>deadline:raise TimeoutError('Follow-up publication wait limit')
                time.sleep(20)
        write(OUT/'publication_status.json',dict(phase='final_reanalysis_and_typesetting',task=task,time=time.time(),pid=os.getpid()))
        behavioral(task);figures(task);attention_report(task);export_csv(task);fits=psychometric_fit(task);unit_panels(task);queries=query_analysis(task);weak=weak_vda(task);rec=recognition(task);inventory=orientation_inventory(task);maps=condition_maps(task);timing=timing_effects(task);examples=memory_examples(task)
        value_results=value_attention(task)
        commitment_results=commitment(task)
        geometry(task)
        # Discussions may be refined from completed exploratory results while
        # independent figures are being prepared; capture their final source later.
        from importlib import reload
        from . import interpretation,papers as paper_module
        paper_module.DISCUSSION=reload(interpretation).DISCUSSION
        path=build(task)
        # Add direct causal map verification and central-sample results before the appendix.
        source=path.read_text();rr=read(OUT/task/'attention_control_summary.json')
        insert=r'\subsection{Direct verification of attention redistribution}'+'\n'
        insert+='On 32 exactly replayed trials per selected condition, all six priority arms reproduced their original behavioral probabilities within numerical tolerance. The following figure measures the induced changes in spatial allocation at every frame. This subset supplies a direct map-level manipulation check.\n'
        insert+=fig(task,'intervention_maps','Matched map-level verification. Mean designated, alternative and off-site attention mass under sham and four dose-four evidence interventions. These are the first 32 exact movies from the primary intervention experiment, not a new behavioral sample.')
        if len(rr)>1:insert+=fig(task,'intervention_allocation_by_condition','Condition-resolved manipulation checks. The left panel measures final target-region allocation on 32 exact replays; the right measures accuracy on all 128 paired intervention trials. All displayed spatial doses have magnitude four and act during evidence frames. Rows preserve each selected condition, making redistribution and behavioral rescue separately visible.')
        for r in rr:
            for c in r['central_sample']:insert+=f"For {esc(r['cell'])}, a sample-centered encoding dose of {c['dose']:+g} changed paired accuracy by {100*c['delta_accuracy']:+.1f} percentage points on 32 trials (interval {ci(c['ci95'])}). This manipulation addresses the central sample location, separately from the future array-target location.\n\n"
        insert+=r'\subsection{Held-out population response profiles}'+'\n'
        insert+='The 64 CLS coordinates are ordered by the class with the largest final mean response on fitting trials, with within-class order determined by between-class response variance. The displayed means use independent test trials. Four example coordinates are selected by the largest between-class variance on fitting trials; their held-out trajectories are shown with 1.96 standard-error bands. Activation coordinates use the fitting-set mean and standard deviation across trials and time.\n\n'
        insert+=fig(task,'unit_population','All 64 global-memory coordinates, shown separately for each correct-answer class. Unit ordering and standardization are learned from the fitting population; displayed activity comes from held-out trials. Colors represent activation, not firing rate.')
        insert+=fig(task,'unit_examples','Four population coordinates selected on fitting trials and evaluated on independent trials. Curves separate response classes; shaded bands describe uncertainty in the mean activation.')
        insert+=r'\subsection{Correct answers, actual choices and spatial instructions}'+'\n'
        insert+='The correct-answer decoder asks what the stimulus requires; a separate decoder predicts the model\'s actual terminal choice. They use the same independent fitting/test split and capacity-matched regional features, but distinct target labels. Errors allow these two kinds of information to separate. A location decoder uses the cue index when available, otherwise the instructed or target location. Balanced accuracy averages recall over the labels represented in the held-out population. When every response is correct, answer and choice labels coincide.\n\n'
        insert+=r'\begin{center}\small\begin{tabular}{lrrr}\toprule Region & Correct answer & Actual choice & Cue/target location\\\midrule'+'\n'
        for name,r in read(OUT/task/'neural_summary.json')[0]['regions'].items():
            if r.get('constant_zero'):continue
            insert+=f"{esc(name)} & {pct(r['final_balanced_accuracy'])} & {pct(r.get('choice_diagonal',[None])[-1])} & {pct(r.get('location_diagonal',[None])[-1])}"+r'\\'+'\n'
        insert+=r'\bottomrule\end{tabular}\normalsize\end{center}'+'\n'
        insert+=r'\subsection{Class geometry in the full recurrent states}'+'\n'
        insert+=r'For classes $a$ and $b$, let $d^{\rm fit}_j$ and $d^{\rm test}_j$ be their mean-activation differences in coordinate $j$, estimated from independent fitting and test populations and divided by the fitting-set standard deviation at that frame (with a 0.05 floor). The cross-validated distance is $D_{ab}=M^{-1}\sum_{j=1}^{M}d^{\rm fit}_j d^{\rm test}_j$, where $M$ is 64 for global memory and 6,400 for spatial memory. Independent differences remove the positive within-sample noise contribution present when one squared mean difference is used. Negative estimates are retained. Class codes follow the task response labels. This analysis uses every recorded recurrent coordinate and complements the low-dimensional trajectory visualization.'+'\n\n'
        insert+=fig(task,'full_state_geometry','Cross-validated class geometry of the complete recurrent states. Left: mean off-diagonal distance over time, with the memory-test frame marked. Middle and right: class-pair distance matrices at the memory-test and final frames. Color scales are shared between the two matrices within each population. Zero is the null separation baseline; negative estimates remain visible.')
        for name in maps:
            insert+=fig(task,name,'Condition-resolved attention at four temporal landmarks. Memory-search and pop-out rows average trials with top-left targets. Image-recognition rows average seen-probe trials. Each map is shown in the original image layout with a common probability scale across conditions; the row label gives the number of averaged trials. The memory-test frame is the middle delay, or the frame before evidence when a blank delay is absent.')
        for name in examples:
            insert+=fig(task,name,'Concrete search examples from all six family/similarity conditions. Each row uses the first error in the selected condition, or the first trial if all 128 were correct. The enlarged sample crop is followed by the exact array, the same trial\'s attention overlay, and its final attention map. Blue marks the target and coral the chosen location. Attention scales are shown separately for each example. These illustrate failure structure; the factorial results quantify its frequency.')
        if task=='memory_guided_search':
            checkpoint_rows=checkpoint_comparison()
            insert+=r'\subsection{Checkpoint robustness within each stimulus family}'+'\n'
            insert+='The final 220,637 checkpoint is compared with the selected 204,289 checkpoint on the same first 32 native scenes in each family. Each scene is evaluated across all 48 timing settings in that family. Differences are averaged over timing within scene, and intervals resample the 32 paired scenes 3,000 times. These rows therefore refer to a matched subset of the native trials, separately from the independent neural-recording population below.\n\n'
            insert+=r'\small\begin{longtable}{p{.29\textwidth}rrr}\toprule Family & Selected & Final & Change interval (pp)\\\midrule\endhead'+'\n'
            for r in checkpoint_rows:insert+=f"{esc(r['family'])} & {pct(r['selected_accuracy'])} & {pct(r['final_accuracy'])} & [{100*r['ci95'][0]:+.1f}, {100*r['ci95'][1]:+.1f}]"+r'\\'+'\n'
            insert+=r'\bottomrule\end{longtable}\normalsize'+'\n'
            reconstruction=read(OUT/task/'full_crop_reconstruction.json');insert+=r'\subsection{Reconstructing the complete remembered object}'+'\n'
            insert+='The full-crop analysis predicts all 28 by 28 RGB sample pixels, giving 2,352 continuous outputs. Targets are regenerated from the exact neural-trial seeds and their stored raster hashes are verified. Full-coordinate ridge mappings use 192 fitting trials and 128 independent test trials per condition; the penalty is chosen from 0.001, 0.01, 0.1, 1 and 10 using an inner 75/25 fitting split. Separate mappings at encoding, middle delay, first search and final frames compare appearance, global memory and spatial memory. R-squared uses the fitting-mean sample as the baseline. The first held-out example in each condition is shown; images are clipped only for display, while scores use the unbounded predictions.\n\n'
            for r in reconstruction['results']:insert+=f"For {esc(r['cell'])}, middle-delay full-crop R-squared was {r['regions']['cls']['mid_delay']['r2']:.3f} from CLS memory and {r['regions']['spatial']['mid_delay']['r2']:.3f} from spatial memory. "
            insert+='\n\n'+fig(task,'full_crop_reconstruction','Full-resolution object content reconstructed from the original coordinate widths. Rows preserve family and similarity, and columns preserve temporal stage. Blank-frame appearance provides a comparison with stored content in recurrent states.')
            for name in ('reconstructed_examples_00','reconstructed_examples_03'):insert+=fig(task,name,'The first held-out sample and its linear reconstruction from middle-delay global and spatial memory. These examples were fixed by trial index, independently of reconstruction quality.')
            insert+=r'\subsection{Can the decoded memory identify the matching object?}'+'\n'
            insert+='An external readout compares each reconstructed sample with the four actual candidate rasters and chooses the candidate with the smallest mean squared pixel difference. The reconstruction mapping sees only the recorded state; candidate rasters enter the subsequent matching calculation. The same 128 held-out scenes supply the original agent choices and the external choices. Paired comparisons use exact discordance tests and Holm correction over all 72 region-by-time-by-condition comparisons. The table emphasizes the middle blank delay, before array presentation, so this test isolates information retained from the sample.\n\n'
            insert+=r'\small\begin{longtable}{p{.30\textwidth}rrrr}\toprule Condition & Agent & CLS match & Spatial match & Spatial Holm p\\\midrule\endhead'+'\n'
            for r in reconstruction['results']:
                c=r['regions']['cls']['mid_delay'];sp=r['regions']['spatial']['mid_delay']
                insert+=f"{esc(r['cell'].replace('_S8_D12_H8',''))} & {pct(r['agent_accuracy'])} & {pct(c['candidate_matching_accuracy'])} & {pct(sp['candidate_matching_accuracy'])} & {sp['paired_vs_agent']['holm_reconstruction_family']:.4g}"+r'\\'+'\n'
            insert+=r'\bottomrule\end{longtable}\normalsize'+'\n'
            insert+='These results separate retained content from its use by the learned decision computation. For object arrays and distinct fractals, an external comparison of decoded spatial memory with visible candidates substantially exceeds the original agent\'s performance. Thus those failures coexist with usable sample information in spatial memory. Similar fractals remain difficult under both readouts, indicating a further representational or matching limitation for this condition. The external readout has access to an explicit pixel-distance comparison; its performance establishes an available information route, while implementing that route inside the agent is a separate architectural experiment.\n\n'
            insert+=fig(task,'reconstructed_template_matching','Original agent choices versus external candidate matching from reconstructed middle-delay content. Dashed line: 25 percent chance; error bars: Wilson 95 percent intervals. All three readouts use the same 128 held-out scenes in each condition; the external readouts reconstruct the sample before seeing the search array.')
            joint=read(OUT/task/'joint_readout.json')
            insert+=r'\subsection{Matching a decoded memory to decoded visual candidates}'+'\n'
            insert+='A further exploratory readout checks whether the current array encoder also preserves the needed information. For each candidate, its centered three-by-three grid neighborhood provides 288 appearance coordinates (nine sites times 32 channels) at the first array frame. Within each family/similarity condition, one linear ridge decoder shared across the four candidate locations predicts the 2,352 candidate RGB values from those coordinates. A separate probe pair is fitted for each of the six conditions. Four candidates in each of 192 fitting scenes give 768 fitting examples. Penalty selection splits whole scenes 75/25, keeping their four candidates together. The 128 test scenes remain independent, and the fitting objective uses raster reconstruction rather than behavioral labels.\n\n'
            insert+='The separately fitted middle-delay sample decoder and the current-array decoder then supply two decoded pixel representations. Their squared-distance comparison chooses the most similar candidate. This joint readout consumes internal representations at test time. An exact-sample plus decoded-array control separately measures the candidate decoder\'s usefulness. The six spatial-memory joint readouts are compared with original agent choices on the same scenes, with Holm correction across those six paired tests. This follow-up was chosen after observing successful memory reconstruction.\n\n'
            insert+=r'\small\begin{longtable}{p{.26\textwidth}rrrr}\toprule Family & Candidate $R^2$ & Exact sample & Joint spatial & Holm p\\\midrule\endhead'+'\n'
            for r in joint['results']:
                sp=r['readouts']['spatial']
                insert+=f"{esc(r['cell'].replace('_S8_D12_H8',''))} & {r['candidate_decoder']['r2']:.3f} & {pct(r['readouts']['raw_sample']['accuracy'])} & {pct(sp['accuracy'])} & {sp['paired_vs_agent']['holm_joint_family']:.4g}"+r'\\'+'\n'
            insert+=r'\bottomrule\end{longtable}\normalsize'+'\n'
            insert+=fig(task,'joint_readout','External matching from decoded memory and decoded current-array features. Exact sample plus decoded array isolates the candidate readout. CLS and spatial joint readouts use decoded templates from the blank delay and decoded candidates from the first array frame. Agent performance is measured on the same 128 independent scenes; error bars are Wilson 95 percent intervals and chance is 25 percent.')
        if inventory:
            insert+=r'\subsection{Remembered orientation inventory at all four locations}'+'\n'
            insert+='A separate content analysis reconstructs all four original sample orientations at every frame. Each orientation angle is encoded as its doubled-angle cosine and sine, respecting the 180-degree periodicity of a grating. The same training-only 32-component representation and ridge penalty used in the main analysis predict the eight continuous target values. Each location has a separate held-out R-squared relative to its fitting-mean predictor. This directly measures location-bound sample content. In the retrospective-cue task, the identity of the eventual target is revealed only after the delay, so all-location reconstruction is the appropriate pre-cue storage measurement.\n\n'
            insert+=fig(task,'orientation_inventory','The original orientation at each of the four sample locations decoded from global and spatial memory. Curves track continuous sample content rather than the later correct-answer label. Zero is the fitting-mean baseline.')
        if queries:
            insert+=r'\subsection{Identical-array, changed-memory experiment}'+'\n'
            insert+='For each selected condition, 64 new scenes were each replayed four times. Only the sampled candidate changed; all search-array pixels were verified identical across the four movies. Every location served once as the correct answer within each scene. An additional replay replaced the sample with uniform midpoint gray. Intervals resample whole scenes, keeping their four queries together; all-perfect or all-zero accuracy boundaries use a Wilson interval on the 64 scene units. The template probability gain subtracts mean off-diagonal report probability from mean diagonal probability; the attention gain applies the same diagonal-minus-off-diagonal comparison to final spatial masses. These contrasts measure how a change in the remembered sample changes the response to a fixed array.\n\n'
            for r in queries:
                insert+=f"For {esc(r['cell'])}, accuracy was {pct(r['accuracy'])} (scene-bootstrap interval {ci(r['accuracy_ci95'])}); {pct(r['all_four_correct'])} of scenes had all four queries answered correctly. Choice varied across queries in {pct(r['choice_changes'])} of scenes. Template probability gain was {r['template_probability_gain']:.3f} (95 percent interval [{r['template_probability_gain_ci95'][0]:.3f}, {r['template_probability_gain_ci95'][1]:.3f}]) and final attention gain was {r['template_attention_gain']:.3f} (interval [{r['template_attention_gain_ci95'][0]:.3f}, {r['template_attention_gain_ci95'][1]:.3f}]). The sample-blank response is fixed within each array and therefore scores exactly 25 percent across the four balanced candidate labels.\n\n"
            insert+=fig(task,'query_switch','Causal template dependence with array pixels held fixed. Rows identify the candidate used as the remembered sample; columns identify report location. A diagonal pattern indicates that memory changes the selected answer appropriately. Each panel averages 64 matched scenes.')
        if weak:
            insert+=r'\subsection{Weaker-signal follow-up to test the improvement prediction}'+'\n'
            insert+='After the 18-degree VDA4 counterfactual showed high uncued baseline detection, an exploratory follow-up tested 2-, 4- and 8-degree changes, plus catches, in all three VDA sizes. Each event-strength cell contains 128 new matched nuisance draws per arm. The bottom-right cue remains certain and perturbations occur during the change frames. Holm adjustment below covers the entire follow-up family within this task. These tests address whether the apparent absence of an uncued benefit at strong signals reflects limited headroom.\n\n'
            insert+=r'\begin{center}\begin{tabular}{lrrrr}\toprule Event & Degrees & Sham accuracy & Cue-inhibition change & Holm p\\\midrule'+'\n'
            for r in weak:
                if r['arm']!='cued_-4_change':continue
                base=next(z for z in weak if z['event']==r['event'] and z['strength']==r['strength'] and z['arm']=='sham')
                insert+=f"{esc(r['event'])} & {r['strength']} & {pct(base['accuracy'])} & {100*r['delta_accuracy']:+.1f} pp & {r['holm_followup']:.4g}"+r'\\'+'\n'
            insert+=r'\bottomrule\end{tabular}\end{center}'+'\n'
            insert+=fig(task,'weak_vda','Exploratory weak-change counterfactual. Cued and uncued hit fractions are shown across three magnitudes; the catch panel shows false reports without a physical change. Error bars are Wilson 95 percent intervals. All interventions reuse identical movies within each cell.')
        if value_results:
            insert+=r'\subsection{Probability cues, reward value and prechange allocation}'+'\n'
            insert+='The native task varies displayed change-location probability and cue color, which specifies trial value. We stratify the original 384 trials by displayed probability, event location and value. Cued and uncued change reports are hits; reports on catch trials are false alarms. Prechange cue mass is measured at frame 4, before the two test frames, and is expressed relative to a uniform distribution over the 100 sites. Probability intervals use Wilson bounds; attention intervals resample native trials 3,000 times. These native stratifications characterize how the learned observer treats the task instructions, with the paired interventions supplying the causal test.\n\n'
            insert+=f"The model obtained {pct(value_results['reward_fraction'])} of the available native value, calculated as the sum of value on correctly answered trials divided by total offered value. "
            for r in value_results['by_value']:insert+=f"At value {r['value']:g}, accuracy was {pct(r['accuracy'])} across {r['n']} trials. "
            insert+='\n\n'+fig(task,'value_attention','Native probability and value stratification. Left: cued hits, uncued hits and catch false reports. Middle: cue-region attention above uniform at the last prechange sample frame, separated by trial value. Right: accuracy by offered value. Sparse event-probability groups have wide sampling intervals; their sample counts are supplied in the source table.')
        source=source.replace(r'\clearpage\section{Discussion}',insert+r'\clearpage\section{Discussion}')
        passage=r'\subsection{Does an early confident answer persist?}'+'\n'
        passage+='A confidence threshold defines a hypothetical commitment rule. We test that rule directly by comparing its first post-evidence choice with the unchanged terminal choice on the same movie. A rescued error is an incorrect early commitment followed by a correct final answer; a spoiled answer changes from correct to incorrect. For four-class tasks, a revision can also exchange one incorrect alternative for another. These measurements determine whether terminal accuracy transfers to early stopping.\n\n'
        r=next(r for r in commitment_results['pooled'] if r['threshold']==.8)
        passage+=f"At threshold 0.8 across all native presentations, {r['crossings']:,} of {r['n']:,} trials crossed. Of those, {r['changed_choice']:,} changed their answer later, {r['rescued_error']:,} corrected an early error, and {r['spoiled_correct']:,} lost an initially correct answer. These pooled counts are descriptive; the representative condition is shown separately below.\n\n"
        passage+=fig(task,'commitment','Validity of first-threshold commitment in the representative native condition. Early and terminal accuracy are compared on exactly the same crossing trials at each threshold; the second panel separates kinds of later revision. Trials that never cross remain censored in the preceding analysis.')
        passage+='We additionally measure retrospective choice-settling latency: the first post-evidence frame from which the argmax remains equal to its terminal value for the rest of the movie. Frame counts start at one for the first available evidence frame. This measure uses future observations to locate the final revision. It describes the dynamics of the existing computation; it is distinct from the causal first-threshold stopping rule. Terminally correct and incorrect trajectories are kept separate.\n\n'
        passage+=fig(task,'choice_settling','Cumulative distribution of retrospective choice-settling times in the representative condition. A choice is settled once all subsequent prefix classifications agree with the terminal classification. Correct and incorrect terminal decisions can both settle, and are plotted separately.')
        source=source.replace(r'\subsection{Allocation of shared spatial attention}',passage+r'\subsection{Allocation of shared spatial attention}')
        primary=[r for r in primary_family if r['task']==task]
        passage=r'\clearpage\subsection{Predesignated spatial comparisons across the full series}'+'\n'
        passage+=f"The following dose-four evidence comparisons belong to one family of {len(primary_family)} tests across all nineteen tasks and selected conditions. The Holm-adjusted values therefore account for the complete primary spatial family. Positive changes indicate benefit and negative changes indicate impairment relative to the same-movie sham.\n\n"
        passage+=r'\small\begin{longtable}{p{.29\textwidth}p{.32\textwidth}rr}\toprule Condition & Intervention & Change (pp) & Global Holm p\\\midrule\endhead'+'\n'
        for r in primary:passage+=f"{esc(r['cell'].replace('_S8_D12_H8',''))} & {esc(r['arm'].replace('_evidence',''))} & {100*r['delta_accuracy']:+.1f} & {r['holm_global']:.4g}"+r'\\'+'\n'
        passage+=r'\bottomrule\end{longtable}\normalsize'+'\n'
        source=source.replace(r'\subsection{Readable information across representation levels}',passage+r'\subsection{Readable information across representation levels}')
        if timing:
            passage=r'\subsection{Matched-scene effects of sample, delay and search time}'+'\n'
            passage+='The native counter-based generator reuses scene identity and target assignment across timing conditions. We compare the longest and shortest values of each timing factor while matching the other factors, then average those paired differences within scene. Static arrays remain pixel-identical; motion arrays retain the same direction and layout seeds and shared frame prefix as duration increases. Bootstrap intervals resample the 128 scene identities. A two-sided sign-flip test uses 9,999 random sign patterns on the scene-level differences, followed by Holm correction over this task\'s timing comparisons. These exploratory contrasts distinguish changes in encoding opportunity, retention duration and search opportunity.\n\n'
            passage+=r'\small\begin{longtable}{p{.30\textwidth}lrrr}\toprule Group & Factor & Short/long & Change (pp) & Holm p\\\midrule\endhead'+'\n'
            for r in timing:passage+=f"{esc(r['group'])} & {esc(r['factor'].replace('_frames',''))} & {r['low']}/{r['high']} & {100*r['delta_accuracy']:+.1f} & {r['holm_p']:.4g}"+r'\\'+'\n'
            passage+=r'\bottomrule\end{longtable}\normalsize'+'\n'
            passage+=fig(task,'timing_effects','Paired timing effects with scene identity preserved. Each value is the longest-duration accuracy minus the shortest-duration accuracy, averaged over matched remaining factors. Intervals resample complete scenes.')
            source=source.replace(r'\subsection{Time course and commitment thresholds}',passage+r'\subsection{Time course and commitment thresholds}')
        if rec:
            passage=r'\subsection{Load, false recognition and serial position}'+'\n'
            passage+='Zero-study trials are negative controls: every probe is novel. The nonzero study loads test recognition after four, twelve or twenty-four images. Hit and false-alarm curves separate missed stored items from false familiarity. Serial-position quartiles run from the earliest to latest study items, pooled across probe-hold durations. Intervals resample probe-image identities across those conditions.\n\n'
            for load in (4,12,24):
                hit=next(z for z in rec['load_curves'] if z['load']==load and z['report']=='hit');fa=next(z for z in rec['load_curves'] if z['load']==load and z['report']=='false_alarm')
                passage+=f"At load {load}, the hit rate was {pct(hit['rate'])} and the false-alarm rate was {pct(fa['rate'])}. "
            passage+='The load-related error pattern can therefore be assessed as a change in recognition sensitivity and report tendency, alongside the population-memory measurements.\n\n'
            passage+=fig(task,'recognition_errors','Recognition error structure. Left: hit and false-alarm fractions by study load. Right: hit fractions by study-position quartile. Intervals preserve repeated probe photographs across probe-hold conditions.')
            source=source.replace(r'\subsection{Time course and commitment thresholds}',passage+r'\subsection{Time course and commitment thresholds}')
        if fits:
            fittext=r'\subsection{Exploratory psychometric threshold fit}'+'\n'
            fittext+=r'The native-label accuracy sweep is summarized by $P(x)=\gamma+(1-\gamma-\lambda)[1-\exp(-(x/a)^b)]$. Here $x$ is signal strength, $\gamma$ is reciprocal class count, $a>0$ sets scale, $b>0$ sets shape, and $\lambda\in[0,0.2]$ is the fitted lapse fraction. Binomial likelihood is minimized over the empirical levels. The midpoint threshold $x_{50}=a(\log 2)^{1/b}$ lies halfway between the guessing floor and fitted upper asymptote. This is an exploratory descriptive addition after observing the empirical curves. Three hundred bootstrap resamples retain the same nuisance-trial index across signal levels. Event-conditioned detection reports remain empirical. The approach follows the fitting and uncertainty principles discussed by Wichmann and Hill (2001; \url{https://pubmed.ncbi.nlm.nih.gov/11800458/}, \url{https://pubmed.ncbi.nlm.nih.gov/11800459/}).'+'\n\n'
            for r in fits:
                bounds=r['threshold_ci95'];bound_text=f"[{bounds[0]:.4g}, {bounds[1]:.4g}]" if bounds is not None else 'unavailable because no bootstrap fit converged'
                fittext+=f"The fitted midpoint was {r['threshold']:.4g} task units (bootstrap interval {bound_text}), with slope {r['slope_at_threshold']:.4g} probability per task unit and lapse {pct(r['lapse'])}. The largest absolute discrepancy from an observed point was {r['max_absolute_residual']:.3f}. "
                if r['boundary_parameter']:fittext+='A fitted parameter reached its allowed boundary; the empirical curve remains the primary result. '
                if not r['threshold_within_tested_range']:fittext+='The fitted threshold lies outside the sampled range and is an extrapolation. '
                fittext+='\n\n'
            fittext+=fig(task,'psychometric_fit','Empirical points and descriptive Weibull function. The vertical line marks the midpoint between guessing and the fitted upper asymptote. Threshold units match the stimulus-strength axis.')
            if task=='contour':
                fittext=fittext.replace(r'[1-\exp(-(x/a)^b)]',r'\exp(-(x/a)^b)').replace('Here $x$ is signal strength','Here $x$ is contour orientation jitter in degrees')
                fittext+='Increasing jitter weakens contour alignment, so this fitted function decreases toward chance and its signed slope is negative.\n\n'
            source=source.replace(r'\subsection{Time course and commitment thresholds}',fittext+r'\subsection{Time course and commitment thresholds}')
        source=source.replace('Intervals use 3,000 paired resamples;', 'Intervals use 3,000 paired resamples; zero-discordance outcomes use a conservative exact bound on the discordance probability, and all-gain/all-loss boundaries use Wilson intervals;')
        source=source.replace('retention-boundary frame','memory-test frame').replace('at the retention boundary','at the memory-test frame (the middle blank-delay frame, or the frame immediately before evidence when no blank delay exists)').replace('Full retention AUC','Memory-test AUC')
        assert source.count(r'\section{Discussion}')==1
        for heading in ('Does an early confident answer persist?','Direct verification of attention redistribution','Predesignated spatial comparisons across the full series'):
            assert r'\subsection{'+heading+'}' in source,(task,heading)
            assert source.index(heading)<source.index(r'\section{Discussion}'),(task,'Results must precede discussion')
        if task=='memory_guided_search':assert 'Can the decoded memory identify the matching object?' in source
        if fits:assert 'Exploratory psychometric threshold fit' in source
        path.write_text(source,encoding='utf8');compile_paper(path)
    audit_data();global_correction();synthesis();audit_pdfs()
    write(OUT/'publication_status.json',dict(phase='awaiting_final_visual_review',time=time.time(),pid=os.getpid()))

if __name__=='__main__':main()
