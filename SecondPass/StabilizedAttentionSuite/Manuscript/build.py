"""Build the local manuscript from authored sections and verified source records."""
from pathlib import Path
import json,re,html,hashlib,datetime
from bs4 import BeautifulSoup
from results_data import build_results
from task_prose import direct_task_prose
ROOT=Path(__file__).parent
esc=html.escape

MANUAL=[
('morgan2025',2025,'Jonathan Morgan; Badr Albanna; James P. Herman','A recurrent vision transformer shows signatures of primate visual attention','https://arxiv.org/html/2502.10955v1','Primary full-text preprint v1 inspected; local research/morgan2025.html'),
('lucas1981',1981,'Bruce D. Lucas; Takeo Kanade','An Iterative Image Registration Technique with an Application to Stereo Vision','https://idl.uw.edu/living-papers-paper/lucas-kanade/','Original paper reproduced by University of Washington; method and metadata inspected'),
('barron1994',1994,'John L. Barron; David J. Fleet; Steven S. Beauchemin','Performance of Optical Flow Techniques','https://www.cs.toronto.edu/~fleet/research/Papers/ijcv-94.pdf','Author-hosted primary paper; metadata and abstract checked'),
('black1996',1996,'Michael J. Black; P. Anandan','The Robust Estimation of Multiple Motions: Parametric and Piecewise-Smooth Flow Fields','https://www.csd.uoc.gr/~hy576/bibliography/RobustMotion.pdf','University-hosted primary paper; bibliographic identity checked'),
('brox2004',2004,'Thomas Brox; Andrés Bruhn; Nils Papenberg; Joachim Weickert','High Accuracy Optical Flow Estimation Based on a Theory for Warping','https://lmb.informatik.uni-freiburg.de/Publications/2004/Bro04a/','Author-laboratory primary publication record and abstract inspected'),
('farneback2003',2003,'Gunnar Farnebäck','Two-Frame Motion Estimation Based on Polynomial Expansion','https://www.diva-portal.org/smash/get/diva2%3A273847/FULLTEXT01.pdf','Primary paper in institutional repository; title, author and method checked'),
('sun2014',2014,'Deqing Sun; Stefan Roth; Michael J. Black','A Quantitative Analysis of Current Practices in Optical Flow Estimation and the Principles Behind Them','https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Sun2014IJCV.pdf','Primary paper; bibliographic identity checked against author publication list'),
('ilg2017',2017,'Eddy Ilg; Nikolaus Mayer; Tonmoy Saikia; Margret Keuper; Alexey Dosovitskiy; Thomas Brox','FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks','https://openaccess.thecvf.com/content_cvpr_2017/html/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.html','Official CVPR proceedings metadata and primary preprint abstract checked'),
('sun2018',2018,'Deqing Sun; Xiaodong Yang; Ming-Yu Liu; Jan Kautz','PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume','https://arxiv.org/abs/1709.02371','Primary preprint abstract inspected; CVPR2018 version, first posted2017'),
('reichardt1961',1961,'Werner Reichardt','Autocorrelation, a principle for evaluation of sensory information by the central nervous system','https://pure.mpg.de/view/item_1795261','Max Planck primary institutional bibliographic record checked')]

RATIONALE={
'motion_direction':('Correspondence without selection','The shared visual front end must identify a single transition despite dot replacement. This task tests the motion population on a single transition.'),
'orientation':('Appearance comparison without spatial competition','This isolates ordered orientation comparison before adding a cue or long delay. Independent phase makes raw pixel subtraction an unreliable substitute for orientation. It tests whether appearance survives fusion with the much wider motion representation.'),
'contrast':('Preserve amplitude information','The model must retain an intensity-modulation difference across two observations. Normalization can help optimization while discarding task-relevant scale if placed carelessly; the raw local statistics explicitly preserve intensity information for the learned readout.'),
'spatial_frequency':('Compare spatial scale','A ratio of frequencies must be distinguished from a change in position or carrier phase. This probes appearance features across spatial scales.'),
'chromatic_increment':('Keep information that grayscale discards','This task compares RGB-axis changes under its defined numerical luminance constraint, directly motivating the RGB appearance route.'),
'contour':('Bind orientation to spatial arrangement','An inventory of orientations is insufficient because the structured and shuffled scenes contain the same orientation values. Relationships among nearby elements matter. This challenges pooling that loses spatial arrangement and motivates testing interactions beyond a single selected patch.'),
'natural_spectrum':('Generalize a visual statistic across photographs','The observer compares frequency balance in two versions of one photograph, with shared RMS scaling. This probes spectral appearance processing on photographs drawn from BSDS.'),
'orientation_ring':('Add spatial relevance to an appearance judgment','A ring specifies which local orientation comparison should control the answer while other locations change independently. The answer requires selecting the cued appearance comparison.'),
'orientation_cued':('Combine location, remembered rule, and comparison','The cue specifies both where to look and which sign of change counts as positive. The observer must retain the rule across a delay. A generic change detector or a cue-blind orientation classifier is insufficient, even when it sees the relevant rotation.'),
'motion_duration_cued':('Integrate evidence, then preserve a decision','The target direction with the largest total count across eight transitions must survive a later delay. The last observed direction can disagree with the answer. This separates evidence accumulation from retention and challenges a short recent-motion summary.'),
'krauzlis_cued_motion':('Separate physical change from relevant change','The cue disappears before motion begins. Foil-only changes are physically real but behaviorally negative. Correct performance requires maintaining spatial relevance and comparing local motion populations without an event-time marker.'),
'spatial_binding':('Remember which feature belonged where','Exactly two items exchange orientations in both classes, and the queried location appears only after retention. The judgment requires retaining the association between each feature and its location.'),
'image_recognition':('Retain a list of changing visual appearances','Each study image can be entirely different. The answer concerns exact membership of the probe in the recent list. Load and repeated-probe duration separately challenge storage and retrieval in a fixed-size recurrent representation.')}

def main():
    refs={}
    for p in (ROOT/'research/reference_records').glob('*.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('match',0)>.9:refs[d['id']]=d
    for d in json.loads((ROOT/'task-atlas/sources.json').read_text(encoding='utf-8')):
        if d.get('year') and (d.get('doi') or d['id']=='sensory-krauskopf1992'):refs[d['id']]=d
    for key,year,authors,title,url,verification in MANUAL:
        refs[key]=dict(id=key,year=year,authors=authors,title=title,url=url,verification=verification)
    refs['carandini2012']['year']=2012 # journal issue year; online publication 2011
    tasks=json.loads((ROOT/'task-atlas/tasks.json').read_text(encoding='utf-8'))
    catalog=json.loads((ROOT/'task-atlas/manifest.json').read_text(encoding='utf-8'))['catalog']
    specs={t['id']:t for t in catalog['tasks']}
    task_html=[]
    for n,t in enumerate(tasks,1):
        key=t['id'];spec=specs[key];head,why=RATIONALE[key]
        neuroscience=direct_task_prose(t['sections']['neuroscience'][0])
        body=''.join('<p>'+esc(direct_task_prose(x))+'</p>' for x in t['sections']['sequence']+t['sections']['rules'])
        task_html.append(f'<article class="task-method" id="task-{key}"><div class="task-number">{n:02}</div><div><p class="eyebrow">{esc(head)}</p><h4>{esc(t["title"])}</h4><p>{esc(t["question"])}</p><p>{esc(why)}</p><div class="condition-chips">'+''.join(f'<span>{esc(c["id"])}</span>' for c in spec['conditions'])+f'<span>{spec["classes"]} choices</span></div><details><summary>Exact sequence, response rule, and neuroscience source</summary>{body}<p>{esc(neuroscience)}</p></details></div></article>')
    section_names=['introduction','tasks','vda','flow','architecture','memory','neuroscience','experiments','results','independent','reading']
    content='\n'.join((ROOT/'sections'/f'{name}.html').read_text(encoding='utf-8') for name in section_names)
    content=content.replace('<!-- PIXELS_TUTORIAL -->',(ROOT/'sections/pixels_tutorial.html').read_text(encoding='utf-8'))
    content=content.replace('<div id="task-methods"></div>',''.join(task_html))
    for key,table_html in build_results().items():
        content=content.replace(f'<div id="{key}"></div>',f'<div id="{key}">{table_html}</div>')
    # Convert atlas citation notation while excluding repository-only provenance labels.
    content=re.sub(r'(?<!\[)\[([a-z]+-[a-z0-9-]+)\](?!\])',lambda m:'[['+m[1]+']]' if m[1] in refs else '',content)
    used=[]
    for key in re.findall(r'\[\[([^\]]+)\]\]',content):
        assert key in refs,'Missing verified reference: '+key
        if key not in used:used.append(key)
    def cite(m):
        key=m[1];d=refs[key];n=used.index(key)+1
        return f'<a class="cite" href="#ref-{key}" title="{esc(d["authors"])} · {d["year"]} · {esc(d["title"])}" aria-label="Reference {n}: {esc(d["title"])}">[{n}]</a>'
    content=re.sub(r'\[\[([^\]]+)\]\]',cite,content)
    params=[('Fixed motion bank',0),('Correlation refinement + temperatures + trace rate',81),('Appearance CNN',5856),('292→64 fusion',18752),('Positional embeddings + CLS token',6464),('Three attention LayerNorms',384),('Query + key projections + bounded scale',8193),('Visual and memory values',8256),('Residual feedforward block',33216),('Spatial GRU, including gate/candidate norms',221760),('CLS GRU, including gate/candidate norms',25152),('Shared 799→64 readout',51200),('Sixteen task heads',2340)]
    assert sum(v for _,v in params)==381654
    table='<div class="table-wrap"><table><thead><tr><th>Component</th><th>Learned parameters</th></tr></thead><tbody>'+''.join(f'<tr><td>{esc(n)}</td><td>{v:,}</td></tr>' for n,v in params)+'<tr><th>Total</th><th>381,654</th></tr></tbody></table></div>'
    content=content.replace('<div id="parameter-table"></div>',table)
    content = re.sub(r'(<canvas\b.*?</canvas>|<svg\b.*?</svg>)', r'<div class="figure-viewport">\1</div>', content, flags=re.S)
    bibliography='<section id="references" class="chapter"><div class="eyebrow">10 · References</div><h2>The scientific foundation.</h2><p>'+str(len(used))+' distinct scholarly references cited in the manuscript. Search by author, title, or year.</p><label class="reference-search">Find a source <input id="reference-search" type="search" placeholder="e.g. motion, Krauzlis, normalization"></label><ol class="bibliography">'
    for key in used:
        d=refs[key];bibliography+=f'<li id="ref-{key}"><div><span class="ref-authors">{esc(d["authors"])}</span> ({d["year"]}). <a href="{esc(d["url"])}" target="_blank" rel="noopener">{esc(d["title"])}</a><details><summary>Source verification</summary><p>{esc(d.get("verification","Bibliographic record verified"))}</p></details></div></li>'
    bibliography+='</ol><p id="reference-empty" hidden>No references match that search.</p></section>'
    nav=[('introduction','The scientific question'),('tasks','Original task atlas'),('vda-tasks','VDA4 / 9 / 16 tasks'),('optical-flow','Optical flow, explained'),('architecture','Pixels → representations'),('attention-memory','Attention → memory → choice'),('neuroscience','The neuroscience in depth'),('experiments','Interventions & further tests'),('results','Measured results'),('independent-findings','October 11: independent findings'),('references',f'References · {len(used)}')]
    hero='<header class="hero" id="top"><div class="eyebrow">Computational neuroscience · Research explainer</div><h1>A shared spatial<br><em>priority map</em><br>for visual selection<br>and working memory.</h1><p class="hero-deck">From pixels and motion correspondence to recurrent attention, memory, and task-dependent choice.</p><div class="hero-meta"><span>19 trained tasks / 437 tested conditions</span><span>One shared attention map</span><span>Methods + measured results</span></div><div class="hero-note"><b>Draft status.</b> Updated October 11, 2026. The new independent-findings chapter adds the completed 19-task test and four measured figures to the original 13-task and 16-task analysis galleries. The original stimulus atlas remains 13 tasks; later search results have separate evaluation conditions.</div><a class="hero-start" href="#independent-findings">Read the latest findings <span>↓</span></a></header>'

    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A shared spatial priority map · Illustrated manuscript</title><link rel="stylesheet" href="task-atlas/vendor/fonts.css"><link rel="stylesheet" href="manuscript.css"><script>window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]},svg:{fontCache:'local'},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script><script defer src="vendor/tex-svg.js"></script><script defer src="manuscript.js"></script></head><body><a class="skip-link" href="#introduction">Skip to manuscript</a><div class="reading-progress" id="reading-progress"></div><aside class="sidebar"><a class="wordmark" href="#top">SELECTION<br><span>& MEMORY</span></a><p class="sidebar-label">An illustrated research manuscript</p><nav aria-label="Manuscript chapters">'''
    config = dict(tex=dict(inlineMath=[[chr(92)+'(', chr(92)+')']], displayMath=[[chr(92)+'[', chr(92)+']']]), svg=dict(fontCache='local'))
    document = re.sub(r'window.MathJax=.*?;</script>', lambda _: 'window.MathJax='+json.dumps(config)+';</script>', document)
    document+=''.join(f'<a href="#{key}"><span>{i+1:02}</span>{esc(label)}</a>' for i,(key,label) in enumerate(nav))
    document+='</nav><div class="sidebar-bottom"><button id="print-paper">Print / save PDF</button><a href="task-atlas/index.html" target="_blank">Open task atlas ↗</a><a href="../Analysis/run01/gallery/index.html" target="_blank">Frozen analysis & attention maps ↗</a><small>Local, offline reading<br>Research draft · October 2026</small></div></aside><main>'+hero+content+bibliography+'<footer>Methods grounded in the frozen implementation. Interactive teaching figures use specified synthetic examples. <a href="#top">Back to top ↑</a></footer></main></body></html>'
    # Python source escaping must produce exactly one TeX backslash in JS strings after parsing.
    # The guided explainer owns index.html. Keep the extended manuscript as a
    # separately addressable reference so a legacy rebuild cannot replace it.
    (ROOT/'reference.html').write_text(document,encoding='utf-8')
    selected=[{field:refs[k][field] for field in ('id','year','authors','title','url','verification','doi') if field in refs[k]} for k in used]
    (ROOT/'research/bibliography.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'research/parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    (ROOT/'task-data.js').write_text('window.MANUSCRIPT_TASKS='+json.dumps([dict(id=t['id'],title=t['title'],group=t['group']) for t in tasks])+';',encoding='utf-8')
    # Insert task data before the behavioral script.
    document=document.replace('<script defer src="manuscript.js">','<script src="task-data.js"></script><script defer src="pixels-tutorial.js"></script><script defer src="manuscript.js">')
    (ROOT/'index.html').write_text(document,encoding='utf-8')
    words=len(BeautifulSoup(document,'html.parser').get_text(' ',strip=True).split())
    receipt=dict(built=datetime.datetime.now(datetime.timezone.utc).isoformat(),references=len(used),words_including_references=words,
        tasks=19,conditions=437,earlier_vda_tasks=16,earlier_vda_conditions=44,original_atlas_tasks=13,original_atlas_conditions=35,results_blank=False,sections=section_names,missing_reference_keys=[],status='built; browser QA pending')
    (ROOT/'research/build_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(receipt)

if __name__=='__main__':main()
