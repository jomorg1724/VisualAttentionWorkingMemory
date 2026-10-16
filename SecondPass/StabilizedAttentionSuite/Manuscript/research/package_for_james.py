"""Assemble a portable copy of the illustrated report and its display dependencies."""
from pathlib import Path
import re,json,html,hashlib,shutil,zipfile,base64
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
from PIL import Image

REPO=Path(__file__).resolve().parents[4]
MAN=Path(__file__).resolve().parents[1]
DEST=REPO/'output/share_james_2026-10-11'
DEST.mkdir(parents=True,exist_ok=True)
files=set(); queue=[]
def add(p):
    p=p.resolve()
    try:p.relative_to(REPO)
    except ValueError:return
    if p.is_file() and p not in files:
        if p.suffix in ('.pt','.pyc') or '__pycache__' in p.parts:return
        files.add(p);queue.append(p)

for name in ('index.html','manuscript.css','manuscript.js','pixels-tutorial.js','task-data.js'):
    add(MAN/name)
for folder in ('assets','vendor','pixels-figures','task-atlas'):
    for p in (MAN/folder).rglob('*'):add(p)
for sub in ('StabilizedAttentionSuite','StabilizedAttentionVDA'):
    b=REPO/'SecondPass'/sub/'Analysis/run01'
    for folder in ('gallery','figures','animations','atlas'):
        for p in (b/folder).rglob('*'):add(p)
for name in ('bibliography.json','results_sources.json','implementation_sources.json','parameters.json','build_receipt.json'):
    add(MAN/'research'/name)

missing=set()
def local_refs(p):
    if p.suffix.lower() not in ('.html','.css','.js'):return []
    s=p.read_text(encoding='utf-8',errors='replace')
    if p.suffix.lower()=='.html':
        soup=BeautifulSoup(s,'html.parser')
        refs=[el.get(k) for el in soup.find_all(True) for k in ('src','href','poster') if el.get(k)]
    else:refs=[]
    refs += re.findall(r'url\(\s*[\"\x27]?([^\)\"\x27]+)',s)
    if p.suffix.lower()=='.js':
        candidates = re.findall(r'[\"\x27]([^\"\x27\n]+\.(?:png|gif|jpg|jpeg|svg|html|json|css|js|woff2?))[\"\x27]',s)
        if p.name == 'summary.js':
            candidates = ['../figures/'+r if '/' not in r and r.endswith('.png') else r for r in candidates]
        if p.name == 'data.js' and p.parent.name == 'task-atlas':
            # This ledger also stores source-code citations, which are not URLs.
            candidates = [r for r in candidates if r.startswith('assets/')]
        if p.parent.name == 'vendor':
            # Bundled MathJax embeds optional module names, not display dependencies.
            candidates = []
        refs += candidates
    # These are photograph provenance IDs in trial metadata, not loaded URLs.
    return [r for r in refs if not r.startswith('BSDS500/')]
while queue:
    p=queue.pop()
    for raw in local_refs(p):
        if raw.startswith(('data:','https:','http:','mailto:','tel:','#','javascript:')):continue
        path=unquote(urlsplit(raw).path)
        if not path or '${' in path or path.startswith('/'):continue
        target=(p.parent/path).resolve()
        if target.is_file():add(target)
        else:missing.add((str(p.relative_to(REPO)),raw))

for p in files:
    target=DEST/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)

def dataimage(path):
    typ='gif' if path.suffix=='.gif' else 'png'
    return 'data:image/'+typ+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
figs=MAN/'assets/independent_20261011'
full='SecondPass/StabilizedAttentionSuite/Manuscript/index.html'
css='body{margin:0;background:#f7f6f1;color:#172d36;font:18px/1.65 Georgia,serif}main{max-width:1040px;margin:auto;padding:42px 26px}h1,h2,h3,nav{font-family:system-ui,sans-serif}h1{font-size:38px;line-height:1.18}h2{font-size:26px;margin-top:40px}a{color:#087f81}nav{display:flex;flex-wrap:wrap;gap:12px;padding:18px;background:#e5ece8;border-radius:8px}nav a{padding:8px}.figure{background:white;padding:12px;border:1px solid #d6ded8;border-radius:7px;margin:26px 0}.figure img{display:block;width:100%;height:auto}.figure p{font-size:15px;margin:12px}.meta{font:14px/1.5 system-ui,sans-serif;color:#607078}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cards img{width:100%}@media(max-width:650px){h1{font-size:29px}.cards{grid-template-columns:1fr}}'
intro='<h1>Local motion evidence, shared attention, and working memory</h1><p class="meta">Jonathan Morgan • Independent computational findings • October 11, 2026</p><p>The model now learns moving-dot tasks that earlier versions failed to learn, alongside appearance and working-memory judgments. The main mechanistic link to James’s analyses is a separation between constructing local comparison evidence and selecting which evidence influences a decision.</p>'
nav='<nav><a href="'+full+'#independent-findings">Latest detailed findings</a><a href="'+full+'">Full illustrated report</a><a href="SecondPass/StabilizedAttentionSuite/Analysis/run01/gallery/index.html#atlas">Play 13-task attention movies</a><a href="SecondPass/StabilizedAttentionVDA/Analysis/run01/gallery/index.html#atlas">Play VDA attention movies</a><a href="SecondPass/StabilizedAttentionSuite/Manuscript/task-atlas/index.html">Native task animations</a></nav>'
claims='<h2>The principal results</h2><p>The latest completed, validation-selected 19-task checkpoint was evaluated on 111,872 fresh presentations across 437 conditions. Mean balanced accuracy: motion direction 100%, cued motion-duration memory 98.54%, native cued motion change 94.45%, motion-memory search 81.15%. Object-memory search and pop-out remain weaker; this is broader demonstrated task learning, not uniform success or robustness across seeds.</p><p>A frozen reciprocal attention-trajectory swap rescued an earlier collapsed model from 50% to 75.73% balanced accuracy and reduced a good model from 74.91% to 50%, on the same 300 diagnostic validation movies. This confirms causal dependence on the attention-controlled trajectory in that experiment, including its memory writes.</p><p>In the frozen VDA-4 analysis, cued inhibition improved opposite-location detection by 19.79 percentage points (paired 95% interval 15.62–24.22), while cued detection fell and catch false alarms decreased. The opposite change deliberately violated a 100%-valid cue. The explicit long-window comparison branch is zero on these seven-frame trials.</p>'
captions=[('01_task_capabilities','Nineteen trained tasks; the difficult new search conditions remain visible.'),('02_motion_generalization','Motion change and retention across timing conditions. Historical checkpoints use independent test banks and different training exposure.'),('03_attention_routing_rescue','Frozen reciprocal attention swaps. The intervention changes selection and recurrent writes; it does not isolate the explicit comparison branch.'),('04_vda_reciprocal_effects','Paired spatial trade-offs. Scene-level uncertainty concerns one frozen checkpoint; the earlier suite’s null balanced-accuracy benefits remain in the full report.')]
body=intro+nav+claims
for name,caption in captions:
    body+='<figure class="figure"><img src="'+dataimage(figs/(name+'.png'))+'" alt="'+html.escape(caption)+'"><p>'+html.escape(caption)+' <a href="SecondPass/StabilizedAttentionSuite/Manuscript/assets/independent_20261011/'+name+'.pdf">Vector PDF</a></p></figure>'
body+='<h2>Example measured attention animations</h2><p>These GIFs are display examples from the earlier frozen 13-task and VDA checkpoints, not measurements at the latest 19-task checkpoint. The full galleries provide frame controls, original inputs, and condition metadata.</p><div class="cards">'
for p,title in [(REPO/'SecondPass/StabilizedAttentionSuite/Analysis/run01/animations/krauzlis_cued_motion.gif','Native cued motion • checkpoint 54,808'),(REPO/'SecondPass/StabilizedAttentionVDA/Analysis/run01/figures/attention_vda4.gif','VDA-4 attention • checkpoint 16,000')]:
    body+='<figure class="figure"><img src="'+dataimage(p)+'" alt="'+title+'"><p>'+title+'</p></figure>'
body+='</div><h2>Reading the complete package</h2><p>Download the ZIP and extract the entire folder before opening <b>START_HERE.html</b> in a browser. Keep the sibling folders intact. This summary embeds its four plots and two GIFs, so it remains readable as a single file; the full illustrated report, animations and interactive controls use the accompanying folders.</p><p>For sharing, upload the ZIP to Google Drive and restrict access to the intended collaborators. Drive preview is only a download surface for this package: opening the downloaded HTML in a browser preserves the scripts and animations. A PDF can supplement the figures, but it cannot preserve interactive demonstrations.</p><p>Evidence is computational: engineered motion comparisons, terminal supervised losses and attention-logit interventions differ from James’s original reinforcement-learning implementation and biological SC perturbations. Source hashes are provided with the new figures; no checkpoint weights or private third-party manuscripts are included.</p>'
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jonathan Morgan — independent findings for James</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
(DEST/'START_HERE.html').write_text(page,encoding='utf-8')
(DEST/'READ_ME.txt').write_text('Extract the entire ZIP. Open START_HERE.html in a browser.\nThe full report and interactive galleries require the sibling folders.\nIncluded: original task atlas, 13-task and VDA frozen galleries, all 16 original GIFs, four new measured figures and source ledgers.\nThe latest search suite is summarized numerically; its new search tasks are not added to the original 13-task stimulus atlas.\nCheckpoint weights, active training files, and James/Pitkow private manuscript are excluded.\n',encoding='utf-8')

errors=[]
for p in DEST.rglob('*'):
    if p.is_file() and p.suffix.lower() in ('.html','.css','.js'):
        for ref in local_refs(p):
            if ref.startswith(('data:','https:','http:','mailto:','tel:','#','javascript:')):continue
            r=unquote(urlsplit(ref).path)
            if not r or '${' in r or r.startswith('/'):continue
            if not (p.parent/r).is_file():errors.append({'file':str(p.relative_to(DEST)),'reference':ref})
gifs=[]
for p in DEST.rglob('*.gif'):
    with Image.open(p) as im:
        count=im.n_frames
        for i in range(count):im.seek(i);im.load()
        gifs.append({'path':str(p.relative_to(DEST)),'frames':count})
manifest=[{'path':str(p.relative_to(DEST)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(DEST.rglob('*')) if p.is_file() and p.name not in ('manifest.json','package_verified.json')]
receipt={'file_count':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'local_link_errors':errors,'source_missing':sorted(missing),'gifs_decoded':gifs,'no_checkpoints':not any(p.suffix=='.pt' for p in DEST.rglob('*')),'scope':'Portable display dependency collection; no model replication; browser verification follows.'}
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(DEST/'package_verified.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
if errors:print('LINK ERRORS',json.dumps(errors[:25]));raise SystemExit(2)
zip_path=DEST.with_suffix('.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in DEST.rglob('*'):
        if p.is_file():z.write(p,DEST.name+'/'+p.relative_to(DEST).as_posix())
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
print(json.dumps({'package':str(DEST),'files':len(manifest),'zip_MB':zip_path.stat().st_size/1e6,'gifs':len(gifs),'gif_frames':sum(g['frames'] for g in gifs),'local_link_errors':len(errors),'missing_source':len(missing)},indent=2))
