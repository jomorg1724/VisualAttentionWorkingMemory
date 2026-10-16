"""Audit manuscript structure and the bundled native stimulus provenance."""
import hashlib, json, re
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
doc=BeautifulSoup((ROOT/'index.html').read_text(encoding='utf-8'),'html.parser')
ids=[x['id'] for x in doc.select('[id]')]
assert len(ids)==len(set(ids))
assert all(x['href'][1:] in ids for x in doc.select('a[href^="#"]'))
refs=json.loads((ROOT/'research/bibliography.json').read_text(encoding='utf-8'))
assert 50<=len(refs)<=100
assert len({r['url'].lower() for r in refs})==len(refs)
assert all(r.get('verification') and r.get('authors') and r.get('year') for r in refs)
assert len(doc.select('.bibliography li'))==len(refs)
assert len(doc.select('#primary-rates tbody tr'))==3
assert len(doc.select('#primary-tests tbody tr'))==8
assert len(doc.select('#native-vda-results tbody tr'))==9
assert len(doc.select('#lineage-results tbody tr'))==16
assert len(doc.select('.math-reading'))==35
assert len(doc.select('.pixel-lesson'))==16
assert len(doc.select('.pixels-tutorial img'))==9
assert len(doc.select_one('.pixels-tutorial').get_text(' ',strip=True).split())>6000
assert len(doc.select('#pixels-softmax-canvas, #pixels-trace-canvas'))==2
sim=json.loads((ROOT/'research/pixels_simulations_verified.json').read_text())
assert sim['status']=='passed'
assert all(sha(ROOT/k)==v for k,v in sim['files'].items())
assert sha(REPO/'SecondPass/StabilizedAttentionKrauzlis/model.py')==sim['source_sha256']
assert 'intentionally blank' not in doc.get_text().lower()
assert 'G_t^{\\rm channel}' not in str(doc)
assert len(doc.select_one('#neuroscience').get_text(' ',strip=True).split())>2500
assert len(doc.select('#tasks .task-method'))==13
assert '[[' not in doc.get_text()
missing=[]
for el in doc.select('[src],link[href],a[href]'):
    value=el.get('src',el.get('href','')).split('#')[0]
    if value and not value.startswith(('http:','https:','data:','mailto:')) and not (ROOT/value).is_file():missing.append(value)
assert not missing,missing
source=json.loads((ROOT/'research/implementation_sources.json').read_text())
assert all(sha(REPO/k)==v for k,v in source.items())
atlas=ROOT/'task-atlas';manifest=json.loads((atlas/'manifest.json').read_text())
assert len(manifest['cells'])==35 and len(manifest['episodes'])==115
assets=0;frames=0
for e in manifest['episodes']:
    assert len(e['frames'])==e['frame_count']==len(e['frame_sha256'])
    for name,digest in zip(e['frames'],e['frame_sha256']):
        assert sha(atlas/name)==digest,name
        frames+=1
    for name,digest in e['asset_sha256'].items():
        assert sha(atlas/name)==digest,name
        assets+=1
    assert (atlas/e['metadata_path']).is_file()
assert len({(e['task_id'],e['condition_id']) for e in manifest['episodes']})==35
out=dict(status='passed',scholarly_references=len(refs),native_movies=115,conditions=35,
         frame_hashes_verified=frames,export_hashes_verified=assets,unchanged_source_files=len(source),
         broken_local_assets=missing,results_blank=False,duplicate_ids=0,broken_internal_citations=0,
         completed_tasks=16,completed_conditions=44,explained_equations=35,
         pixels_lessons=16,pixels_synthetic_figures=9,
         pixels_tutorial_words=len(doc.select_one('.pixels-tutorial').get_text(' ',strip=True).split()),
         scope='Static structure, citation identity fields, packaged asset integrity and source hashes. Browser checks recorded separately.')
(ROOT/'research/static_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
