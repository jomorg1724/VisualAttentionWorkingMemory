"""Verify figure inventory, source identity, numerical exports and PDF bounds."""
from pathlib import Path
import csv, hashlib, json, math, re
import pymupdf
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
m=json.loads((OUT/'manifest.json').read_text())
assert m['figure_count']==len(m['figures'])==29
assert len({f['id'] for f in m['figures']})==29
for path,expected in m['sources'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
p=ROOT/'WorkingMemory/PlainBaseline/runs/local_kda_program_20260917/analysis/psychometric/psychometric.json'
src=json.loads(p.read_text()); rows=list(csv.DictReader((OUT/'data/psychometric_points.csv').open()))
expected=[]
for sweep,sw in src['sweeps'].items():
    for cond,v in (sw.items() if sweep=='magnitude' else [(sweep,sw)]):
        for pt in v['points']:expected.append((sweep,cond,pt))
assert len(rows)==len(expected)
for row,(sweep,cond,pt) in zip(rows,expected):
    assert row['sweep']==sweep and row['condition']==cond
    for key in ['ba','auc','accuracy']:
        assert math.isclose(float(row[key]),pt[key],rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(float(row['ci_low']),pt['ba_ci'][0],abs_tol=1e-12)
    assert math.isclose(float(row['ci_high']),pt['ba_ci'][1],abs_tol=1e-12)
raw=json.loads((ROOT/'WorkingMemory/PlainBaseline/runs/local_kda_program_20260917/analysis/kda_probe/kda_probe.json').read_text())
attention_rows=list(csv.DictReader((OUT/'data/implicit_attention_profiles.csv').open())); source_values=[]
for d in raw['delays']:
    for sc in d['scales']:
        for region,vals in sc['attention_abs_mean'].items():
            source_values.extend(vals)
assert len(attention_rows)==len(source_values)
assert all(math.isclose(float(r['mean_absolute_weight']),v,rel_tol=1e-12,abs_tol=1e-12) for r,v in zip(attention_rows,source_values))
with pymupdf.open(OUT/'KDA_results_atlas.pdf') as doc:
    assert len(doc)==29
    out_of_bounds=[]
    for i,(page,f) in enumerate(zip(doc,m['figures'])):
        assert len(page.get_text())>150
        for word in page.get_text('words'):
            if word[0]<-1 or word[1]<-1 or word[2]>page.rect.width+1 or word[3]>page.rect.height+1:
                out_of_bounds.append((i+1,word[:5]))
        for ext in ['svg','png','pdf']:assert (OUT/f[ext]).stat().st_size>1000
        with pymupdf.open(OUT/f['pdf']) as single:assert len(single)==1
    assert not out_of_bounds,out_of_bounds
html=(OUT/'index.html').read_text()
ids=re.findall(r'\bid="([^"]+)"',html)
assert len(ids)==len(set(ids))
for target in re.findall(r'(?:src|href)="([^"]+)"',html):
    if not target.startswith(('#','http','mailto:')):assert (OUT/target).exists(),target
result=dict(status='passed',figures=29,atlas_pages=29,verified_source_files=len(m['sources']),psychometric_points=len(rows),attention_values=len(source_values),duplicate_html_ids=0,missing_local_links=0,out_of_bounds_pdf_words=0)
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
