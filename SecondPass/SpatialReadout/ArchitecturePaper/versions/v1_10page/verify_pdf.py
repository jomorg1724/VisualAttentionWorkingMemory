"""Check PDF extraction/layout and render every page for visual review."""
from pathlib import Path
import json, hashlib, re, unicodedata
import fitz
HERE=Path(__file__).resolve().parent
pdf=HERE/'SpatialReadout_Architecture.pdf'
doc=fitz.open(pdf)
render=HERE/'rendered'; render.mkdir(exist_ok=True)
rows=[]
for i,p in enumerate(doc):
    text=p.get_text()
    (render/f'page_{i+1:02d}.txt').write_text(text)
    words=p.get_text('words')
    outside=[list(w[:5]) for w in words if w[0]<0 or w[1]<0 or w[2]>p.rect.width or w[3]>p.rect.height]
    bounds=[min(w[0] for w in words),min(w[1] for w in words),max(w[2] for w in words),max(w[3] for w in words)] if words else None
    p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(render/f'page_{i+1:02d}.png')
    rows.append(dict(page=i+1,characters=len(text),words=len(words),bounds=bounds,outside_page=outside,replacement_glyphs=text.count('\ufffd'),first_lines=text.splitlines()[:5]))
log=(HERE/'SpatialReadout_Architecture.log').read_text()
issues=[x for x in log.splitlines() if any(s in x for s in ('Overfull','Underfull','Missing character','undefined','LaTeX Warning'))]
fonts=[]
for p in doc:
    for entry in p.get_fonts(full=True):
        if entry[0] not in [f['xref'] for f in fonts]:
            name,ext,kind,data=doc.extract_font(entry[0])
            fonts.append(dict(xref=entry[0],name=name,extension=ext,type=kind,embedded_bytes=len(data)))
result=dict(pdf=str(pdf),sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),page_count=len(doc),pages=rows,fonts=fonts,all_fonts_embedded=all(f['embedded_bytes']>0 for f in fonts),log_issues=issues,no_out_of_page_text=all(not r['outside_page'] for r in rows),no_replacement_glyphs=all(r['replacement_glyphs']==0 for r in rows),has_expected_parameter_count='1,383,028' in ''.join(p.get_text() for p in doc),visual_review='Pending; render files generated for direct image inspection. Automated bounds do not establish no overlap.')
sections=['Scope and scientific question','End-to-end computation and notation','Sensory interface and the multiscale encoder','Exact spatial KDA','What KDA remembers','Final spatial recurrence and terminal compression','Task interface and current training lineage','Design decisions','Interpretation, limitations','Reproducibility and source map']
result['section_page_alignment'] = len(doc)==len(sections) and all(s in unicodedata.normalize('NFKC',doc[i].get_text()) for i,s in enumerate(sections))
result['total_extracted_words'] = sum(r['words'] for r in rows)
counts=json.loads((HERE/'count_verification.json').read_text())
root=HERE.parents[2]
result['source_hashes_unchanged'] = all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in counts['source_sha256'].items())
result['source_files_checked'] = len(counts['source_sha256'])
result['no_tex_warnings'] = not issues
review=HERE/'visual_review.json'
if review.exists():
    saved=json.loads(review.read_text())
    if saved.get('pdf_sha256') == result['sha256']:
        result['visual_review']=saved
result['automated_pass']=all(result[k] for k in ('all_fonts_embedded','no_out_of_page_text','no_replacement_glyphs','has_expected_parameter_count','section_page_alignment','source_hashes_unchanged','no_tex_warnings'))
(HERE/'pdf_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('fonts','pages')},indent=2))
assert result['automated_pass'], 'PDF/source verification failed; inspect receipt'
