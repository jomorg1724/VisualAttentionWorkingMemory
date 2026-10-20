"""Check literature coverage, LaTeX references and saved-trial reanalysis."""
import hashlib,json,re,time
import fitz
from PIL import Image,ImageDraw
from .design import OUT,TASKS
from .literature_sources import SOURCES
from .neuroscience_content import CHAPTERS
from .neuroscience_report import enhance_source,BEGIN,END,DISCBegin,DISCEnd,RESULTBEGIN,RESULTEND,references

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert set(TASKS)==set(CHAPTERS)
    root=OUT/'papers';review=OUT/'literature_review';review.mkdir(exist_ok=True)
    rows=[];thumbs=[]
    for task in TASKS:
        source=(root/f'{task}.tex').read_text(encoding='utf8')
        assert source.count(BEGIN)==source.count(END)==1,task
        assert source.count(DISCBegin)==source.count(DISCEnd)==1,task
        assert source.index(BEGIN)<source.index(r'\section{Methods}'),task
        assert source.index(r'\section{Discussion}')<source.index(DISCBegin)<source.index(r'\begin{thebibliography}'),task
        assert 'Comparison with biological experiments and theory' not in source,task
        if (OUT/task/'literature_analysis.json').exists():
            assert source.count(RESULTBEGIN)==source.count(RESULTEND)==1
            assert source.index(r'\section{Results}')<source.index(RESULTBEGIN)<source.index(r'\section{Discussion}')
        assert enhance_source(task,source)==source,('non-idempotent',task)
        keys=references(task);assert len([k for k in keys if SOURCES[k]['kind']!='theory'])>=2
        cites=set(k for c in re.findall(r'\\cite\{([^}]+)\}',source) for k in c.split(','))
        items=re.findall(r'\\bibitem\{([^}]+)\}',source)
        assert len(items)==len(set(items)) and cites<=set(items),(task,cites-set(items))
        doc=fitz.open(root/f'{task}.pdf');texts=[p.get_text() for p in doc]
        # Render both integrated passages, including all intervening pages and
        # the result/discussion transition for the two empirical reanalyses.
        intro=next(i for i,t in enumerate(texts) if 'Biological experiments motivating this task' in t)
        methods=next(i for i,t in enumerate(texts) if re.search(r'\d+\s+Methods',t))
        disc=next(i for i,t in enumerate(texts) if 'Interpreting these results alongside biological experiments' in ' '.join(t.split()))
        end=next(i for i in range(disc,len(doc)) if 'Scope of inference' in texts[i])
        pages=set(range(intro,methods+1))|set(range(max(0,disc-1),end+1))
        for i in sorted(pages):
            p=review/f'{task}_{i+1:03}.png';doc[i].get_pixmap(matrix=fitz.Matrix(.6,.6)).save(p);thumbs.append(p)
        rows.append(dict(task=task,primary_experiments=[k for k in keys if SOURCES[k]['kind']!='theory'],comparison_pages=[i+1 for i in sorted(pages)],integrated_into=['introduction','discussion']+(['results'] if (OUT/task/'literature_analysis.json').exists() else []),pdf_sha256=sha(root/f'{task}.pdf')))
    # Independent reconciliation with previously generated condition confusion tables.
    r=read(OUT/'image_recognition/literature_analysis.json')['rows']
    cells={c['cell']:c for c in read(OUT/'image_recognition/behavior.json')['cells']}
    for x in r:
        c=cells[x['cell']]['confusion'];assert c[1][1]==x['hits'] and c[0][1]==x['false_alarms']
        assert sum(c[1])==x['repeat_trials'] and sum(c[0])==x['novel_trials']
        if x['load']==0:assert x['dprime'] is None and x['criterion'] is None
    cells={c['cell']:c for c in read(OUT/'motion_duration_cued/behavior.json')['cells']}
    for r in read(OUT/'motion_duration_cued/literature_analysis.json')['rows']:
        assert r['model_accuracy']==cells[r['cell']]['accuracy']
        assert 0<=r['model_correct_on_conflict']<=r['conflict_trials']<=r['n']
    for relative,value in read(OUT/'literature_reanalysis_provenance.json')['source_files'].items():assert sha(OUT/relative)==value
    doc=fitz.open(root/'neuroscience_comparison.pdf');bounds=[]
    for i,page in enumerate(doc):
        text=page.get_text();assert '??' not in text and '\ufffd' not in text
        for b in page.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for s in line['spans']:
                    x0,y0,x1,y1=s['bbox']
                    if x0<0 or y0<0 or x1>page.rect.width+.5 or y1>page.rect.height+.5:bounds.append((i+1,s['text']))
        p=review/f'companion_{i+1:03}.png';page.get_pixmap(matrix=fitz.Matrix(.6,.6)).save(p);thumbs.append(p)
    assert not bounds,bounds
    log=(root/'neuroscience_comparison.log').read_text(errors='replace')
    warnings=[l for l in log.splitlines() if 'Overfull' in l or 'Float too large' in l or 'undefined' in l]
    assert not warnings,warnings
    for start in range(0,len(thumbs),24):
        sheet=Image.new('RGB',(4*340,6*465),'#e3ebef');draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(thumbs[start:start+24]):
            im=Image.open(p).convert('RGB');im.thumbnail((330,435));x=(j%4)*340+(340-im.width)//2;y=(j//4)*465
            sheet.paste(im,(x,y));draw.text(((j%4)*340+3,y+437),p.stem,fill='#173547')
        sheet.save(review/f'contact_{start//24:02}.jpg',quality=90)
    receipt=dict(time=time.time(),tasks=rows,source_count=len(SOURCES),primary_study_links=sum(len(r['primary_experiments']) for r in rows),
        companion_pages=len(doc),companion_sha256=sha(root/'neuroscience_comparison.pdf'),bounds_errors=bounds,layout_warnings=warnings,
        new_model_forwards=0,condition_statistics_reconciled=True,source_hashes_verified=True,latex_references_resolved=True,
        all_new_pages_rendered=True,rendered_page_count=len(thumbs),contact_sheets=(len(thumbs)+23)//24,visual_review='pending')
    (OUT/'literature_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='tasks'},indent=2))

if __name__=='__main__':main()
