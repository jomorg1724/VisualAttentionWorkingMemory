"""Build the ten-page report from saved artifacts; never load a model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import fitz

HERE = Path(__file__).resolve().parent
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
def run(args):
    subprocess.run(args, cwd=HERE, env=ENV, check=True)

run([sys.executable, str(HERE/'draw_architecture.py')])
run([sys.executable, str(HERE/'draw_validation.py')])
args=['pandoc','architecture_microstimulation.md','--from=markdown+raw_tex+tex_math_single_backslash+hard_line_breaks',
      '--standalone','--include-in-header=header.tex','--resource-path=.']
run(args+['-o','architecture_microstimulation.tex'])
run(args+['--pdf-engine=tectonic','-o','architecture_microstimulation.pdf'])
doc=fitz.open(HERE/'architecture_microstimulation.pdf')
assert len(doc)==10, f'Expected ten pages, got {len(doc)}'
checks=[]
for i,page in enumerate(doc):
    text=page.get_text()
    assert len(text.split())>150, f'Sparse/blank page {i+1}'
    assert '\ufffd' not in text, f'Replacement glyph on page {i+1}'
    headings=[line for line in text.splitlines() if line.startswith(str(i+1)+'. ')]
    assert headings, f'Wrong section/page alignment on page {i+1}'
    outside=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<15 or x1>page.rect.width-15 or y0<10 or y1>page.rect.height-10:
                    outside.append(span['text'])
    assert not outside, (i+1,outside)
    checks.append({'page':i+1,'words':len(text.split()),'heading':headings[0],'within_page_bounds':True})
    page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(HERE/f'preview_page_{i+1:02d}.png')
result={'pdf':str(HERE/'architecture_microstimulation.pdf'),'pages':len(doc),
        'page_checks':checks,'no_model_or_accelerator_calls':True,
        'note':'Programmatic checks complement visual review, not replace it. Fonts use macOS Times New Roman/Arial/Menlo.'}
(HERE/'build_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
