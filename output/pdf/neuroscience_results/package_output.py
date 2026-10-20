from pathlib import Path
import json, shutil, zipfile
root=Path(__file__).resolve().parent
pdf=root/'Shared_Priority_Neuroscience_Results.pdf'
shutil.copy2(root/'main.pdf',pdf)
files=[root/name for name in ['main.tex','references.tex','references.json','build_figures.py','source_manifest.json','task_summary.json','layout_audit.json','BUILD.txt']]
files+=sorted((root/'figures').glob('*.pdf'))
with zipfile.ZipFile(root/'Neuroscience_Results_LaTeX.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p,p.relative_to(root))
audit=json.loads((root/'layout_audit.json').read_text())
assert audit['page_count']<=20
assert not audit['text_blocks_outside_page']
assert not audit['unresolved_question_mark_pair']
assert len(list((root/'figures').glob('*.pdf')))==12
assert len(json.loads((root/'references.json').read_text()))==34
print(pdf)
print('Source archive:',root/'Neuroscience_Results_LaTeX.zip')
