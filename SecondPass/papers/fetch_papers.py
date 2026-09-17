"""Download the open-access papers listed in BIBLIOGRAPHY.md into SecondPass/papers/pdf/ (git-ignored).

Only arXiv ids and direct PDF URLs found in BIBLIOGRAPHY.md are fetched; nothing is guessed. Paywalled references are
listed in the printed summary for manual retrieval. Usage: python SecondPass/papers/fetch_papers.py [--dry-run]
"""
import re,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'pdf';BIB=HERE/'BIBLIOGRAPHY.md'
text=BIB.read_text(encoding='utf-8')
arxiv=sorted(set(re.findall(r'(?:arxiv\.org/(?:abs|html|pdf)/|arXiv:)\s*(\d{4}\.\d{4,5})',text)))
pdfs=sorted(set(u for u in re.findall(r'https?://[^\s)\]]+',text) if u.lower().endswith('.pdf')))
dry='--dry-run' in sys.argv;OUT.mkdir(exist_ok=True);done=[];failed=[]
for aid in arxiv:
    url=f'https://arxiv.org/pdf/{aid}';target=OUT/f'arxiv_{aid}.pdf'
    if target.exists():done.append(target.name);continue
    if dry:print('would fetch',url);continue
    try:
        urllib.request.urlretrieve(url,target);done.append(target.name);time.sleep(1.5)
    except Exception as e:failed.append((url,repr(e)))
for url in pdfs:
    name=re.sub(r'[^A-Za-z0-9._-]+','_',url.split('/')[-1])[:80];target=OUT/name
    if target.exists():done.append(name);continue
    if dry:print('would fetch',url);continue
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'vawm-research'});target.write_bytes(urllib.request.urlopen(req,timeout=60).read());done.append(name);time.sleep(1.5)
    except Exception as e:failed.append((url,repr(e)))
print(f'arXiv ids found: {len(arxiv)}; direct PDF links: {len(pdfs)}; fetched or present: {len(done)}; failed: {len(failed)}')
for u,e in failed:print('FAILED',u,e)
print('Everything else in BIBLIOGRAPHY.md (journal articles without a PDF link) must be retrieved through the library.')
