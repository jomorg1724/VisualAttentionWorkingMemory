"""Verify delivery completeness, internal references and archive integrity."""
import json,re,time,zipfile
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import fitz
from .finalize import OUT,PAPERS,TASKS,read,write,digest

class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.paths.append(value)

def main():
    receipt=read(OUT/'delivery.json');audit=read(OUT/'pdf_audit.json')
    assert audit['visual_inspection']=='passed'
    assert len(audit['papers'])==20
    literature=read(OUT/'literature_verification.json')
    assert literature['visual_review']=='passed'
    assert digest(PAPERS/'neuroscience_comparison.pdf')==literature['companion_sha256']
    assert len(literature['tasks'])==19 and literature['source_count']==43
    for task in TASKS:
        source=(PAPERS/f'{task}.tex').read_text(encoding='utf8')
        assert 'divided by the square root of its coordinate count' in source
        assert source.count(r'\section{Discussion}')==1
        for row in read(OUT/task/'behavior.json')['factors']:
            assert 'ci95' not in row and 'Descriptive marginal' in row['uncertainty']
        for relative in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',source):
            assert (PAPERS/relative).is_file(),(task,relative)
    for row in audit['papers']:
        assert digest(PAPERS/f"{row['task']}.pdf")==row['sha256']
    doc=fitz.open(receipt['volume'])
    assert len(doc)==audit['total_pages']==receipt['full_volume_pages']
    assert len(doc.get_toc())==20
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf8'))
    local=[]
    for link in parser.paths:
        url=urlsplit(link)
        if not url.scheme and url.path:
            path=OUT/unquote(url.path);assert path.is_file(),link;local.append(url.path)
    with zipfile.ZipFile(receipt['archive']) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))
        assert not any(n.lower().endswith(('.pt','.pth','.ckpt','.npz','.mp4','.gif','.avi')) for n in names)
        for task in TASKS:
            assert f'papers/{task}.pdf' in names and f'papers/{task}.tex' in names
            if (OUT/task/'attention_dynamics.json').exists():
                assert f'{task}/attention_dynamics.json' in names
                d=read(OUT/task/'attention_dynamics.json')
                for g in d['groups']:assert f'{task}/figures/{g["sprite"]}' in names
                f=d['figures']
                for stem in f['maps']+f['landmarks']+f['curves']+[f['comparison']]:
                    for ext in ('png','pdf'):assert f'{task}/figures/{stem}.{ext}' in names
                assert f'{task}/source_tables/attention_condition_timecourses.csv' in names
        assert 'attention/index.html' in names and 'attention/verification.json' in names
        for n in ('papers/neuroscience_comparison.pdf','papers/neuroscience_comparison.tex','literature_sources.json','literature_coverage.json','literature_verification.json','literature_reanalysis_provenance.json'):
            assert n in names,n
        for path in local:
            if not path.endswith('.zip'):assert path in names,path
        assert 'statistical_identity_audit.json' in names
        assert json.loads(z.read('pdf_audit.json'))['visual_inspection']=='passed'
    assert digest(OUT/'nineteen_task_papers_and_source.zip')==receipt['archive_sha256']
    write(OUT/'delivery_verified.json',dict(time=time.time(),papers=19,synthesis=True,
        pages=len(doc),bookmarks=20,archive_members=len(names),html_local_links=len(local),
        archive_crc_passed=True,internal_references_passed=True,methods_synchronized=True,
        checkpoint_hashes_unchanged=read(OUT/'data_audit.json')['checkpoint_hashes_unchanged'],
        no_weights_raw_arrays_or_videos_in_archive=True,archive_sha256=receipt['archive_sha256']))
    result=read(OUT/'delivery_verified.json');result.update(neuroscience_companion_pages=literature['companion_pages'],literature_sources=43,task_study_links=literature['primary_study_links'],literature_reanalysis_new_model_forwards=0)
    write(OUT/'delivery_verified.json',result)
    print(json.dumps(read(OUT/'delivery_verified.json'),indent=2))

if __name__=='__main__':main()
