"""Validate sensory editorial contracts and metadata math; no renderer imports."""
import json, math, re, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
tasks=json.loads((ROOT/'content/sensory_tasks.json').read_text())
sources=json.loads((ROOT/'content/sensory_sources.json').read_text())
manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
catalog=json.loads((REPO/'SecondPass/TaskSuite/catalog.json').read_text())
expected=[t['id'] for t in catalog['tasks'] if t['group']=='sensory']
assert [t['id'] for t in tasks]==expected
assert len(set(expected))==7
source_map={s['id']:s for s in sources}
assert len(source_map)==len(sources)
sections={'appears','sequence','rules','ignore','implementation','neuroscience','network','boundary','metrics','adaptation'}
required={'name','value','units','sampling','role','provenance','visibility'}
provenance_rows=0; citations=0; checked=0; quotes=0
for s in sources:
    assert {'id','authors','title','year','url','doi','type','verification','support_location','supports','does_not_support','task_ids'} <= s.keys()
    text=' '.join((ROOT/s['evidence_file']).read_text().split())
    for quote in s['evidence_quotes']:
        assert ' '.join(quote.split()) in text,(s['id'],quote)
        quotes+=1
for t in tasks:
    assert t['group']=='sensory' and t['question'] and t['intro']
    assert set(t['sections'])==sections
    assert all(isinstance(p,list) and p and all(isinstance(s,str) and len(s.split())>=20 for s in p) for p in t['sections'].values())
    assert all(r in source_map for r in t['references'])
    paragraphs=' '.join(p for ps in t['sections'].values() for p in ps)
    refs=re.findall(r'\[(sensory-[a-zA-Z0-9_-]+)\]',paragraphs)
    assert refs and set(refs)<=set(t['references'])
    citations+=len(refs)
    assert any('primary experimental' in source_map[r]['type'] or 'primary psychophysical' in source_map[r]['type'] for r in t['references'])
    for p in t['properties']:
        assert required<=p.keys() and all(isinstance(p[k],str) and p[k] for k in required)
        file,lines=p['provenance'].rsplit(':',1)
        if file.startswith('SciPy '):
            line_count=len((ROOT/'verification/sensory_scipy_filters.txt').read_text().splitlines())
        else:
            path=REPO/file
            assert path.is_file(),path
            line_count=len(path.read_text().splitlines())
        for interval in lines.split(','):
            nums=[int(n) for n in re.findall(r'\d+',interval)]
            assert nums and all(1<=n<=line_count for n in nums),(file,lines,line_count)
        provenance_rows+=1
    episodes=[e for e in manifest['episodes'] if e['task_id']==t['id']]
    assert set(t['worked_examples'])=={e['id'] for e in episodes}
    assert t['showcase_worked_example']['text']==t['worked_examples'][t['showcase_worked_example']['episode_id']]
    for e in episodes:
        m=e['metadata']; y=e['label']; k=t['id']
        assert m['label']==y and m['split']=='train' and e['frame_count']==2
        if k=='motion_direction':
            assert ['right','up','left','down'][y]==m['direction']
            assert m['survivor_count_domain']==128 and m['reborn_count_domain']==128
        elif k=='orientation':
            d=m['signed_orientation_degrees']; theta=math.radians(d)
            wrapped=0.5*math.atan2(math.sin(2*theta),math.cos(2*theta))
            assert (wrapped>0)==bool(y) and abs(d) in (4,10,22)
        elif k=='contrast':
            a=m['frame_contrasts']; assert a[y]>a[1-y]
            assert math.isclose(a[y]-a[1-y],m['contrast_increment'])
            assert math.isclose(a[1-y],m['pedestal'])
        elif k=='spatial_frequency':
            f=m['frame_frequencies'];assert f[y]>f[1-y]
            assert math.isclose(f[y]/f[1-y],2**m['frequency_octave_increment'])
        elif k=='chromatic_increment':
            c=m['frame_colors'];u=m['axis_linear_rgb'];d=m['chromatic_increment']
            for j in range(3):assert math.isclose(c[y][j]-c[1-y][j],d*u[j],abs_tol=1e-7)
            assert abs(m['frame_luminances'][0]-m['frame_luminances'][1])<1e-6
        elif k=='contour':
            assert m['orientation_multiset_matched'] and m['path_elements']==7 and m['total_elements']==32
            assert m['alignment_jitter_degrees'] in (2,8,16)
        elif k=='natural_spectrum':
            b=m['betas_by_frame'];assert b[y]<b[1-y]
            assert math.isclose(b[1-y]-b[y],m['beta_delta']) and m['base_id'].startswith('BSDS500/train/')
            assert 0<m['actual_common_rms']<=.15 and m['clipping'] is False
        checked+=1
hashes=json.loads((ROOT/'verification/sensory_source_hashes.json').read_text())
for p,h in hashes.items():assert hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h
words={t['id']:len((' '.join([t['intro']]+[p for ps in t['sections'].values() for p in ps])).split()) for t in tasks}
report={'status':'pass','task_count':len(tasks),'source_count':len(sources),'required_section_count_per_task':len(sections),'provenance_property_rows':provenance_rows,'inline_citation_occurrences':citations,'verbatim_evidence_quotes_checked':quotes,'metadata_worked_examples_checked':checked,'word_counts_excluding_properties_examples':words,'total_editorial_words':sum(words.values()),'source_hashes_unchanged':True,'stimulus_rendering_performed':False,'model_operations':0,'scope':'Static source inspection, literature retrieval/OCR, JSON authoring and metadata arithmetic only; no runtime/checkpoint/training/cloud access.'}
(ROOT/'verification/sensory_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
