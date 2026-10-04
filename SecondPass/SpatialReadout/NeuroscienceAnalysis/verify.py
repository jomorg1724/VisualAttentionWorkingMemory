"""Read-back verification of finished artifacts, no model/accelerator operation.
Checks scored counts, masks, all spatial axes, implicit recon errors, PDF pages,
calibration disjointness, full pinned intervention coverage, and file:// viewer.
"""
import json,hashlib,subprocess,tempfile,time,csv,re
from pathlib import Path
from collections import Counter
import numpy as np
import fitz
ROOT=Path(__file__).resolve().parent

def main():
    completion=json.loads((ROOT/'completion.json').read_text());budget=json.loads((ROOT/'budget.json').read_text())
    assert time.time()<budget['deadline_unix'],'Original nonrenewable deadline expired'
    rows=[json.loads(x) for x in (ROOT/'data/trials.jsonl').read_text().splitlines()];plan=json.loads((ROOT/'plan.json').read_text());grid=json.loads((ROOT/'intervention_grid.json').read_text());cal=json.loads((ROOT/'calibration.json').read_text())
    assert len(rows)==completion['scored_rows']
    errors=[]
    for r in rows:
        logits=np.array(r['logits']);e=np.exp(logits-logits.max());prob=e/e.sum();errors.append(abs(prob[1]-r['prob1']));assert r['prediction']==int(logits.argmax()) and r['correct']==(r['prediction']==r['label'])
    assert max(errors)<1e-6
    conditions={}
    for task in ['orientation_cued','spatial_binding']:
        rr=[r for r in rows if r['group']=='causal' and r['task']==task];counts=Counter(r['condition'] for r in rr);assert len(counts)==len(grid[task])+1 and set(counts.values())=={plan['causal_n']}
        sham={r['base_id'] for r in rr if r['condition']=='sham'}
        for c in counts:assert {r['base_id'] for r in rr if r['condition']==c}==sham
        conditions[task]=dict(conditions=len(counts),trials_per_condition=plan['causal_n'],presentations=len(rr))
    fit=set(cal['fit_ids']);val=set(cal['validation_ids']);test={r['trial_id'] for r in rows};assert not fit&val and not (fit|val)&test
    mapcounts=[];maxerr=0
    for p in sorted((ROOT/'maps').glob('*.npz')):
        a=dict(np.load(p));B,T=a['images'].shape[:2]
        for s,side in enumerate([25,13,7]):
            for q in ['beta','alpha_mean8','coeff_final_read','coeff_source2_all_reads']:
                v=a[f's{s}_{q}'];assert v.shape==(B,T,side,side,2) and np.isfinite(v).all()
            m=a[f's{s}_masks'];assert np.allclose(m.sum((1,2)),m[0].sum()) and len(set((m>0).sum((1,2))))==1
            maxerr=max(maxerr,float(a[f's{s}_reconstruction_error'].max()))
        assert a['gru_write'].shape==(B,T,7,7) and a['gru_reset'].shape==(B,T,7,7)
        mapcounts.append(dict(name=p.stem,trials=B,frames=T))
    assert len(mapcounts)==12 and maxerr<1e-4
    figures=json.loads((ROOT/'figure_manifest.json').read_text());pdf=fitz.open(ROOT/'Neuroscience_atlas.pdf');assert len(pdf)==len(figures)
    bounds_errors=[]
    for number,page in enumerate(pdf,1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    rect=fitz.Rect(span['bbox'])
                    if rect.x0<0 or rect.y0<0 or rect.x1>page.rect.width or rect.y1>page.rect.height:bounds_errors.append([number,span['text'],list(rect)])
    assert not bounds_errors,bounds_errors
    from html.parser import HTMLParser
    class LinkParser(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key not in ('href','src') or not value or value.startswith(('http:','https:','#','data:')):continue
                assert (ROOT/value.split('?')[0]).exists(),value
    for file in ['index.html','viewer.html']:LinkParser().feed((ROOT/file).read_text())
    for r in figures:
        for ext in ['png','svg','pdf']:assert (ROOT/'figures'/f'{r["id"]}.{ext}').stat().st_size>1000
    # All accelerator exposure, including the direct/recorded/zero-dose parity batch.
    scored_frames=sum(r['metadata'].get('frame_count',2) for r in rows)
    exposure=dict(scored_presentations=len(rows),scored_frames=scored_frames,map_presentations=sum(x['trials'] for x in mapcounts),map_frames=sum(x['trials']*x['frames'] for x in mapcounts),calibration_presentations=cal['n'],calibration_frames=cal['n']*4,profile_presentations=16,profile_frames=16*16,accelerator_parity_presentations=24,accelerator_parity_frames=24*16,note='Parity count is direct8 + recorded8 + zero-dose8 at16frames, all verified in accelerator_parity.json. CPU preparation was before first MPS and is excluded here.')
    exposure['total_presentations']=sum(v for k,v in exposure.items() if k.endswith('_presentations'));exposure['total_frames']=sum(v for k,v in exposure.items() if k.endswith('_frames'))
    (ROOT/'exposure_audit.json').write_text(json.dumps(exposure,indent=2))
    # Standalone Chrome avoids changing or using the user's default browser profile.
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    url=(ROOT/'viewer.html').as_uri()+'?dataset=orientation_cued_D12&trial=3&frame=14'
    with tempfile.TemporaryDirectory(prefix='isolated_chrome_',dir=checks) as profile:
        browser_timed_out=False
        try:
            result=subprocess.run([chrome,'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',f'--user-data-dir={profile}','--window-size=1440,1100','--virtual-time-budget=4000',f'--screenshot={checks/"viewer_screenshot.png"}','--dump-dom',url],capture_output=True,text=True,timeout=15)
        except subprocess.TimeoutExpired as exc:
            # Chrome can finish DOM/screenshot but retain shutdown pipe handles.
            # Retain the actual output and verify it; never substitute fake DOM.
            from types import SimpleNamespace
            browser_timed_out=True
            stdout=exc.stdout.decode() if isinstance(exc.stdout,bytes) else (exc.stdout or '')
            stderr=exc.stderr.decode() if isinstance(exc.stderr,bytes) else (exc.stderr or '')
            result=SimpleNamespace(stdout=stdout,stderr=stderr,returncode=None)
        (checks/'viewer_dom.html').write_text(result.stdout);(checks/'chrome.log').write_text(result.stderr)
        assert result.returncode in (0,None),result.stderr[-2000:]
        assert '"trial": 3' in result.stdout and '"frame": 14' in result.stdout and '"task": "orientation_cued"' in result.stdout,'Viewer did not populate requested trial/frame'
        assert 'pure blank' in result.stdout and 'KDA head0' in result.stdout
    catalog=json.loads((ROOT.parents[1]/'TaskSuite/catalog.json').read_text())
    ring=[t for t in catalog['tasks'] if t['id']=='orientation_ring'][0]
    assert [c['kwargs']['delay'] for c in ring['conditions']]==[0]
    process=subprocess.run(['ps','-p',str(json.loads((ROOT/'execution.json').read_text())['pid']),'-o','pid,stat,command'],capture_output=True,text=True).stdout
    assert ('<defunct>' in process or len(process.strip().splitlines())<=1),process
    receipt=dict(verified=True,scored_rows=len(rows),scored_softmax_max_error=max(errors),causal_coverage=conditions,calibration_ids_disjoint=True,map_datasets=mapcounts,implicit_reconstruction_max_error=maxerr,pdf_pages=len(pdf),viewer_local_file_test=True,browser_shutdown_timed_out_after_captured_output=browser_timed_out,viewer_verified_trial=3,viewer_verified_frame=14,worker_process_status=process.strip(),accelerator_worker_released=True,ring_delays_above0_are_ood=True,verification_unix=time.time(),elapsed_from_first_mps=time.time()-budget['start_unix'],deadline_unchanged=budget['deadline_unix'],within_cap=time.time()<budget['deadline_unix'])
    (ROOT/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
