import sys,json,time,hashlib,zipfile,shutil,base64
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.deps'))
from playwright.sync_api import sync_playwright
from PIL import Image
import numpy as np
OUT=HERE/'Krauzlis90_HumanDemo'
ZIP=HERE/'Krauzlis90_HumanDemo.zip'
shutil.copyfile(HERE/'generate.py',OUT/'audit'/'generate.py')
def pack():
    with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(HERE))
pack()
relocated=HERE/'.qa-extracted'
if relocated.exists():shutil.rmtree(relocated)
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    z.extractall(relocated)
TARGET=relocated/OUT.name
results=[]; errors=[];external=[]
with sync_playwright() as p:
    browser=p.chromium.connect_over_cdp('http://127.0.0.1:9224')
    context=browser.new_context(viewport={'width':1100,'height':1050},offline=True)
    page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
    page.goto((TARGET/'index.html').as_uri());page.wait_for_function('!document.getElementById("ready").disabled')
    assert page.evaluate('TRIALS.length')==10
    assert page.locator('#feedback').inner_text()==''
    assert page.locator('#yes').is_disabled() and page.locator('#next').is_disabled()
    assert page.evaluate('document.getElementById("video").autoplay===false && document.getElementById("video").loop===false')
    page.screenshot(path=str(OUT/'audit'/'viewer-ready.png'),full_page=True)
    for i in range(10):
        assert page.locator('#feedback').inner_text()==''
        assert 'Correct answer:' not in page.locator('body').inner_text()
        assert page.locator('#yes').is_disabled()
        # Exercise each actual MP4, fully, with the normal Ready button.
        page.select_option('#mode','mp4');page.click('#ready')
        page.wait_for_function('!document.getElementById("video").paused')
        page.wait_for_function('document.getElementById("video").ended',timeout=10000)
        quality=page.evaluate('({duration:video.duration,frames:video.getVideoPlaybackQuality().totalVideoFrames,ended:video.ended,loop:video.loop})')
        assert quality['ended'] and not quality['loop']
        n=page.evaluate(f'TRIALS[{i}].n')
        assert abs(quality['duration']-n/10)<.011
        assert page.locator('#feedback').inner_text()==''
        # Replay using exact PNG sequence; inspect native cue pixels on first trial.
        page.select_option('#mode','png');page.click('#replay')
        if i==0:
            page.screenshot(path=str(OUT/'audit'/'viewer-cue.png'),full_page=True)
        page.wait_for_function('!document.getElementById("yes").disabled',timeout=10000)
        log=page.evaluate('window.playbackAudit[window.playbackAudit.length-1]')
        assert log['frames']==list(range(n))
        assert page.locator('#feedback').inner_text()==''
        # Compare canvas final raster against native uint8 report image enlarged exactly.
        data=page.evaluate('document.getElementById("screen").toDataURL().split(",")[1]')
        import io
        actual=np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB'))
        native=np.load(TARGET/'native'/f'trial{i+1:02d}.npz')['frames'][-1]
        expected=np.rint(native.transpose(1,2,0)*255).astype(np.uint8).repeat(6,0).repeat(6,1)
        assert np.array_equal(actual,expected)
        page.click('#yes' if i%2==0 else '#no')
        assert 'Correct answer:' in page.locator('#feedback').inner_text()
        assert 'Changed:' in page.locator('#feedback').inner_text()
        assert f'/ {i+1} answered' in page.locator('#score').inner_text()
        assert page.locator('#yes').is_disabled() and page.locator('#no').is_disabled()
        before=page.locator('#score').inner_text()
        page.evaluate('document.getElementById("yes").click()')
        assert page.locator('#score').inner_text()==before
        results.append(dict(id=f'trial{i+1:02d}',png_frames=n,png_frame_order_exact=True,canvas_final_pixels_exact=True,mp4=quality,answer_hidden_until_guess=True,answer_reveal_and_score=True,first_answer_locked=True,png_timing_ms=log['times']))
        if i<9:page.click('#next')
    assert page.locator('#next').is_disabled()
    page.reload();page.wait_for_function('!document.getElementById("ready").disabled')
    assert page.locator('#score').inner_text()=='Score: 0 / 0 answered'
    assert page.locator('#feedback').inner_text()==''
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(OUT/'audit'/'viewer-mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    context.close()
assert not errors and not external
report=dict(passed=True,browser='Chrome via Playwright CDP, real local browser',browser_tool_fallback='browser_exec could not attach: unsupported default profile / missing DevToolsActivePort. Isolated headless Chrome used instead.',file_url=True,network_disabled=True,relocated_zip_tested=True,ten_assets=True,all_trials_png_and_mp4_played=True,no_initial_autoplay=True,no_loop=True,no_answers_before_guess=True,score_and_replay_and_next_verified=True,reload_clears_score=True,mobile_no_horizontal_overflow=True,console_errors=errors,external_requests=external,trials=results)
(OUT/'browser_qa.json').write_text(json.dumps(report,indent=2))
# Repackage QA receipts, then verify archive entries byte-for-byte after extraction.
pack()
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    for f in OUT.rglob('*'):
        if f.is_file():assert z.read(str(f.relative_to(HERE)))==f.read_bytes()
receipt=dict(zip_path=str(ZIP),zip_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),zip_bytes=ZIP.stat().st_size,files=sum(f.is_file() for f in OUT.rglob('*')),mp4_files=len(list((OUT/'videos').glob('*.mp4'))),zip_crc_ok=True,all_archive_files_byte_equal=True)
(HERE/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(passed=True,trials=len(results),errors=errors,external_requests=external,delivery=receipt),indent=2))
