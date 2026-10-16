"""CPU-only frozen best-policy recordings; complete episodes on prespecified seeds."""
import hashlib,json,sys,subprocess,time,faulthandler,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from SecondPass.DeadlyCorridor.agent import Agent
from SecondPass.DeadlyCorridor.environment import Corridor,BUTTONS,ACTIONS

BASE=Path(__file__).resolve().parent
REC=BASE/'recovered'
TRAIN=REC/'vawm/SecondPass/DeadlyCorridor/CloudRuntime/continuation01/training'
OUT=BASE/'playback'
FONT=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
SMALL=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def canvas(frame,title,subtitle,footer):
    im=Image.new('RGB',(640,576),'#101827');d=ImageDraw.Draw(im)
    d.text((14,9),title,font=FONT,fill='#fafafa');d.text((14,34),subtitle,font=SMALL,fill='#b7c5d8')
    im.paste(Image.fromarray(frame).resize((640,480),Image.Resampling.NEAREST),(0,64))
    d.text((14,550),footer,font=SMALL,fill='#d3e5fb');return np.asarray(im)

def writer(path):
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x576','-r','35','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

def main():
    faulthandler.dump_traceback_later(35,repeat=True)
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    bestpath=TRAIN/'best.pt';before=sha(bestpath);saved=torch.load(bestpath,map_location='cpu',weights_only=False)
    agent=Agent().eval();agent.load_state_dict(saved['agent']);agent.requires_grad_(False)
    expected=saved['config']['source_sha256']
    for name,h in expected.items():assert sha(Path('SecondPass/DeadlyCorridor')/name)==h,name
    env=Corridor();assert env.provenance['wad_sha256']==saved['config']['environment']['wad_sha256']
    episodes=[];combined=writer(OUT/'six-games.mp4')
    try:
        for number,seed in enumerate([212101,212102,212103,222101,222102,222103],1):
            # A replay changes engine mode/RNG state. Use a fresh engine before
            # the next policy episode so playback cannot influence collection.
            if number>1:env.close();env=Corridor(seed=seed)
            cohort='Validation seed' if number<=3 else 'Fresh seed'
            demo=OUT/f'game-{number}.lmp';env.game.set_seed(seed)
            print('START_GAME',number,flush=True)
            # ViZDoom's console command parser requires a simple recording name.
            env.game.new_episode(f'codex_doom_game_{number}.lmp');obs=env.observe()
            print('ENGINE_READY',number,flush=True)
            memory=None;total=0.;actions=[];returns=[];ticks=[]
            with torch.inference_mode():
                while True:
                    action,memory,_=agent.act(torch.from_numpy(obs).float()[None]/255,memory,True)
                    obs,r,done,info=env.step(action);total+=r;actions.append(action);returns.append(total);ticks.append(info['episode_tics'])
                    if done:break
            row=dict(game=number,seed=seed,cohort=cohort,raw_return=total,decisions=len(actions),tics=info['episode_tics'],dead=info['dead'],actions=actions,cumulative_returns=returns,decision_end_tics=ticks)
            # Replay the recorded engine demo to render every game tic, including
            # the three intermediate tics skipped by the policy.
            print('EPISODE_COMPLETE',number,total,flush=True)
            env.game.replay_episode(f'codex_doom_game_{number}.lmp');video=writer(OUT/f'game-{number}.mp4');frames=0;last=None
            while not env.game.is_episode_finished():
                state=env.game.get_state()
                if state is not None:
                    tick=env.game.get_episode_time();idx=min(int(np.searchsorted(ticks,tick,side='right')),len(actions)-1)
                    active=[name.lower().replace('_',' ') for name,on in zip(BUTTONS,ACTIONS[actions[idx]]) if on]
                    footer=' + '.join(active) or 'No buttons'
                    last=canvas(state.screen_buffer.copy(),f'Game {number} | {cohort} {seed}',f'Best checkpoint {saved["steps"]:,} | Greedy policy | Native skill 5',footer)
                    if frames==0:
                        Image.fromarray(last).save(OUT/f'game-{number}.jpg')
                        for _ in range(35):combined.stdin.write(last.tobytes())
                    video.stdin.write(last.tobytes());combined.stdin.write(last.tobytes());frames+=1
                env.game.advance_action(1)
            video.stdin.close();assert video.wait()==0
            source_demo=Path(f'codex_doom_game_{number}.lmp')
            if source_demo.exists():source_demo.replace(demo)
            if last is not None:
                end=Image.fromarray(last);d=ImageDraw.Draw(end);d.rectangle((0,544,640,576),fill='#101827')
                d.text((14,550),f'Final reward {total:+.1f} | '+('Died' if info['dead'] else 'Episode ended'),font=FONT,fill='#ffffff')
                for _ in range(70):combined.stdin.write(np.asarray(end).tobytes())
            row['video_frames']=frames;row['video_seconds']=frames/35;episodes.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['actions','cumulative_returns','decision_end_tics']}),flush=True)
    finally:
        env.close();combined.stdin.close();assert combined.wait()==0
    assert sha(bestpath)==before
    report=dict(checkpoint_step=saved['steps'],checkpoint_sha256=before,selection_validation=saved['validation'],policy='greedy argmax',inference_device='cpu',action_repeat=4,video_fps=35,video_method='Native engine demo replay, every tic',environment=env.provenance,training_environment=saved['config']['environment'],episodes=episodes,checkpoint_unchanged=True)
    (OUT/'recordings.json').write_text(json.dumps(report,indent=2))
    cards=''.join(f'<article><h2>Game {r["game"]} · {r["cohort"]}</h2><video controls preload="metadata" poster="game-{r["game"]}.jpg" src="game-{r["game"]}.mp4"></video><p>Seed {r["seed"]} · reward <b>{r["raw_return"]:+.1f}</b> · {r["tics"]/35:.1f} game seconds · {"died" if r["dead"] else "episode ended"}</p></article>' for r in episodes)
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Doom · trained agent playback</title><style>body{background:#0c1421;color:#e7effa;font:18px system-ui;margin:0 auto;padding:36px;max-width:1180px}h1{font-size:42px}p{line-height:1.6;color:#bbcce0}video{width:100%;background:#000;border-radius:12px}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{padding:20px;background:#172438;border-radius:16px}h2{font-size:23px}.hero{max-width:760px;margin:25px auto}button{padding:10px 18px;border:1px solid #6483a6;border-radius:8px;background:#263c57;color:white;font-size:16px;cursor:pointer}a{color:#8edbfa}@media(max-width:750px){main{grid-template-columns:1fr}}</style></head><body><h1>Watch the trained Doom agent</h1>'''
    html+=f'<p>Best checkpoint: <b>{saved["steps"]:,} decisions</b> · validation average reward <b>{saved["validation"]["mean_return"]:+.1f}</b> across ten episodes. Six complete games below use the greedy policy. The first three use validation seeds; the next three use fresh seeds. Every game is included in the order specified before recording.</p>'
    html+='<p>The agent chooses an action every four game tics. These videos replay the actual engine recordings at 35 frames per second, showing all intermediate motion. The view matches the training settings: weapon, crosshair, and HUD hidden. Use slow playback to inspect these short episodes.</p><p><button onclick="speed(1)">Real time</button> <button onclick="speed(.5)">Half speed</button> <button onclick="speed(.25)">Quarter speed</button></p><div class="hero"><h2>All six games</h2><video controls preload="metadata" poster="game-1.jpg" src="six-games.mp4"></video></div><main>'+cards+'</main><p><a href="recordings.json">Episode details and provenance</a> · <a href="six-games.mp4" download>Download all six games</a></p><script>function speed(r){document.querySelectorAll("video").forEach(v=>v.playbackRate=r)}document.querySelectorAll("video").forEach(v=>v.addEventListener("play",()=>document.querySelectorAll("video").forEach(w=>{if(v!==w)w.pause()})))</script></body></html>'
    (OUT/'index.html').write_text(html,encoding='utf8');faulthandler.cancel_dump_traceback_later();print('PLAYBACK_READY',flush=True)

if __name__=='__main__':main()
