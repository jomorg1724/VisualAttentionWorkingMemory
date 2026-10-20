"""Local, deterministic architecture teaching animation. No training or model inference."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, wave, math, subprocess, shutil, hashlib, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
SCENES=json.loads((ROOT/'script.json').read_text())
W,H,FPS=1280,720,15
BG='#0d1827'; CARD='#17273b'; INK='#f3f4ee'; MUTED='#a9b9ca'
TEAL='#68d6bf'; GOLD='#f2c574'; BLUE='#84b9ff'; PURPLE='#b5a1ee'; CORAL='#f49288'
FONT='C:/Windows/Fonts/segoeui.ttf'; BOLD='C:/Windows/Fonts/segoeuib.ttf'
fonts={}
def font(n,b=False):
    key=(n,b)
    if key not in fonts: fonts[key]=ImageFont.truetype(BOLD if b else FONT,n)
    return fonts[key]
def tx(d,xy,s,n=24,fill=INK,b=False,anchor=None):
    d.text(xy,s,font=font(n,b),fill=fill,anchor=anchor)
def wrapped(d,s,x,y,width,n=24,fill=INK,b=False,gap=7):
    words=s.split(); lines=[]; line=''
    for word in words:
        trial=(line+' '+word).strip()
        if d.textlength(trial,font=font(n,b))>width and line: lines.append(line);line=word
        else: line=trial
    if line:lines.append(line)
    for line in lines:tx(d,(x,y),line,n,fill,b);y+=n+gap
    return y
def card(d,box,title,sub='',color=TEAL):
    x,y,x2,y2=box;d.rounded_rectangle(box,18,fill=CARD)
    d.rounded_rectangle((x,y,x+6,y2),3,fill=color)
    tx(d,(x+22,y+17),title,25,color,True)
    if sub:wrapped(d,sub,x+22,y+59,x2-x-40,21,MUTED)
def arrow(d,points,color=TEAL,t=0,flow=True):
    d.line(points,fill=color,width=3,joint='curve')
    a=np.array(points[-2],float);b=np.array(points[-1],float);v=b-a;v/=max(1,np.linalg.norm(v));p=np.array([-v[1],v[0]])
    d.polygon([tuple(b),tuple(b-12*v+6*p),tuple(b-12*v-6*p)],fill=color)
    if flow:
        lengths=[math.dist(points[j],points[j+1]) for j in range(len(points)-1)]
        z=(t*.3%1)*sum(lengths)
        for j,l in enumerate(lengths):
            if z<=l:
                f=z/max(l,1);x=points[j][0]*(1-f)+points[j+1][0]*f;y=points[j][1]*(1-f)+points[j+1][1]*f
                d.ellipse((x-5,y-5,x+5,y+5),fill=INK);break
            z-=l
def grid(d,x,y,size=210,t=0,kind='memory',cols=10):
    cell=size/cols
    for r in range(cols):
        for c in range(cols):
            if kind=='attention':
                value=math.exp(-((r-3)**2+(c-6)**2)/3)
                col=tuple(int(a+(b-a)*value) for a,b in zip((26,41,61),(242,197,116)))
            else:
                value=.2+.6*(.5+.5*math.sin(r*.8+c*.6+t*.35))
                col=tuple(int(a+(b-a)*value) for a,b in zip((26,41,61),(104,214,191)))
            d.rounded_rectangle((x+c*cell+1,y+r*cell+1,x+(c+1)*cell-2,y+(r+1)*cell-2),2,fill=col)
def vector(d,x,y,width=260,rows=2,t=0,color=BLUE):
    cols=16;cell=width/cols
    for r in range(rows):
        for c in range(cols):
            h=9+15*(.5+.5*math.sin(c*.8+r+t*.4))
            d.rounded_rectangle((x+c*cell,y+r*34+27-h,x+c*cell+cell-5,y+r*34+27),2,fill=color)
def dots(d,x,y,size,t):
    d.rounded_rectangle((x,y,x+size,y+size),12,fill='#253247')
    for i in range(28):
        dx=((i*37+13+t*10)%(size-20))+10;dy=((i*53+7)%(size-20))+10
        d.ellipse((x+dx-2,y+dy-2,x+dx+2,y+dy+2),fill=INK)
def pill(d,xy,s,color=TEAL):
    x,y=xy;w=d.textlength(s,font=font(18,True))+28
    d.rounded_rectangle((x,y,x+w,y+32),16,fill=color)
    tx(d,(x+14,y+4),s,18,BG,True)
def scene_frame(index,local,duration,sentence,overall,total):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    tx(d,(52,29),'ATTENTION / MEMORY / DECISION',17,TEAL,True)
    tx(d,(1228,29),f'{index+1:02d} / 09',17,MUTED,anchor='ra')
    tx(d,(52,72),SCENES[index]['title'],37,INK,True)
    tx(d,(52,125),'Spatial-memory-query model  •  schematic illustration',17,MUTED)
    t=local;u=min(1,local/max(duration,1))
    if index==0:
        card(d,(55,206,275,395),'Visual evidence','Appearance + motion',BLUE);dots(d,85,322,65,t)
        card(d,(355,206,580,395),'Attention','One shared map',GOLD);grid(d,397,300,80,kind='attention')
        card(d,(665,185,940,275),'Spatial memory','6,400 stored numbers',TEAL)
        card(d,(665,292,940,382),'CLS memory','64 stored numbers',BLUE)
        card(d,(665,399,940,489),'Comparison','735 stored numbers',PURPLE)
        card(d,(1020,265,1228,397),'Readout','Task beliefs',CORAL)
        arrow(d,[(275,300),(355,300)],BLUE,t)
        for y,col in [(230,TEAL),(337,BLUE),(444,PURPLE)]:
            arrow(d,[(580,300),(620,300),(620,y),(665,y)],col,t)
        arrow(d,[(940,337),(1020,337)],BLUE,t);arrow(d,[(940,444),(980,444),(980,365),(1020,365)],PURPLE,t)
        tx(d,(55,505),'Spatial memory also feeds attention; the two pooled memories feed the readout.',21,MUTED)
    elif index==1:
        dots(d,65,226,190,t);tx(d,(65,433),'100 × 100 RGB pixels',21,MUTED)
        card(d,(345,195,650,320),'Appearance','Learn color and structure',BLUE);vector(d,370,289,250,1,t,BLUE)
        card(d,(345,345,650,480),'Temporal evidence','Crops at lags 1, 2 and 4',TEAL)
        arrow(d,[(255,320),(300,320),(300,255),(345,255)],BLUE,t)
        arrow(d,[(300,320),(300,410),(345,410)],TEAL,t)
        card(d,(780,196,1218,480),'Fusion + position','64 features per spatial site',GOLD)
        grid(d,805,292,155,t);tx(d,(990,315),'100 sites',27,INK,True);tx(d,(990,359),'× 64 features',24,MUTED)
        arrow(d,[(650,255),(715,255),(715,320),(780,320)],BLUE,t)
        arrow(d,[(650,410),(715,410),(715,350),(780,350)],TEAL,t)
        tx(d,(365,435),'81 candidate shifts per lag',20,MUTED)
        tx(d,(65,513),'Additional inputs: RGB statistics, signal strength, velocity, motion differences and pixel error.',20,MUTED)
    elif index==2:
        grid(d,80,207,265,t);tx(d,(80,489),'10 × 10 sites × 64 channels',23,TEAL,True)
        card(d,(470,202,795,338),'Spatial GRU','3 × 3 neighborhoods + learned gates',TEAL)
        card(d,(470,365,795,492),'Attention write gate','Higher priority permits stronger updates',GOLD)
        card(d,(920,204,1215,492),'Updated map','Retain some contents; write new contents.',TEAL);grid(d,960,320,145,t+2)
        arrow(d,[(345,310),(470,270)],TEAL,t);arrow(d,[(795,270),(920,300)],TEAL,t)
        arrow(d,[(795,425),(855,425),(855,360),(920,360)],GOLD,t)
        tx(d,(480,518),'Normalize gate and candidate signals before sigmoid / tanh.',20,MUTED)
    elif index==3:
        card(d,(65,200,445,408),'CLS recurrent memory h','64 numbers • changes during a trial',BLUE);vector(d,100,310,310,2,t,BLUE)
        card(d,(65,432,445,526),'Learned CLS token c','64 parameters • fixed during a trial',PURPLE)
        tx(d,(495,344),'+',45,INK,True)
        card(d,(580,268,895,446),'Normalize + project','LayerNorm → learned query projection',GOLD)
        card(d,(1010,290,1215,422),'Query','64 numbers',GOLD)
        arrow(d,[(445,298),(540,298),(540,325),(580,325)],BLUE,t)
        arrow(d,[(445,480),(540,480),(540,380),(580,380)],PURPLE,t)
        arrow(d,[(895,350),(1010,350)],GOLD,t)
        tx(d,(580,483),'The newer model also adds a spatial-memory contribution.',20,MUTED)
    elif index==4:
        card(d,(55,190,370,310),'Previous CLS + token','One 64-number contribution',BLUE)
        card(d,(55,350,370,493),'Previous spatial memory','6,400 → 64 learned projection',TEAL)
        card(d,(480,260,705,420),'One query','Sum contributions; unit normalize',GOLD)
        arrow(d,[(370,250),(425,250),(425,310),(480,310)],BLUE,t)
        arrow(d,[(370,420),(425,420),(425,370),(480,370)],TEAL,t)
        card(d,(835,190,1225,315),'100 keys','Current visual + local memory → matching',GOLD)
        card(d,(835,354,1225,493),'100 values','Current visual + local memory → content',PURPLE)
        arrow(d,[(705,312),(835,252)],GOLD,t)
        tx(d,(490,480),'Compare query with keys.',22,GOLD,True)
        tx(d,(55,518),'The added spatial route changes the query source; it does not add more queries.',20,MUTED)
    elif index==5:
        card(d,(55,195,400,410),'Query · each key','Cosine similarity → scale → softmax',GOLD)
        tx(d,(83,318),'100 scores → 100 weights',22,INK,True)
        tx(d,(83,366),'Sum of weights = 1',23,GOLD,True)
        grid(d,490,210,210,t,kind='attention');tx(d,(490,440),'One 10 × 10 map',23,GOLD,True)
        arrow(d,[(400,300),(490,300)],GOLD,t)
        for y,title,sub,col in [(185,'Pool values','Update CLS memory',BLUE),(310,'Modulate writes','Update spatial memory',TEAL),(435,'Pool comparisons','Update comparison memory',PURPLE)]:
            card(d,(850,y,1225,y+98),title,sub,col)
            arrow(d,[(700,310),(770,310),(770,y+49),(850,y+49)],col,t)
        tx(d,(55,480),'Each weight multiplies every channel',22,MUTED)
        tx(d,(55,511),'of that location’s selected value.',22,MUTED)
    elif index==6:
        tx(d,(70,192),'EARLIER OBSERVATIONS',20,MUTED,True);tx(d,(915,192),'CURRENT FRAME t',20,INK,True)
        for j in range(16):
            x=70+j*69;col=PURPLE if j<8 else (TEAL if j>=12 else '#314056')
            d.rounded_rectangle((x,244,x+58,304),7,fill=col)
            tx(d,(x+29,314),str(j-15),17,MUTED,anchor='ma')
        tx(d,(70,350),'Older reference: up to 8 frames',23,PURPLE,True)
        tx(d,(650,350),'4-frame gap',21,MUTED);tx(d,(920,350),'Recent: 4 frames',22,TEAL,True)
        card(d,(65,411,485,528),'Local comparisons','Reference, recent signal, and differences',PURPLE)
        card(d,(700,411,1225,528),'Comparison memory z','z ← 0.8 z + 0.2 selected comparison',PURPLE)
        arrow(d,[(485,470),(700,470)],GOLD,t)
        tx(d,(513,433),'Attention',18,GOLD,True)
    elif index==7:
        card(d,(55,194,340,304),'CLS memory','64 numbers',BLUE)
        card(d,(55,357,340,467),'Comparison memory','735 numbers',PURPLE)
        card(d,(460,255,740,429),'Shared readout','799 → 64 features',CORAL)
        arrow(d,[(340,250),(400,250),(400,310),(460,310)],BLUE,t)
        arrow(d,[(340,410),(400,410),(400,370),(460,370)],PURPLE,t)
        card(d,(865,208,1225,452),'Task head → beliefs','Illustrative four-choice probabilities',CORAL)
        probs=[.1,.7,.12,.08]
        for j,p in enumerate(probs):
            x=897+j*78;d.rounded_rectangle((x,394-p*130,x+43,394),4,fill=GOLD if j==1 else BLUE)
            tx(d,(x+21,406),f'{round(p*100)}%',18,INK,anchor='ma')
        arrow(d,[(740,340),(865,340)],CORAL,t)
        arrow(d,[(1100,477),(1100,531),(170,531),(170,477)],TEAL,t)
        tx(d,(315,494),'Next frame: spatial + CLS memory influence attention again',23,TEAL,True)
    elif index==8:
        card(d,(65,194,610,377),'Correct terminal answer','Cross-entropy loss; VDA uses value / 3 weighting',CORAL)
        card(d,(675,194,1220,377),'Predict pixels using motion','Transport error × 0.05',TEAL)
        tx(d,(87,309),'“Increase probability of the right answer.”',22,INK)
        tx(d,(697,309),'“Make motion explain image changes.”',22,INK)
        card(d,(300,436,985,531),'Backpropagation + Adam','Shared learning signals update connected parameters.',GOLD)
        arrow(d,[(335,377),(335,405),(500,405),(500,436)],CORAL,t)
        arrow(d,[(950,377),(950,405),(800,405),(800,436)],TEAL,t)
    # Burned-in synchronized sentence, with deterministic wrapping.
    d.rectangle((0,558,W,687),fill='#101f31')
    wrapped(d,sentence,64,579,1152,25,INK,gap=8)
    tx(d,(52,695),SCENES[index]['short'],15,MUTED)
    tx(d,(1228,695),'Source-checked diagram • illustrative activity',15,MUTED,anchor='ra')
    d.rectangle((0,714,W,719),fill='#22354c')
    d.rectangle((0,714,int(W*overall/total),719),fill=TEAL)
    return im

def srt_time(t):
    ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'
def build_audio():
    segments=[];chunks=[];offset=0;fmt=None;chapters=[]
    for i,scene in enumerate(SCENES):
        start=offset;parts=[]
        for j,s in enumerate(scene['sentences']):
            with wave.open(str(ROOT/'voice'/f'{i:02}-{j:02}.wav'),'rb') as w:
                f=(w.getnchannels(),w.getsampwidth(),w.getframerate())
                if fmt is None:fmt=f
                assert fmt==f
                audio=w.readframes(w.getnframes());sec=w.getnframes()/fmt[2]
            pad=.35 if j<len(scene['sentences'])-1 else 1.1
            parts.append({'start':offset,'end':offset+sec+pad,'text':s})
            chunks.extend([audio,b'\x00'*round(pad*fmt[2])*fmt[0]*fmt[1]])
            offset+=sec+pad
        segments.append({'start':start,'end':offset,'parts':parts})
        chapters.append({'title':scene['short'],'start':start,'end':offset})
    with wave.open(str(ROOT/'narration.wav'),'wb') as w:
        w.setnchannels(fmt[0]);w.setsampwidth(fmt[1]);w.setframerate(fmt[2]);w.writeframes(b''.join(chunks))
    (ROOT/'chapters.json').write_text(json.dumps(chapters,indent=2))
    subs=[]
    for seg in segments:
        for p in seg['parts']:
            subs.append(f"{len(subs)+1}\n{srt_time(p['start'])} --> {srt_time(p['end'])}\n{p['text']}\n")
    (ROOT/'captions.srt').write_text('\n'.join(subs),encoding='utf8')
    return segments,offset,chapters

def main():
    segments,total,chapters=build_audio()
    (ROOT/'stills').mkdir(exist_ok=True)
    for i,seg in enumerate(segments):
        local=(seg['end']-seg['start'])*.5
        scene_frame(i,local,seg['end']-seg['start'],seg['parts'][1]['text'],seg['start']+local,total).save(ROOT/'stills'/f'{i+1:02}.png')
    poster=scene_frame(0,2,20,'Spatial memory, CLS memory and comparison memory each play a different role.',0,total)
    poster.save(ROOT/'poster.png')
    ffmpeg=shutil.which('ffmpeg')
    cmd=[ffmpeg,'-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-i',str(ROOT/'narration.wav'),'-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(ROOT/'architecture-explainer.mp4')]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=(ROOT/'ffmpeg.log').open('w'))
    index=0
    for frame in range(math.ceil(total*FPS)):
        now=frame/FPS
        while index<len(segments)-1 and now>=segments[index]['end']:index+=1
        seg=segments[index]
        line=next((v['text'] for v in seg['parts'] if now<v['end']),seg['parts'][-1]['text'])
        im=scene_frame(index,now-seg['start'],seg['end']-seg['start'],line,now,total)
        p.stdin.write(im.tobytes())
        if frame%(FPS*20)==0:print(f'Rendered {now:.0f} / {total:.0f} seconds',flush=True)
    p.stdin.close()
    if p.wait()!=0:raise RuntimeError((ROOT/'ffmpeg.log').read_text())
    html=(ROOT/'player-template.html').read_text(encoding='utf8').replace('__CHAPTERS__',json.dumps(chapters))
    (ROOT/'index.html').write_text(html,encoding='utf8')
    (ROOT/'build.json').write_text(json.dumps({'seconds':total,'frames':math.ceil(total*FPS),'fps':FPS,'width':W,'height':H,'scenes':len(SCENES),'voice':'Microsoft Zira Desktop','illustrative_activity':True,'model':'spatial_memory_query_v1','trained_model_inference':False,'sha256':hashlib.sha256((ROOT/'architecture-explainer.mp4').read_bytes()).hexdigest()},indent=2))
    print('Complete',round(total,1),'seconds',flush=True)
if __name__=='__main__':main()
