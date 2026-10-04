from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
HERE=Path(__file__).resolve().parent
OUT=HERE/'tmp/pdfs'
INK=HexColor('#163B4A'); TEAL=HexColor('#E4F0EF'); GOLD=HexColor('#FFF1CC'); GRAY=HexColor('#EDF0F4'); MUTED=HexColor('#54717B')
def label(c,x,y,lines,size=10,color=INK):
 c.setFillColor(color);c.setFont('Helvetica',size)
 for i,line in enumerate(lines):c.drawCentredString(x,y-i*(size+4),line)
def box(c,x,y,w,h,lines,fill=TEAL,size=10):
 c.setFillColor(fill);c.setStrokeColor(MUTED);c.roundRect(x,y,w,h,5,fill=1,stroke=1)
 label(c,x+w/2,y+h/2+(len(lines)-1)*(size+4)/2-size/3,lines,size)
def arrow(c,x,y,xx,yy):
 c.setStrokeColor(INK);c.setFillColor(INK);c.setLineWidth(1);c.line(x,y,xx,yy)
 import math
 a=math.atan2(yy-y,xx-x);p=c.beginPath();p.moveTo(xx,yy)
 p.lineTo(xx-5*math.cos(a-.45),yy-5*math.sin(a-.45));p.lineTo(xx-5*math.cos(a+.45),yy-5*math.sin(a+.45));p.close();c.drawPath(p,fill=1,stroke=0)
c=canvas.Canvas(str(OUT/'architecture_routes.pdf'),pagesize=(520,460))
box(c,80,399,360,44,['Shared CNN + three spatial KDA fields','Centered stack of current and two past RGB frames'])
arrow(c,260,399,260,375);box(c,155,340,210,35,['Deepest map: 160 x 7 x 7'])
arrow(c,210,340,127,308);arrow(c,310,340,393,308)
label(c,127,288,['IMPLEMENTED'],10);label(c,393,288,['PLANNED DESIGN'],10)
box(c,20,220,215,50,['Flatten 7,840 coordinates','Linear + ReLU -> 256'],GRAY)
box(c,285,220,215,50,['1 x 1 spatial projection','64 x 7 x 7'],GRAY)
arrow(c,127,220,127,195);arrow(c,393,220,393,195)
box(c,20,140,215,55,['Vector GRU','256 recurrent scalars'],GOLD)
box(c,285,140,215,55,['3 x 3 ConvGRU','64 x 7 x 7 recurrent state'],GOLD)
arrow(c,127,140,127,113);arrow(c,393,140,393,113)
box(c,20,61,215,52,['Final state -> selected head','Compression before recurrence'],GRAY)
box(c,285,61,215,52,['Final map -> flatten -> 256 -> head','Compression after recurrence'],GRAY,size=9)
label(c,260,26,['Gold = trial-local recurrent state. Both routes also contain KDA state upstream.'],9)
c.save()
c=canvas.Canvas(str(OUT/'kda_step.pdf'),pagesize=(520,225))
label(c,260,208,['ONE SITE, ONE HEAD: FOUR OPERATIONS'],11)
for x,title,lines in [(5,'1. RETAIN',['Old S -> row-wise decay','Sbar = D(alpha) S']),(137,'2. PREDICT',['Read with current key','prediction = Sbar^T k']),(269,'3. CORRECT',['error = v - prediction','S = Sbar + beta k error^T']),(401,'4. READ',['Read updated memory','output = S^T q'])]:
 box(c,x,100,114,73,[title]+lines,GOLD if x in (5,269) else TEAL,size=8)
for x in [119,251,383]:arrow(c,x,136,x+18,136)
label(c,260,69,['Key asks: what do I already predict for this association?', 'Query asks: what should downstream processing receive now?'],10)
label(c,260,20,['The current write occurs before the current read. Keys and queries may differ.'],9)
c.save()
c=canvas.Canvas(str(OUT/'trial_timelines.pdf'),pagesize=(520,324))
label(c,260,307,['TRIAL STRUCTURE: PRESENTED FRAMES'],11)
rows=[('Ring orientation',[('Cue',1),('Sample',2),('Probe',1)],'4'),('Signed orientation',[('Cue',1),('Sample',2),('Blank', 'D'),('Probe',1)],'D + 4'),('Motion duration',[('Cue',1),('Ref',1),('Motion',8),('Blank','D'),('Report',1)],'D + 11'),('Krauzlis change',[('Cue',2),('Fix',5),('Ref',1),('Base','B'),('Post',8),('Report',1)],'B + 17'),('Binding',[('Instr',1),('Sample',2),('Blank','D'),('Query',1),('Probe',1)],'D + 5'),('Recognition',[('Instr',1),('Study','N'),('Blank',3),('Probe','H')],'N + 4 + H')]
for j,(name,items,total) in enumerate(rows):
 y=252-j*39;c.setFont('Helvetica',9);c.setFillColor(INK);c.drawString(5,y+12,name)
 x=115;w=49
 for k,(lab,num) in enumerate(items):
  box(c,x+k*(w+3),y,w,31,[lab,str(num)],GOLD if lab in ('Blank','Fix') else TEAL,size=8)
 c.setFont('Helvetica',8);c.setFillColor(MUTED);c.drawRightString(518,y+11,total)
label(c,260,9,['Schematic block widths are not durations. Numbers in blocks are frame counts.'],9)
c.save()
