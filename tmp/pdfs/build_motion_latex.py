"""Emit a standalone LaTeX source: equations, tables and figures are native TeX."""
from pathlib import Path
import re,json
from bs4 import BeautifulSoup, NavigableString
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf'
data=json.loads((OUT/'optic_flow_pathway_verification.json').read_text())
observed=data['observed'];maxerr=data['fft_direct_max_abs_error']
TEAL='teal';parts=[];sections=0;eqno=0
mapping={'α':r'\(\alpha\)','β':r'\(\beta\)','γ':r'\(\gamma\)','δ':r'\(\delta\)','ε':r'\(\epsilon\)','η':r'\(\eta\)','θ':r'\(\theta\)','τ':r'\(\tau\)','φ':r'\(\phi\)','ω':r'\(\omega\)','π':r'\(\pi\)','ℓ':r'\(\ell\)','Δ':r'\(\Delta\)','Ω':r'\(\Omega\)','̃':r'\textasciitilde{}','̂':r'\textasciicircum{}','–':'--','—':'---','−':'-','×':r'\(\times\)','→':r'\(\to\)','…':r'\ldots{}','’':"'",'‘':"'",'“':'``','”':"''"}
def esc(s):
    # Page references in the former fixed-page layout become stable section references.
    s=re.sub(r'([Pp])ages? (\d+)-(\d+)',lambda m:('Sections' if m[1]=='P' else 'sections')+' '+str(int(m[2])-1)+'-'+str(int(m[3])-1),s)
    s=re.sub(r'([Pp])age (\d+)',lambda m:('Section' if m[1]=='P' else 'section')+' '+str(int(m[2])-1),s)
    s=s.replace('Î','IHATPLACEHOLDER').replace('C̃','CTILDEPLACEHOLDER')
    chars={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}
    s=''.join(mapping.get(c,chars.get(c,c)) for c in s)
    s=s.replace('/',r'/\allowbreak{}')
    return s.replace('IHATPLACEHOLDER',r'\(\widehat I\)').replace('CTILDEPLACEHOLDER',r'\(\widetilde C\)')
def html(s):
    def walk(n):
        if isinstance(n,NavigableString):return esc(str(n))
        body=''.join(walk(x) for x in n.children)
        if n.name in ('b','strong'):return r'\textbf{'+body+'}'
        if n.name in ('i','em'):return r'\emph{'+body+'}'
        if n.name=='sub':return r'\textsubscript{'+body+'}'
        if n.name in ('super','sup'):return r'\textsuperscript{'+body+'}'
        if n.name=='br':return r'\par '
        if n.name in ('link','a'):return r'\href{'+n.get('href','')+'}{'+body+'}'
        return body
    return walk(BeautifulSoup(s.replace('<link ','<a ').replace('</link>','</a>'),'html.parser'))
def P(s):parts.append(html(s)+'\n\n')
def H(s):parts.append(r'\subsection*{'+html(s)+'}\n')
def small(s):parts.append('{\n'+r'\small\color{muted} '+html(s)+'\par}\n\n')
def page(title,sub):
    global sections
    if title.startswith('19. References'):
        constants_appendix()
    if sections:parts.append(r'\clearpage'+'\n')
    if sections==0:
        parts.append(r'\phantomsection\addcontentsline{toc}{section}{Reading guide}\label{guide}'+'\n')
        parts.append(r'{\color{teal}\small\sffamily IMPLEMENTATION WALKTHROUGH\par}\medskip'+'\n')
        parts.append(r'{\sffamily\bfseries\fontsize{27}{32}\selectfont '+html(title)+r'\par}\bigskip'+'\n')
    else:
        title=re.sub(r'^\d+\.\s*','',title)
        parts.append(r'\section{'+html(title)+'}'+r'\label{sec'+str(sections)+'}\n')
    sections+=1;P('<b>'+sub+'</b>')
def E(tex):
    global eqno
    eqno+=1
    # Split the two unusually wide displays across aligned rows; all symbols stay native math.
    if tex.startswith(r'\omega_'):
        a,b=tex.split(r',\qquad ');tex=r'\begin{aligned}'+a+r',\\[4pt]'+b+r'.\end{aligned}'
    parts.append(r'\begin{equation}'+tex+r'\end{equation}'+'\n')
def table(rows,widths):
    spec=''.join('p{'+f'{w/sum(widths)*.90:.3f}'+r'\linewidth}' for w in widths)
    parts.append(r'\begin{center}\small\renewcommand{\arraystretch}{1.16}\setlength{\tabcolsep}{4pt}\begin{tabular}{'+spec+r'}\toprule'+'\n')
    for i,row in enumerate(rows):
        parts.append(' & '.join((r'\textbf{'+html(str(x))+'}') if i==0 else html(str(x)) for x in row)+r' \\'+'\n')
        if i==0:parts.append(r'\midrule'+'\n')
    parts.append(r'\bottomrule\end{tabular}\end{center}'+'\n')

def constants_appendix():
    page('Where the spatial numbers come from','Separate chosen settings, derived counts, and quantities that training adjusts.')
    P('A fixed setting specifies the calculation we elected to perform. A derived number follows arithmetically from those settings. A learned value is adjusted by optimization, although its initialization and bounds can still be chosen. Stating these categories makes the method reproducible and makes each design decision available for testing.')
    table([['Number','Origin','Reason and consequence'],
    ['100 by 100 input','Chosen input format','The present backbone accepts this resolution. It establishes the coordinate scale for every pixel displacement.'],
    ['Stride 10; 10 by 10 sites; 100 sites','10 chosen; grid dimensions derived','Stride sets spacing. With the implemented padding and crop size, floor((100+14-24)/10)+1 = 10 sites per axis; 10 squared = 100.'],
    ['24 by 24 crops','Chosen spatial support','Larger crops supply more correspondence evidence; smaller crops provide more local measurements.'],
    ['Padding 7 per side','Alignment choice consistent with crop and stride','(24-10)/2=7 aligns crop centers with the 10-pixel blocks. Neighboring crops share 24-10=14 columns.'],
    ['Radius 4; nine offsets; 81 pairs','4 chosen; 9 and 81 derived','Search extent is limited to four pixels per component. Nine integer offsets include both endpoints and zero; nine squared = 81.'],
    ['FFT side length 32','Chosen computational size','Power-of-two size above 24. Its eight zero-padded positions per axis suffice for exact finite-crop correlation at retained shifts up to four pixels.'],
    ['Hann denominator 23','Derived from 24 samples','Indices 0 through 23 place both endpoints at zero in the nonperiodic Hann formula.'],
    ['Integer shift spacing 1 pixel','Chosen candidate discretization','Softmax expectations can be fractional even though individual candidates lie on this integer grid.']], [110,120,260])
    P('The earlier CorrespondencePredictiveMemory implementation used 50-pixel crops, a 25-pixel stride, FFT side 64 and radius 10, giving 441 displacement pairs. The current values belong to a different local representation. That historical difference documents a design change; the inspected notes do not establish that radius 4 is an optimum.')
    page('Where the temporal and scale numbers come from','Their computational effects are explicit; their particular values are design settings.')
    table([['Setting','Origin','What the setting controls'],
    ['Lags 1, 2, 4','Chosen frame intervals','Samples different temporal separations. Maximum represented component speed is R divided by lag: 4, 2 and 1 pixels per frame at R=4.'],
    ['Recent width 4; reference width 8; gap parameter 8','Chosen temporal windows','Compares recent evidence with an older baseline. The four intervening frames follow from 8 excluded recent frames minus the four-frame recent window.'],
    ['Trace rate initially 0.1','Chosen initialization; learned thereafter','Initial adaptation is 10% per reliable observation. The actual update also multiplies by reliability.'],
    ['Comparison memory decay 0.8; update 0.2','0.8 chosen; 0.2 = 1-0.8','Retains 80% of the previous comparison memory and adds 20% of the current selected vector.'],
    ['Temperature initially 0.04; bounds 0.015-0.2','Initialization and bounds chosen; temperature learned','Sets the sensitivity of softmax weights to score differences.'],
    ['Refinement multiplier 0.1; population multiplier 10','Chosen feature scales','Scales the learned correction and the population features. These coefficients specify the relative magnitudes presented to subsequent computation.'],
    ['Energy offsets 0.01 and 0.0001','Chosen reliability scales','A ratio z/(z+c) equals one half when z=c. The constants set that half-weight energy scale for two different energy definitions.'],
    ['Floors 1e-8, 1e-6, 1e-4','Chosen numerical safeguards','Prevent tiny denominators in correlation/cosine, temporal averaging and pixel error. Each floor belongs to its explicitly stated equation.'],
    ['Pixel loss multiplier 0.05','Chosen objective weight','Sets the contribution of image prediction relative to the behavioral loss.']], [112,122,256])
    P('These statements explain the role of each choice. Evidence that one value is preferable would require a controlled comparison, keeping data, compute and evaluation consistent. The present implementation supplies the values and their effects; it does not contain an optimization study of every constant.')

# Entirely self-contained native diagrams; no \includegraphics and no image dependencies.
figures={
'crops':r'''\begin{tikzpicture}[font=\small]
\begin{scope}[x=.033cm,y=-.033cm]
\node[anchor=west,font=\small\bfseries] at (0,-17){A. Location in the image};
\foreach \v in {0,10,...,100}{\draw[gray!25] (\v,0)--(\v,100);\draw[gray!25] (0,\v)--(100,\v);}
\foreach \r in {0,...,9}{\foreach \c in {0,...,9}{\fill[gray!55] (10*\c+4.5,10*\r+4.5) circle (.55pt);}}
\fill[teal!15] (32.5,32.5) rectangle (56.5,56.5);
\fill[orange!15] (42.5,32.5) rectangle (66.5,56.5);
\fill[violet!30] (42.5,32.5) rectangle (56.5,56.5);
\draw[teal,thick] (32.5,32.5) rectangle (56.5,56.5);
\draw[orange,thick,dashed] (42.5,32.5) rectangle (66.5,56.5);
\fill[teal] (44.5,44.5) circle (1.3pt);\fill[orange] (54.5,44.5) circle (1.3pt);
\node[align=center] at (50,116){100 by 100 pixels\\100 site centers (dots)};
\draw[->,gray] (68,44.5)--(140,44.5) node[midway,above,font=\scriptsize]{enlarge};
\end{scope}
\begin{scope}[xshift=5.2cm,x=.23cm,y=-.23cm]
\node[anchor=west,font=\small\bfseries] at (0,-2.45){B. Two adjacent 24 by 24 crops};
\fill[teal!12] (0,0) rectangle (10,24);
\fill[violet!23] (10,0) rectangle (24,24);
\fill[orange!12] (24,0) rectangle (34,24);
\foreach \v in {0,...,34}{\draw[gray!30,very thin] (\v,0)--(\v,24);}
\foreach \v in {0,...,24}{\draw[gray!30,very thin] (0,\v)--(34,\v);}
\draw[teal,line width=1.2pt] (0,0) rectangle (24,24);
\draw[orange,line width=1.2pt,dashed] (10,0) rectangle (34,24);
\node[align=center,font=\scriptsize,text=teal] at (4.8,5){Only\\crop A};
\node[align=center,font=\small\bfseries,text=violet] at (17,5){Shared\\pixels};
\node[align=center,font=\scriptsize,text=orange] at (29,5){Only\\crop B};
\fill[teal] (12,12) circle (2pt);\node[above left,text=teal] at (12,12){A};
\fill[orange] (22,12) circle (2pt);\node[above right,text=orange] at (22,12){B};
\draw[gray,dotted] (12,12)--(12,16);\draw[gray,dotted] (22,12)--(22,16);
\draw[<->,thick] (12,16)--(22,16) node[midway,below,fill=white,inner sep=1pt,font=\scriptsize]{10-pixel stride};
\draw[<->,violet,thick] (10,21)--(24,21) node[midway,above,fill=white,inner sep=1pt,font=\scriptsize]{14 shared columns};
\node[anchor=east,font=\scriptsize] at (-.8,.5){33};
\node[anchor=east,font=\scriptsize] at (-.8,23.5){56};
\foreach \x/\lab in {.5/33,10.5/43,23.5/56,33.5/66}{\node[below,font=\scriptsize] at (\x,24.5){\lab};}
\node[font=\scriptsize] at (17,28){Image column index $x$ (one square = one pixel)};
\node[anchor=west,text=teal,font=\scriptsize] at (0,31){A: solid border; site $(4,4)$; center $(44.5,44.5)$};
\node[anchor=west,text=orange,font=\scriptsize] at (0,33){B: dashed border; site $(4,5)$; center $(54.5,44.5)$};
\end{scope}\end{tikzpicture}''',
'window':r'''\begin{tikzpicture}
\begin{axis}[width=.47\linewidth,height=4.0cm,xmin=0,xmax=23,ymin=0,ymax=1.05,xlabel={Crop coordinate $n$},ylabel={Weight},title={Hann taper}]
\addplot[teal,thick,domain=0:23,samples=80]{.5*(1-cos(deg(2*pi*x/23)))};
\end{axis}
\begin{scope}[xshift=7.0cm]
\foreach \a in {0,...,23}{\foreach \b in {0,...,23}{\pgfmathsetmacro{\shade}{100*.5*(1-cos(360*\a/23))*.5*(1-cos(360*\b/23))}\fill[teal!\shade!white] (\a*.12,\b*.12) rectangle ++(.12,.12);}}
\node[font=\small] at (1.44,3.3){2D weight $w(a)w(b)$};
\node[font=\small,align=center] at (1.44,-.5){Light: near zero\\Dark: near one};
\end{scope}\end{tikzpicture}''',
'population':r'''\begin{tikzpicture}
\begin{axis}[width=.47\linewidth,height=4.3cm,ybar,bar width=5pt,ymin=0,ymax=1.1,xtick={0,1,2},xticklabels={$.2$,$.1$,$0$},xlabel={Score},ylabel={Weight},title={Three alternatives},legend style={font=\tiny,at={(.99,.99)},anchor=north east}]
\addplot coordinates{(0,.50648)(1,.3072)(2,.18632)};\addlegendentry{$\tau=.2$}
\addplot coordinates{(0,.66524)(1,.24473)(2,.09003)};\addlegendentry{$\tau=.1$}
\addplot coordinates{(0,.91842)(1,.07539)(2,.00619)};\addlegendentry{$\tau=.04$}
\end{axis}
\begin{axis}[at={(7.0cm,0)},width=.44\linewidth,height=4.3cm,ymin=0,ymax=1,xmin=-4,xmax=4,xlabel={Displacement},title={Two alternatives},ytick={0,.5,1}]
\addplot[ybar,bar width=10pt,fill=teal,draw=teal]coordinates{(-3,.5)(3,.5)};
\addplot[orange,dashed]coordinates{(0,0)(0,1)};
\node[font=\scriptsize,anchor=north] at (axis cs:0,.95){Mean = 0};
\end{axis}\end{tikzpicture}''',
'windows':r'''\begin{tikzpicture}[x=.028\linewidth,y=1cm]
\foreach \t in {5,...,20}{
\ifnum\t<13 \def\cc{teal}\else\ifnum\t<17 \def\cc{gray!45}\else\def\cc{orange}\fi\fi
\fill[\cc] (\t-.43,0) rectangle (\t+.43,.45);
\node[font=\scriptsize] at (\t,-.2){\t};}
\node[font=\small,text=teal] at (8.5,.85){Reference: 5--12};
\node[font=\small] at (14.5,.85){Gap};
\node[font=\small,text=orange] at (18.5,.85){Recent: 17--20};
\node[font=\small] at (12.5,-.7){Current frame $t=20$};
\end{tikzpicture}''',
'trace':r'''\begin{tikzpicture}
\begin{axis}[width=.47\linewidth,height=4.0cm,xmin=0,xmax=34,ymin=0,ymax=1,xlabel={Frame},ylabel={Weight},title={Current and remembered},legend style={font=\scriptsize,at={(.02,.98)},anchor=north west}]
\addplot[teal,thick]coordinates{(0,.1)(9,.1)(10,.9)(34,.9)};\addlegendentry{Current}
\addplot[orange,thick,domain=0:9,samples=10]{.1+(1/81-.1)*.9^x};
\addplot[orange,thick,domain=10:34,samples=25]{.9+(.1+(1/81-.1)*.9^10-.9)*.9^(x-10)};\addlegendentry{Trace}
\end{axis}
\begin{axis}[at={(7.0cm,0)},width=.44\linewidth,height=4.0cm,xmin=0,xmax=34,ymin=0,ymax=1,xlabel={Frame},title={Absolute discrepancy}]
\addplot[orange,thick,domain=0:9,samples=10]{abs((1/81-.1)*.9^x)};
\addplot[orange,thick,domain=10:34,samples=25]{abs((.1+(1/81-.1)*.9^10-.9)*.9^(x-10))};
\end{axis}\end{tikzpicture}'''
}
# Preserve the measured 81-cell arrays as native TeX filled rectangles from previous CPU computation.
# Recompute only this small CPU example, independent of production training.
import sys
sys.path.insert(0,str(ROOT))
import torch
from SecondPass.StabilizedAttentionKrauzlis.model import LocalMotionBank
torch.set_num_threads(1);bank=LocalMotionBank().eval();g=torch.Generator().manual_seed(912)
base=.2+.6*torch.rand((100,100),generator=g)
movie=torch.stack([torch.roll(base,t,1) for t in range(10)])[None,:,None]
with torch.no_grad():cs,rs=bank(movie)
f=[r'\begin{tikzpicture}[x=.37cm,y=-.37cm]']
for j,lag in enumerate([1,2,4]):
    f.append(r'\begin{scope}[xshift='+str(j*4.8)+'cm]')
    f.append(r'\node[font=\small] at (4,-1.3){Lag '+str(lag)+'};')
    for y in range(9):
        for x in range(9):
            c=float(cs[0,8,j,44,y*9+x]);s=max(0,min(100,(c+.2)/1.2*100))
            f.append(fr'\fill[teal!{s:.3f}!white] ({x-.5},{y-.5}) rectangle ({x+.5},{y+.5});')
    f.append(fr'\draw[red,very thick] ({4+lag-.25},4)--({4+lag+.25},4);\draw[red,very thick] ({4+lag},3.75)--({4+lag},4.25);')
    for k in [0,4,8]:
        f.append(fr'\node[font=\scriptsize] at ({k},9){{{k-4}}};\node[font=\scriptsize] at (-1.1,{k}){{{k-4}}};')
    f.append(r'\node[font=\scriptsize] at (4,10){Horizontal shift};\end{scope}')
f.append(r'\end{tikzpicture}');figures['lags']='\n'.join(f)
def fig(name,caption,width=490):
    if name=='crops':caption='Figure 1. Left: where two neighboring sites lie in the full image. Right: the same crops enlarged to show individual pixels. Crop A covers columns 33-56; crop B covers 43-66. Both cover rows 33-56. Their shared region contains 14 columns by 24 rows = 336 pixels. Their centers are 10 pixels apart.'
    if name=='window':caption='Figure 2. Native vector plots of the one-dimensional Hann taper and its two-dimensional product. Windowing emphasizes the crop interior.'
    drawing=figures[name]
    if name in ('population','trace'):
        drawing=drawing.replace('width=.47','width=.43').replace('width=.44','width=.40').replace('at={(7.0cm,0)}','at={(8.0cm,0)}')
    parts.append(r'\begin{center}'+drawing+r'\end{center}'+'\n');small(caption)

# Reuse only the authored prose/equation calls, not the old PDF renderer.
s=(ROOT/'tmp/pdfs/build_motion_pathway.py').read_text(encoding='utf-8')
s=s[s.index("page('How our model"):s.index('\ndef footer')]
exec(s)
preamble=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=.72in,top=.65in,bottom=.72in]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs,array,tabularx}
\usepackage{xcolor,tikz,pgfplots}
\pgfplotsset{compat=1.18,every axis/.append style={axis lines=left,grid=none,tick label style={font=\scriptsize},label style={font=\small},title style={font=\small}}}
\usepackage{microtype}
\usepackage{titlesec,fancyhdr,lastpage}
\usepackage[unicode,colorlinks=true,linkcolor=teal,urlcolor=teal]{hyperref}
\definecolor{teal}{HTML}{087E8B}
\definecolor{navy}{HTML}{152F46}
\definecolor{muted}{HTML}{526778}
\definecolor{orange}{HTML}{DB7137}
\hypersetup{pdftitle={The optic flow pathway: mathematics, logic and intuition},pdfauthor={Visual Attention Working Memory project}}
\titleformat{\section}{\sffamily\Large\bfseries\color{navy}}{\thesection.}{.5em}{}
\titleformat{\subsection}{\sffamily\normalsize\bfseries\color{teal}}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{12pt}
\titlespacing*{\subsection}{0pt}{9pt}{5pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{7pt}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{2em}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\sffamily\scriptsize\color{muted}VISUAL ATTENTION \& WORKING MEMORY}
\fancyfoot[R]{\sffamily\scriptsize\color{muted}\thepage\ / \pageref*{LastPage}}
\renewcommand{\footrulewidth}{.2pt}
\begin{document}
\fontsize{10.5}{13.6}\selectfont
\color{navy}
'''
out=preamble+'\n'.join(parts)+r'\end{document}'+'\n'
# The original description of its production process is updated for this deliverable.
out=out.replace('This document follows the actual backbone', 'This LaTeX document follows the actual backbone')
out=out.replace('Each equation', 'Each equation')
(OUT/'optic_flow_pathway_latex.tex').write_text(out,encoding='utf-8')
print(json.dumps({'tex':str(OUT/'optic_flow_pathway_latex.tex'),'sections':sections,'native_equations':eqno,'native_tikz_figures':len(figures),'external_image_files':0},indent=2))
