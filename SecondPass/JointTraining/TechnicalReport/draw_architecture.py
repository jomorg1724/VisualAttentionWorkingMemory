"""CPU-only vector architecture schematic; no model calls or GPU use."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.1, 2.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
ink = '#19304A'; blue = '#E8F0F7'; gold = '#FFF1D6'
def box(x, y, w, h, text, color=blue):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.08',facecolor=color,edgecolor=ink,linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=ink,linespacing=1.3)
def arrow(x,y,xx,yy):
    ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':ink,'lw':1})
box(.06,2.53,1.38,1.05,'RGB frame\n+ 2 past frames\n9 × 100 × 100')
box(1.80,2.53,1.36,1.05,'Conv 1\n32 × 50 × 50')
box(3.50,2.53,1.36,1.05,'Conv 2 + KDA\n(64 + 32)\n× 25 × 25')
box(5.20,2.53,1.36,1.05,'Conv 3 + KDA\n(96 + 32)\n× 13 × 13')
box(6.90,2.53,1.36,1.05,'Conv 4 + KDA\n(128 + 32)\n× 7 × 7')
box(8.62,2.53,1.30,1.05,'Flatten\nLinear + ReLU\n256 features')
for x,xx in ((1.44,1.80),(3.16,3.50),(4.86,5.20),(6.56,6.90),(8.26,8.62)): arrow(x,3.06,xx,3.06)
for x in (3.50,5.20,6.90):
    box(x,1.22,1.36,.71,'Local state S\n2 × 8 × 16 / site',gold)
    arrow(x+.39,1.93,x+.39,2.53); arrow(x+.96,2.53,x+.96,1.93)
ax.text(4.17,.84,'States persist across timesteps; reset between trials.',ha='center',color=ink,fontsize=8)
box(8.62,1.22,1.30,.71,'Shared GRU\n256-unit state',gold); arrow(9.27,2.53,9.27,1.93)
box(8.62,.03,1.30,.70,'Selected task\nlinear head'); arrow(9.27,1.22,9.27,.73)
ax.text(.1,.2,'Learned memory: spatial associative fields + global GRU.\nNo GRU-to-encoder feedback; no global softmax over locations.',va='bottom',color=ink,fontsize=8)
fig.tight_layout(pad=.1)
fig.savefig(ROOT/'architecture.pdf',bbox_inches='tight')
fig.savefig(ROOT/'architecture.png',dpi=170,bbox_inches='tight')
print(ROOT/'architecture.pdf')
