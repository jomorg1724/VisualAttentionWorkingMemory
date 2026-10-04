"""Offline scientific atlas from real saved trial scores and map arrays only.
Existing system Python plotting environment; no model execution in this file.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import base64,csv,hashlib,html,json,math,textwrap,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy.stats import norm
from PIL import Image
OUT=Path(__file__).resolve().parent
BLUE='#176b9b';AMBER='#b17817';GRAY='#8796a4';INK='#192b3b';RED='#bb5353'
COLORS=[BLUE,AMBER,'#cf9a40','#895c22',GRAY]
CENTERS=np.array([[27,27],[73,27],[27,73],[73,73],[50,50]])


def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def csvwrite(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'data'/name).open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def binomial(values):
    a=np.asarray(values,dtype=float);n=len(a)
    if not n:return dict(n=0,rate=None,low=None,high=None)
    p=float(a.mean());z=norm.ppf(.975);d=1+z*z/n;mid=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return dict(n=n,rate=p,low=max(0,min(p,float(mid-half))),high=min(1,max(p,float(mid+half))))
def summary(rows):
    y=np.array([r['label'] for r in rows]);p=np.array([r['prediction'] for r in rows]);correct=(y==p)
    accuracy=binomial(correct);classes=np.unique(y);ba=float(np.mean([correct[y==c].mean() for c in classes]))
    out=dict(n=len(y),accuracy=accuracy['rate'],accuracy_low=accuracy['low'],accuracy_high=accuracy['high'],balanced_accuracy=ba,positive_rate=float((p==1).mean()),mean_prob1=float(np.mean([r['prob1'] for r in rows])))
    if set(classes)<=set([0,1]) and len(classes)==2:
        hit=binomial(p[y==1]==1);false=binomial(p[y==0]==1)
        h=(np.sum(p[y==1]==1)+.5)/(np.sum(y==1)+1);f=(np.sum(p[y==0]==1)+.5)/(np.sum(y==0)+1)
        out.update(hit_rate=hit['rate'],hit_n=hit['n'],false_alarm_rate=false['rate'],negative_n=false['n'],dprime=float(norm.ppf(h)-norm.ppf(f)),criterion=float(-.5*(norm.ppf(h)+norm.ppf(f))))
    else:out.update(dprime=None,criterion=None)
    return out

def paired_effect(sham,other):
    baseline={r['base_id']:r for r in sham};rows=[(baseline[r['base_id']],r) for r in other if r['base_id'] in baseline]
    assert len(rows)==len(other), 'Missing sham partner'
    da=np.array([int(b['correct'])-int(a['correct']) for a,b in rows]);dr=np.array([int(b['prediction']==1)-int(a['prediction']==1) for a,b in rows]);dp=np.array([b['prob1']-a['prob1'] for a,b in rows])
    rng=np.random.default_rng(374);inds=rng.integers(len(rows),size=(2000,len(rows)))
    out=dict(n=len(rows),delta_accuracy=float(da.mean()),delta_response=float(dr.mean()),delta_prob1=float(dp.mean()))
    flip=binomial([a['prediction']!=b['prediction'] for a,b in rows])
    out.update(response_flip_rate=flip['rate'],response_flip_wilson_low=flip['low'],response_flip_wilson_high=flip['high'])
    for key,values in [('accuracy',da),('response',dr),('prob1',dp)]:
        ci=np.quantile(values[inds].mean(1),[.025,.975]);out[key+'_low']=float(ci[0]);out[key+'_high']=float(ci[1])
    s=summary(sham);v=summary(other);out['delta_dprime']=v['dprime']-s['dprime'] if v['dprime'] is not None and s['dprime'] is not None else None;out['delta_criterion']=v['criterion']-s['criterion'] if v['criterion'] is not None and s['criterion'] is not None else None
    # Bootstrap trial identities once, preserving sham/intervention pairing.
    y=np.array([a['label'] for a,b in rows]);y1=np.array([b['label'] for a,b in rows]);p0=np.array([a['prediction'] for a,b in rows]);p1=np.array([b['prediction'] for a,b in rows])
    if set(y)=={0,1} and set(y1)=={0,1}:
        def dpcrit(p,labels):
            pp=p[inds];yb=labels[inds];nh=(yb==1).sum(1);nf=(yb==0).sum(1);h=(((pp==1)&(yb==1)).sum(1)+.5)/(nh+1);f=(((pp==1)&(yb==0)).sum(1)+.5)/(nf+1)
            return norm.ppf(h)-norm.ppf(f),-.5*(norm.ppf(h)+norm.ppf(f))
        d0,c0=dpcrit(p0,y);d1,c1=dpcrit(p1,y1)
        for name,val in [('dprime',d1-d0),('criterion',c1-c0)]:
            ci=np.quantile(val,[.025,.975]);out[name+'_low']=float(ci[0]);out[name+'_high']=float(ci[1])
    return out


def canvas(rows=1,cols=1,size=(13.3,8.6)):
    f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False);f.subplots_adjust(left=.075,right=.95,bottom=.23,top=.84,wspace=.34,hspace=.62);return f,ax

def configure():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'text.color':INK,'axes.labelcolor':INK,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b6c2cc','axes.grid':True,'grid.color':'#dce3e9','grid.alpha':.65,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})

class Atlas:
    def __init__(self):self.pdf=PdfPages(OUT/'Neuroscience_atlas.pdf');self.entries=[]
    def save(self,fig,slug,title,caption,source):
        number=len(self.entries)+1;fid=f'{number:02d}_{slug}';fig.suptitle(title,x=.055,y=.956,ha='left',fontsize=20,fontweight='bold');fig.text(.055,.145,'\n'.join(textwrap.wrap(caption,148)),fontsize=9.5,color='#526577',va='top',linespacing=1.4);fig.text(.055,.025,f'FROZEN CHECKPOINT 35039  •  FIGURE {number:02d}  •  {source}',fontsize=8,color='#526577')
        for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{fid}.{ext}',dpi=150)
        self.pdf.savefig(fig);plt.close(fig);self.entries.append(dict(id=fid,title=title,caption=caption,source=source))


def line_rate(ax,groups,color,label,marker='o'):
    pts=[]
    for x,rows in sorted(groups.items()):
        b=binomial([r['prediction']==1 for r in rows]);pts.append((x,b['rate'],b['low'],b['high'],b['n']))
    if not pts:return
    p=np.array(pts);ax.errorbar(p[:,0],p[:,1],yerr=np.maximum(0,np.array([p[:,1]-p[:,2],p[:,3]-p[:,1]])),fmt=marker+'-',color=color,label=label,capsize=2,ms=4)
    ax.set_ylim(-.04,1.04);ax.set_ylabel('Fraction reporting positive');return pts

def rotations(row):
    m=row['metadata'];target=m['target_location'];sign=m.get('cue_sign',1);return np.array(m['rotations_degrees'])*sign,target

def psych_groups(rows,foil=False):
    groups=defaultdict(list)
    for r in rows:
        delta,target=rotations(r);x=float(delta[(target+1)%4] if foil else delta[target])
        if foil and delta[target]!=0:continue
        if x==0 and r.get('magnitude',r['metadata'].get('rotation_magnitude'))!=0:continue
        groups[x].append(r)
    return groups


def phase(ax,meta):
    blanks=meta.get('blank_frames',[])
    ax.axvspan(.5,2.5,color='#dceee8',alpha=.4)
    if blanks:
        ax.axvspan(blanks[0]-.5,blanks[-1]+.5,color='#edf0f3',alpha=.5);ax.axvspan(blanks[0]-.5,min(blanks[0]+1.5,blanks[-1]+.5),color='#f4dfab',alpha=.7)
    if meta.get('cue_timing'):ax.axvline(meta['cue_frames'][0],color=AMBER,ls=':',lw=1)
    ax.axvline(meta.get('probe_frame',1),color=RED,ls=':',lw=1)


def main():
    configure();start=time.time();budget=json.loads((OUT/'budget.json').read_text());deadline=budget['deadline_unix'];execution=json.loads((OUT/'execution.json').read_text());plan=json.loads((OUT/'plan.json').read_text());rows=[json.loads(line) for line in (OUT/'data'/'trials.jsonl').read_text().splitlines()]
    assert rows and all(r['prediction']==int(np.argmax(r['logits'])) and r['correct']==(r['label']==r['prediction']) for r in rows)
    assert len(rows)==execution['counts']['scored_presentations']
    groups=defaultdict(list)
    for r in rows:groups[(r['group'],r['task'],r.get('delay',0),r.get('magnitude',-1),r['condition'])].append(r)
    summaries=[dict(group=k[0],task=k[1],delay=k[2],magnitude=k[3],condition=k[4],support='OOD ring delay: training catalog D0 only' if k[1]=='orientation_ring' and k[2]>0 else 'OOD magnitude' if k[3] in (0,3,6,10) else 'native renderer condition',**summary(v)) for k,v in groups.items()];csvwrite('condition_summary.csv',summaries)
    flat=[]
    for r in rows:
        m=r['metadata'];row={k:v for k,v in r.items() if k not in ('metadata','intervention')};row.update(target=m.get('target_location'),cue_sign=m.get('cue_sign'),difficulty=m.get('difficulty'),rotation_magnitude=m.get('rotation_magnitude'),rotations=m.get('rotations_degrees'),swapped_locations=m.get('swapped_locations'),stimulus_trial_id=m['trial_id']);flat.append(row)
    csvwrite('trial_scores.csv',flat)
    atlas=Atlas();psych=[r for r in rows if r['group']=='psychometric'];native=[r for r in rows if r['group']=='native'];causal=[r for r in rows if r['group']=='causal']
    # Empirical probabilities: not the mean softmax and not a fitted curve.
    fig,axes=canvas(1,2)
    psych_points=[]
    for i,d in enumerate([0,12]):
        rr=[r for r in psych if r['delay']==d]
        for foil,color,name in [(False,BLUE,'Target evidence'),(True,AMBER,'Foil evidence | target unchanged')]:
            pts=line_rate(axes[0,i],psych_groups(rr,foil),color,name)
            for x,p,lo,hi,n in pts or []:psych_points.append(dict(delay=d,evidence=name,signed_degrees=x,response_rate=p,low=lo,high=hi,n=int(n)))
        axes[0,i].set_title(f'Signed-cue orientation, D{d}');axes[0,i].set_xlabel('Signed cue-aligned rotation (degrees)');axes[0,i].legend(fontsize=8);axes[0,i].axvline(0,color=GRAY,ls=':',lw=1)
    atlas.save(fig,'signed_psychometrics','Does the observer use target rather than foil evidence?','Empirical final positive choices with 95% Wilson intervals. Negative = opposite-sign or unchanged target. Blue: target rotation. Amber: designated next-location foil, conditioning target unchanged. Zero is the all-zero catch sweep only. 0/3/6/10 degrees are analysis-only OOD magnitudes; 15/30/45 are native. Native pattern couples foil rotations; these are conditional descriptions, not independently randomized evidence weights.','data/psychometric_points.csv; trial_scores.csv')
    csvwrite('psychometric_points.csv',psych_points)
    # Matched cue relocation with the new target explicitly unchanged.
    orig={ (r['base_id'],r['magnitude']):r for r in psych if r['delay']==12}
    relocated=[r for r in rows if r['group']=='matched_relocation'];pairs=[(orig[(r['base_id'],r['magnitude'])],r) for r in relocated]
    subset=[(a,b) for a,b in pairs if rotations(b)[0][rotations(b)[1]]==0]
    fig,axes=canvas(1,2);gp=[defaultdict(list),defaultdict(list)]
    for a,b in subset:
        delta,target=rotations(a);x=float(delta[target])
        if x==0 and a['magnitude']!=0:continue
        gp[0][x].append(a);gp[1][x].append(b)
    for i,(g,c,label) in enumerate(zip(gp,[BLUE,AMBER],['Same patch cued','Same patch now foil; new target unchanged'])):line_rate(axes[0,0],g,c,label)
    axes[0,0].legend(fontsize=8);axes[0,0].set_xlabel('Physical old-target rotation, cue-aligned (degrees)')
    paired_rows=[]
    # Magnitude strata keep repeated underlying scenes together in each contrast.
    for mag in plan['magnitudes']:
        pp=[(a,b) for a,b in pairs if a['magnitude']==mag];a=[x for x,y in pp];b=[y for x,y in pp]
        effect=paired_effect(a,b);paired_rows.append(dict(magnitude=mag,both_members_correct=float(np.mean([x['correct'] and y['correct'] for x,y in pp])),labels_disagree=sum(x['label']!=y['label'] for x,y in pp),**effect))
    for r in paired_rows:axes[0,1].errorbar(r['magnitude'],r['delta_response'],yerr=[[r['delta_response']-r['response_low']],[r['response_high']-r['delta_response']]],fmt='o',color=BLUE,capsize=3)
    axes[0,1].axhline(0,color=GRAY,lw=1);axes[0,1].set_xlabel('Magnitude (degrees)');axes[0,1].set_ylabel('Relocated minus original positive response')
    atlas.save(fig,'matched_cue','Matched sensory scenes; cue identity changes the requested report','Only cue overlays change, with raw pre-overlay scenes and all rotations held fixed. New labels are recomputed. Left conditions the newly cued target to be unchanged; the same old-target physical evidence is now irrelevant. Right uses all matched pairs at each magnitude; 95% paired trial bootstrap. This is target reassignment, NOT an invalid-cue benefit or a cue-validity experiment.','data/matched_cue_pairs.csv')
    csvwrite('matched_cue_pairs.csv',paired_rows)
    # Seven sensory native difficulty plus class/order contrasts.
    sensory=['motion_direction','orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum']
    fields=dict(motion_direction='displacement_pixels',orientation='signed_orientation_degrees',contrast='contrast_increment',spatial_frequency='frequency_octave_increment',chromatic_increment='chromatic_increment',contour='alignment_jitter_degrees',natural_spectrum='beta_delta')
    fig,axes=canvas(2,4);sensory_rows=[]
    for ax,task in zip(axes.flat,sensory):
        rr=[r for r in native if r['task']==task];by=defaultdict(list)
        for r in rr:
            m=r['metadata'];key=fields[task];value=m.get(key,m.get('difficulty','native'))
            if task=='orientation':value=abs(value)
            by[str(value)].append(r)
        keys=sorted(by,key=lambda a:float(a) if a.replace('.','',1).isdigit() else a)
        for j,k in enumerate(keys):
            s=summary(by[k]);ax.errorbar(j,s['accuracy'],yerr=[[s['accuracy']-s['accuracy_low']],[s['accuracy_high']-s['accuracy']]],fmt='o',color=BLUE,capsize=3);sensory_rows.append(dict(task=task,level=k,**s))
        ax.set_xticks(range(len(keys)),keys,rotation=25,fontsize=8);ax.set_ylim(.6,1.02);ax.set_title(task.replace('_',' '));ax.set_ylabel('Accuracy');ax.set_xlabel('Native difficulty / increment')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,1,'Independent final-head observations\nNo spatial cue in these tasks\nDirection/order contrasts saved in CSV\nNo forced OOD difficulty curves',va='top',linespacing=1.8)
    atlas.save(fig,'native_sensory','Native sensory competence at checkpoint 35039','All seven sensory tasks use native generators and native difficulty draws. Points are empirical accuracy with Wilson 95% intervals and saved cell denominators. At a ceiling these data cannot identify thresholds or lapses. Binary interval labels mean frame0/frame1 for stronger evidence, not generic change/no-change. No cued/uncued allocation comparator is fabricated.','data/native_sensory_difficulty.csv')
    csvwrite('native_sensory_difficulty.csv',sensory_rows)
    order=[]
    for task in sensory:
        for label in sorted(set(r['label'] for r in native if r['task']==task)):
            rr=[r for r in native if r['task']==task and r['label']==label];order.append(dict(task=task,label=label,**summary(rr)))
    csvwrite('sensory_direction_interval_order.csv',order)
    fig,axes=canvas(1,2)
    for task,color in [('orientation_cued',BLUE),('spatial_binding',AMBER),('orientation_ring',GRAY)]:
        xx=[];yy=[];lo=[];hi=[]
        for d in plan['delays']:
            rr=[r for r in native if r['task']==task and r['delay']==d];s=summary(rr);xx.append(d);yy.append(s['accuracy']);lo.append(s['accuracy_low']);hi.append(s['accuracy_high'])
        axes[0,0].errorbar(xx,yy,yerr=[np.array(yy)-lo,np.array(hi)-yy],fmt='o--' if task=='orientation_ring' else 'o-',color=color,label=task+(' (D>0 OOD)' if task=='orientation_ring' else ''),capsize=3)
    axes[0,0].set_xlabel('Blank delay (frames; ring D>0 OOD)');axes[0,0].set_ylabel('Accuracy, Wilson95%');axes[0,0].set_ylim(.5,1.03);axes[0,0].legend(fontsize=8)
    rr=[r for r in native if r['task']=='orientation_ring' and r['delay']==0];line_rate(axes[0,1],psych_groups(rr),BLUE,'Ring target native D0');axes[0,1].set_xlabel('Signed target rotation (degrees)');axes[0,1].legend()
    atlas.save(fig,'delay_ring','Retention and ring-cue response','Cued orientation and binding: native D0/4/12/24, matched renderer seed across delays. Ring: ONLY D0 is trained; dashed D4/12/24 are supplemental OOD native-renderer delays, not solved-task validation. Right shows trained ring D0 signed rotation, without a sign glyph. First two nominal blanks overlap samples in stack-three inputs; only later blanks are stimulus-free. No decay constant is fitted at a ceiling.','data/condition_summary.csv')
    binding_rows=[]
    for d in plan['delays']:
        for kind in (0,1):
            rr=[r for r in native if r['task']=='spatial_binding' and r['delay']==d and r['label']==kind];b=binomial([r['prediction']==1 for r in rr]);binding_rows.append(dict(delay=d,event='target_swap' if kind else 'foil_only_swap',**b))
    csvwrite('binding_swaps_delay.csv',binding_rows)
    fig,axes=canvas(1,2)
    for kind,color in [('target_swap',BLUE),('foil_only_swap',AMBER)]:
        rr=[r for r in binding_rows if r['event']==kind];axes[0,0].errorbar([r['delay'] for r in rr],[r['rate'] for r in rr],yerr=[[r['rate']-r['low'] for r in rr],[r['high']-r['rate'] for r in rr]],fmt='o-',color=color,label=kind,capsize=3)
    axes[0,0].set_xlabel('Blank delay (frames)');axes[0,0].set_ylabel('Fraction positive');axes[0,0].legend()
    for j,d in enumerate(plan['delays']):
        rr=[r for r in psych if r['delay']==d and r['magnitude']==6];line_rate(axes[0,1],psych_groups(rr),[BLUE,AMBER,GRAY,RED][j],f'D{d}')
    axes[0,1].set_xlabel('Cue-aligned target rotation (degrees)');axes[0,1].legend()
    atlas.save(fig,'binding_subthreshold_delay','Target swaps and below-native orientation difficulty','Binding requires only the subsequently queried location to swap; every trial contains two exchanged items. The query is AFTER retention, not a pre-cue. Right: 6-degree orientation is an explicit analysis-only OOD magnitude, useful for measuring non-ceiling behavior without modifying training. Curves are observed choices, not fitted thresholds.','data/binding_swaps_delay.csv; trials.jsonl')
    # Actual maps and bootstrap region allocation, no reconstruction from means.
    allocation=[];map_manifest=[]
    (OUT/'viewer').mkdir(exist_ok=True)
    for file in sorted((OUT/'maps').glob('*.npz')):
        if time.time()>deadline-100:raise TimeoutError('Rendering map reserve exhausted')
        data=dict(np.load(file));info=json.loads(file.with_suffix('.json').read_text());task=info['task'];meta=info['metadata'];B,T=data['images'].shape[:2];name=file.stem
        index_record=dict(name=name,task=task,n=B,timesteps=T,script=f'viewer/{name}.js');map_manifest.append(index_record)
        # Sidecar JS uses base64 little-endian float32; offline file:// requires no fetch.
        payload=dict(name=name,task=task,n=B,timesteps=T,metadata=meta,quantities={},images=[])
        for b in range(B):
            paths=[]
            for t in range(T):
                p=OUT/'viewer'/f'{name}_b{b}_t{t}.png';Image.fromarray(np.clip(data['images'][b,t].transpose(1,2,0)*255,0,255).astype('uint8')).save(p);paths.append('viewer/'+p.name)
            payload['images'].append(paths)
        for key in [k for k in data if any(k.endswith(q) for q in ['beta','alpha_mean8','coeff_final_read','coeff_source2_all_reads','gru_write','gru_reset','gru_state_energy'])]:
            a=np.ascontiguousarray(data[key],dtype='<f4');signed='coeff' in key;vmax=float(np.max(np.abs(a))) if signed else float(a.max()) if 'energy' in key else 1.
            payload['quantities'][key]=dict(shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode(),vmin=-vmax if signed else 0,vmax=vmax)
        (OUT/'viewer'/f'{name}.js').write_text('window.loadDataset('+json.dumps(payload,separators=(',',':'))+');')
        if not name.endswith('D12'):continue
        sample_frames=sorted(set([0,2,4,5,T-2,T-1]));sample_frames=[t for t in sample_frames if t<T]
        for scale in range(3):
            fig,axes=canvas(3,len(sample_frames));fig.subplots_adjust(wspace=.15,hspace=.3,top=.82,bottom=.25)
            for col,t in enumerate(sample_frames):
                im=data['images'][0,t].transpose(1,2,0);axes[0,col].imshow(im);axes[0,col].set_title(f'Frame {t}');axes[0,col].grid(False)
                target=meta[0]['target_location']
                if t in meta[0].get('sample_frames',[]):phase_label='sample + cue' if task=='orientation_cued' else 'sample (no query)'
                elif t==meta[0].get('probe_frame'):phase_label='probe / report'
                elif t in meta[0].get('blank_frames',[]):phase_label='blank / stack overlap' if t in meta[0]['blank_frames'][:2] else 'pure blank'
                else:phase_label='cue / query' if t in meta[0].get('cue_frames',[]) else 'initial'
                axes[0,col].set_title(f'Frame {t}\n{phase_label}',fontsize=9)
                for loc,(cx,cy) in enumerate(CENTERS[:4]):axes[0,col].add_patch(Circle((cx,cy),12,fill=False,lw=1,ec=BLUE if loc==target else AMBER))
                for head in range(2):
                    ax=axes[head+1,col];mp=data[f's{scale}_beta'][0,t,:,:,head];ax.imshow(mp,vmin=0,vmax=1,cmap='viridis',interpolation='nearest',extent=(0,100,100,0));ax.grid(False)
                    if col==0:ax.set_ylabel(f'Beta head {head}')
                for row in range(3):axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='viridis'),cax=fig.add_axes([.961,.29,.01,.32]),ticks=[0,.5,1])
            atlas.save(fig,f'{task}_trial_beta_s{scale}',f'{task.replace("_"," ")} — real trial, scale {scale}','Trial0, not a trial average. Top: actual scene, blue target/eventual-query site and amber foils. Bottom: head-separated beta write/correction gates, fixed scale0–1. Maps cover actual 25/13/7 grids, not reconstructed regional values. Frame4 remains stack-overlap; frame5 is first pure blank. Every trial/frame and alpha/coefficient/ConvGRU map is selectable offline. Binding target is unknown to observer until query.','maps/'+file.name+'; viewer.html')
        # Finest alpha and implicit coefficient at selected SOURCE/read times.
        fig,axes=canvas(2,3)
        for h in range(2):
            for j,(key,t,label) in enumerate([(f's0_alpha_mean8',T-2,'Alpha, late retention/query'),(f's0_coeff_final_read',2,'Coefficient: final read ← source2'),(f's0_coeff_source2_all_reads',T-2,'Coefficient: late read ← source2')]):
                arr=data[key][0,t,:,:,h];bound=float(np.abs(data[key]).max()) if 'coeff' in key else 1;im=axes[h,j].imshow(arr,vmin=-bound if 'coeff' in key else 0,vmax=bound,cmap='RdBu_r' if 'coeff' in key else 'viridis',interpolation='nearest',extent=(0,100,100,0));axes[h,j].grid(False);axes[h,j].set_title(f'{label}\nhead{h}');fig.colorbar(im,ax=axes[h,j],shrink=.75)
        atlas.save(fig,f'{task}_alpha_signed','Retention gates and signed local temporal coefficients','Actual trial0 arrays. Alpha is the mean over 8 retention rows WITHIN each separate head. Signed coefficients can be negative and are neither spatial softmax probabilities nor final-decision attribution. Final-read arrays retain every source frame; fixed-source2 arrays retain every read frame. Stable range per quantity across saved trials/times. Exact weighted-value reconstruction was verified before output convolution.','maps/'+file.name)
        fig,axes=canvas(1,3)
        for ax,key,title in zip(axes[0],['gru_write','gru_reset','gru_state_energy'],['ConvGRU write (mean64)','ConvGRU reset (mean64)','ConvGRU state mean-square64']):
            vmax=1 if 'energy' not in key else float(data[key].max());im=ax.imshow(data[key][0,T-2],vmin=0,vmax=vmax,cmap='viridis',interpolation='nearest');ax.set_title(title);ax.grid(False);fig.colorbar(im,ax=ax,shrink=.7)
        atlas.save(fig,f'{task}_convgru','Downstream 7×7 recurrent readout — distinct measurements',f'Actual trial0 at frame{T-2}. These are ConvGRU write/reset gates and state energy, not KDA gates, implicit coefficients or attention probabilities. Channel averaging is stated explicitly. Full per-timestep arrays retained; no intermediate head is scored as an in-distribution decision.','maps/'+file.name)
        for quantity in ['beta','alpha_mean8','coeff_final_read','coeff_source2_all_reads']:
            fig,axes=canvas(3,2)
            for scale in range(3):
                field=data[f's{scale}_{quantity}'];mk=data[f's{scale}_masks'];locs=np.einsum('btyxh,syx->bths',field,mk)/mk.sum((1,2))[None,None,None,:]
                # Signed and mean-absolute are computed separately, never abs of mean.
                abs_locs=np.einsum('btyxh,syx->bths',np.abs(field),mk)/mk.sum((1,2))[None,None,None,:]
                # Explicit indexing avoids NumPy advanced-axis reordering.
                targets=data['targets']
                ordered=np.stack([np.take(locs[b],[int(targets[b])]+[(int(targets[b])+j)%4 for j in (1,2,3)]+[4],axis=-1) for b in range(B)])
                abso=np.stack([np.take(abs_locs[b],[int(targets[b])]+[(int(targets[b])+j)%4 for j in (1,2,3)]+[4],axis=-1) for b in range(B)])
                rng=np.random.default_rng(321);boot=ordered[rng.integers(B,size=(1000,B))].mean(1);lo,hi=np.quantile(boot,[.025,.975],axis=0);mean=ordered.mean(0)
                for h in range(2):
                    ax=axes[scale,h]
                    for s,site in enumerate(['target','foil1','foil2','foil3','background']):
                        ax.plot(range(T),mean[:,h,s],color=COLORS[s],label=site,lw=1.4);ax.fill_between(range(T),lo[:,h,s],hi[:,h,s],color=COLORS[s],alpha=.12)
                        for b in range(B):
                            for t in range(T):allocation.append(dict(dataset=name,task=task,trial=b,scale=scale,head=h,quantity=quantity,site=site,frame=t,axis='source' if quantity=='coeff_final_read' else 'read',value=float(ordered[b,t,h,s]),mean_absolute=float(abso[b,t,h,s]),mask_mass=float(mk.sum((1,2))[s]),mask_pixels=int((mk[s]>0).sum())))
                    phase(ax,meta[0]);ax.set_title(f'{field.shape[2]}×{field.shape[3]}, head{h}');ax.set_xlabel('Source frame (read fixed final)' if quantity=='coeff_final_read' else 'Read/update frame');ax.set_ylabel(quantity.replace('_',' '))
                    if scale==0 and h==1:ax.legend(fontsize=7,ncol=2)
            atlas.save(fig,f'{task}_allocation_{quantity}',f'{task.replace("_"," ")} — {quantity.replace("_"," ")} allocation',f'{B} independent native D12 trials; each foil separate. Matched translated kernels have equal support/mass at each scale; plotted means are per-mask-mass, not regional totals. Shading is95% trial-bootstrap uncertainty, never pixels/heads as replicates. Amber phase marks two stack-overlap blanks. For binding, target is only a retrospective eventual-query grouping before the query. Signed coefficient panel preserves sign; separate mean-absolute values are in source CSV.','data/allocation_per_trial.csv; maps/'+file.name)
        # Separate absolute summary, finest scale, heads separate.
        fig,axes=canvas(1,2)
        field=np.abs(data['s0_coeff_final_read']);mk=data['s0_masks'];regional=np.einsum('btyxh,syx->bths',field,mk)/mk.sum((1,2))[None,None,None,:]
        for h in range(2):
            for j,site in enumerate(['target','foil1','foil2','foil3','background']):
                a=np.stack([regional[b,:,h,4 if j==4 else (data['targets'][b]+j)%4] for b in range(B)]);axes[0,h].plot(range(T),a.mean(0),color=COLORS[j],label=site)
            phase(axes[0,h],meta[0]);axes[0,h].set_title(f'Head{h}');axes[0,h].set_xlabel('Source frame; read fixed at final');axes[0,h].set_ylabel('Mean absolute implicit coefficient');axes[0,h].legend(fontsize=8)
        atlas.save(fig,f'{task}_absolute_coeff','Absolute coefficient magnitude is not signed allocation','Mean of absolute local coefficients (not absolute of the signed mean). Heads remain separate; source time is not read time. The corresponding signed maps and all-read source2 maps remain in raw arrays and the offline viewer. These quantities describe local memory reads, not patch-to-patch attention or behavioral importance.','data/allocation_per_trial.csv')
    csvwrite('allocation_per_trial.csv',allocation)
    contrasts=defaultdict(dict)
    for r in allocation:
        key=(r['dataset'],r['scale'],r['head'],r['quantity'],r['frame'],r['trial'])
        contrasts[key][r['site']]=r['value']
    grouped=defaultdict(list)
    for key,v in contrasts.items():grouped[key[:-1]].append(v['target']-np.mean([v['foil1'],v['foil2'],v['foil3']]))
    allocation_contrasts=[]
    for key,values in grouped.items():
        a=np.array(values);rng=np.random.default_rng(733);boot=a[rng.integers(len(a),size=(2000,len(a)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
        allocation_contrasts.append(dict(dataset=key[0],scale=key[1],head=key[2],quantity=key[3],frame=key[4],n=len(a),target_minus_meanfoil=float(a.mean()),low=float(lo),high=float(hi)))
    csvwrite('allocation_contrasts.csv',allocation_contrasts)
    # Paired causal tables retain every cell, signals and negative subtypes.
    effects=[];subtypes=[]
    for task in ['orientation_cued','spatial_binding']:
        rr=[r for r in causal if r['task']==task];sham=[r for r in rr if r['condition']=='sham']
        for condition in sorted(set(r['condition'] for r in rr)):
            cell=[r for r in rr if r['condition']==condition];meta=cell[0].get('intervention',{});effects.append({'task':task,'condition':condition,**meta,**summary(cell),**paired_effect(sham,cell)})
            for name,test in [('signal',lambda r:r['label']==1),('noise',lambda r:r['label']==0),('unchanged_target',lambda r:'rotations_degrees' in r['metadata'] and rotations(r)[0][rotations(r)[1]]==0),('opposite_target',lambda r:'rotations_degrees' in r['metadata'] and rotations(r)[0][rotations(r)[1]]<0)]:
                subset=[r for r in cell if test(r)]
                if subset:subtypes.append(dict(task=task,condition=condition,subtype=name,**binomial([r['prediction']==1 for r in subset])))
    csvwrite('paired_causal_effects.csv',effects);csvwrite('causal_response_subtypes.csv',subtypes)
    for kind in ['inhibit','stimulate']:
        fig,axes=canvas(1,2)
        for ax,task in zip(axes[0],['orientation_cued','spatial_binding']):
            rr=[r for r in effects if r['task']==task and r.get('kind')==kind and (kind=='inhibit' or r.get('direction_name')=='feature')]
            for s,(site,c) in enumerate(zip(['cued','foil','background'],[BLUE,AMBER,GRAY])):
                cr=[r for r in rr if r['site']==site]
                for j,r in enumerate(cr):
                    x=j+(s-1)*.17;ax.errorbar(x,r['delta_accuracy'],yerr=[[r['delta_accuracy']-r['accuracy_low']],[r['accuracy_high']-r['delta_accuracy']]],fmt='o',color=c,capsize=2,label=site if j==0 else None)
                if s==0:ax.set_xticks(range(len(cr)),[r['epoch']+'\n'+str(r['dose']) for r in cr],rotation=35,fontsize=8)
            ax.axhline(0,color=GRAY,lw=1);ax.set_title(task.replace('_',' '));ax.set_ylabel('Paired accuracy change from sham');ax.legend(fontsize=8)
        atlas.save(fig,f'{kind}_site_epoch',('Local suppression' if kind=='inhibit' else 'Calibrated feature pulses')+' — site, timing and dose',f'{plan["causal_n"]} paired independent base trials per condition, a prespecified orientation magnitude mixture at D12. One update per pulse. Masks are matched in support/mass, not necessarily removed activation energy; achieved RMS is saved. Error bars95% paired trial bootstrap; zero-width bars do not prove zero future effect (finite flip bounds in CSV). Binding encoding/retention cued means eventual target, not available cue. Underlying finest-scale KDA state is untouched; downstream recurrence can carry effects.','data/paired_causal_effects.csv; achieved_interventions.jsonl')
        fig,axes=canvas(1,2)
        task='orientation_cued';rr=[r for r in causal if r['task']==task];sham=[r for r in rr if r['condition']=='sham']
        for ax,foil in zip(axes[0],[False,True]):
            line_rate(ax,psych_groups(sham,foil),INK,'Sham')
            for site,c in zip(['cued','foil','background'],[BLUE,AMBER,GRAY]):
                selected=[r for r in rr if r.get('intervention',{}).get('kind')==kind and r['intervention']['site']==site and r['intervention']['epoch']=='probe' and r['intervention']['dose']==1 and (kind=='inhibit' or r['intervention'].get('direction_name')=='feature')]
                if selected:line_rate(ax,psych_groups(selected,foil),c,site)
            ax.set_xlabel(('Foil' if foil else 'Target')+' signed cue-aligned rotation (degrees)');ax.legend(fontsize=8)
        atlas.save(fig,f'{kind}_psychometric',('Suppression' if kind=='inhibit' else 'Positive feature stimulation')+' psychometrics','Probe-update dose1 versus the identical sham trials. Left target evidence; right designated foil while target unchanged. Empirical fractions with Wilson95% intervals. Small denominators per magnitude are explicit in trial records:64 TOTAL per condition, not64 per point. No threshold fit or universal benefit is inferred from sparse points. Feature direction is absolute sin(2theta), not a learned rotation/decision direction.','data/trial_scores.csv; paired_causal_effects.csv')
    fig,axes=canvas(1,3)
    for ax,epoch in zip(axes[0],['encoding','retention','probe']):
        for site,c in zip(['cued','foil','background'],[BLUE,AMBER,GRAY]):
            for direction,style in [('feature','o-'),('random','s--')]:
                rr=sorted([r for r in effects if r['task']=='orientation_cued' and r.get('kind')=='stimulate' and r['site']==site and r['epoch']==epoch and r.get('direction_name')==direction],key=lambda r:r['dose'])
                if not rr:continue
                ax.errorbar([r['dose'] for r in rr],[r['delta_response'] for r in rr],yerr=[[r['delta_response']-r['response_low'] for r in rr],[r['response_high']-r['delta_response'] for r in rr]],fmt=style,color=c,capsize=2,label=f'{site}/{direction}',ms=4)
        ax.axhline(0,color=GRAY,lw=1);ax.set_title(epoch);ax.set_xlabel('Signed dose / calibration RMS');ax.set_ylabel('Paired positive-response shift')
    axes[0,2].legend(fontsize=6,ncol=2)
    atlas.save(fig,'stimulation_controls','Opposite pulses and matched random-direction controls','Solid circles: independently calibrated absolute-orientation direction. Dashed squares: orthogonal random direction with identical channel RMS. Positive/negative doses and three matched sites are measured, not inferred.95% paired bootstrap. A null mean shift can mask opposing sign/location effects; a shift is not necessarily a sensitivity improvement. Physical current, neural cell type and milliseconds are not specified by model units.','calibration.json; data/calibration.npz; paired_causal_effects.csv')
    fig,axes=canvas(1,2)
    for ax,metric in zip(axes[0],['dprime','criterion']):
        rr=[r for r in effects if r['task']=='orientation_cued' and r.get('epoch')=='probe' and r.get('dose')==1 and (r.get('kind')=='inhibit' or r.get('direction_name') in ('feature','random'))]
        for j,r in enumerate(rr):
            val=r['delta_'+metric];ax.errorbar(j,val,yerr=[[val-r[metric+'_low']],[r[metric+'_high']-val]],fmt='o',color=dict(cued=BLUE,foil=AMBER,background=GRAY)[r['site']],capsize=3)
        ax.set_xticks(range(len(rr)),[r['kind'][:3]+'/'+r.get('direction_name','')+'\n'+r['site'] for r in rr],rotation=40,fontsize=7);ax.axhline(0,color=GRAY,lw=1);ax.set_ylabel('Paired change in '+metric)
    atlas.save(fig,'sensitivity_criterion','Sensitivity and report criterion must be distinguished','Signal: aligned target rotation. Noise: unchanged or opposite target, pooled explicitly here; separate subtype response rates saved. d′ = Z(hit)−Z(false alarm); criterion = −[Z(hit)+Z(false alarm)]/2. All rates use loglinear(k+.5)/(n+1) correction, including ceiling cells.95% paired trial bootstrap; one checkpoint, exploratory multiple contrasts, no familywise significance claim.','data/paired_causal_effects.csv; causal_response_subtypes.csv')
    # Actual tested locations only: no interpolation over unmeasured sites.
    spatial=[];fig,axes=canvas(2,3)
    for row,kind in enumerate(['inhibit','stimulate']):
        for ax,site in zip(axes[row],['cued','foil','background']):
            sham={r['base_id']:r for r in causal if r['task']=='orientation_cued' and r['condition']=='sham'}
            cell=[r for r in causal if r['task']=='orientation_cued' and r.get('intervention',{}).get('kind')==kind and r['intervention']['site']==site and r['intervention']['epoch']=='probe' and r['intervention']['dose']==1 and (kind=='inhibit' or r['intervention'].get('direction_name')=='feature')]
            for loc in (range(4) if site!='background' else [4]):
                subset=[r for r in cell if (r['metadata']['target_location'] if site=='cued' else (r['metadata']['target_location']+1)%4 if site=='foil' else 4)==loc]
                if not subset:continue
                delta=np.mean([r['correct']-sham[r['base_id']]['correct'] for r in subset]);cx,cy=CENTERS[loc];ax.scatter([cx],[cy],c=[delta],vmin=-1,vmax=1,cmap='RdBu_r',s=650,edgecolors=GRAY);ax.text(cx,cy,f'{delta:+.2f}\nn={len(subset)}',ha='center',va='center',fontsize=7);spatial.append(dict(kind=kind,dose=1,site=site,location=loc,x=int(cx),y=int(cy),n=len(subset),delta_accuracy=float(delta)))
            ax.set_xlim(0,100);ax.set_ylim(100,0);ax.set_aspect('equal');ax.set_title(kind+' / '+site);ax.set_xlabel('Image x (pixels)');ax.set_ylabel('Image y (pixels)')
    atlas.save(fig,'tested_spatial_sites','Spatial effects: only locations actually perturbed','Top: finest KDA emitted-field suppression1. Bottom: positive feature stimulation1RMS. Probe update, orientation D12. Each circle is a tested mask center approximately mapped to image pixels; no interpolation over unmeasured locations. Labels are paired accuracy differences and actual denominators. Smaller physical-location strata are descriptive; relative-site uncertainty is in the paired table.','data/tested_spatial_sites.csv')
    csvwrite('tested_spatial_sites.csv',spatial)
    # Calibration and limits complete the observer-centered sequence.
    cal=json.loads((OUT/'calibration.json').read_text());cd=dict(np.load(OUT/'data'/'calibration.npz'));half=cal['fit_trials'];angles=cd['angles'][half:];projection=cd['validation_projection']
    fig,axes=canvas(1,2);axes[0,0].scatter(np.sin(2*angles).ravel(),projection.ravel(),s=10,alpha=.5,color=BLUE);axes[0,0].set_xlabel('Independent sin(2×actual angle)');axes[0,0].set_ylabel('Projection onto frozen feature direction');axes[0,0].set_title(f'Validation correlation r={cal["validation_correlation"]:.3f}')
    axes[0,1].axis('off');axes[0,1].text(0,1,'Calibration and causal limits\n\n'+f'{half} fit trials; {cal["validation_trials"]} independent validation trials\nActivation RMS: {cal["activation_rms"]:.4f}\n32-channel direction has RMS1\nAbsolute orientation preference, not change preference\nNo main-model gradients, optimizer or training\nNative final output head only\nOne checkpoint; no population-level biological claim',va='top',linespacing=1.5)
    atlas.save(fig,'calibration_limits','What the feature pulse does—and does not—identify','OLS encoding localizer uses actual cos(2theta)/sin(2theta) orientation on independent native sample frames; trial split precedes pooling locations. Cue location/sign and task label are not calibration targets. Absolute sample orientation is independently drawn from the task label/cue rule. Positive validation confirms feature preference, not a signed rotation comparator or universal behavioral enhancement.','calibration.json; data/calibration.npz; REFERENCE_GUIDE.md')
    fig,axes=canvas(2,4)
    for ax,task in zip(axes.flat,sensory):
        rr=[r for r in order if r['task']==task]
        for j,r in enumerate(rr):ax.errorbar(j,r['accuracy'],yerr=[[r['accuracy']-r['accuracy_low']],[r['accuracy_high']-r['accuracy']]],fmt='o',color=COLORS[j],capsize=3)
        names=['right','up','left','down'] if task=='motion_direction' else ['negative','positive'] if task=='orientation' else ['frame0','frame1']
        ax.set_xticks(range(len(rr)),names,fontsize=8);ax.set_ylim(.65,1.03);ax.set_title(task.replace('_',' '));ax.set_ylabel('Accuracy, Wilson95%')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,1,'Native label/order controls\nNo reversed movie generated\nNo report-rule changes\nPer-class denominators in CSV',va='top',linespacing=1.8)
    atlas.save(fig,'sensory_order_controls','Direction and interval-order controls at native difficulty','Native observational strata, not synthetic reversals. Motion classes are right/up/left/down; orientation is signed change. Other binary tasks report which frame contains the stronger quantity or structured contour. Points show empirical accuracy and95% Wilson intervals. These sensory tasks have no spatial cue and cannot establish selective attention by themselves.','data/sensory_direction_interval_order.csv')
    atlas.pdf.close();dump(OUT/'figure_manifest.json',atlas.entries)
    make_viewer(map_manifest)
    make_gallery(atlas.entries)
    # Replay/source manifest and honest numerical findings.
    native_summary=[r for r in summaries if r['group']=='native' and not (r['task']=='orientation_ring' and r['delay']>0)];best=sorted([r for r in effects if r['condition']!='sham'],key=lambda r:abs(r['delta_accuracy']),reverse=True)
    findings=['# Frozen neuroscience findings — checkpoint35039','',f'Executed {len(rows):,} scored trial presentations; {execution["counts"]["calibration_presentations"]} independent calibration trials and {execution["counts"]["map_presentations"]} map trial presentations. This is an exploratory frozen-observer analysis, not retraining or a later final test.','', '## Native competence']
    for task in sorted(set(r['task'] for r in native_summary)):
        tr=[r for r in native_summary if r['task']==task];findings.append(f'- {task}: '+', '.join(f'D{r["delay"]} BA {r["balanced_accuracy"]:.3f} (n={r["n"]})' for r in tr))
    ring_extra=[r for r in summaries if r['group']=='native' and r['task']=='orientation_ring' and r['delay']>0]
    findings.append('- Supplemental ring delays are OOD (training catalog is D0 only): '+', '.join(f'D{r["delay"]} BA {r["balanced_accuracy"]:.3f}' for r in ring_extra)+'. Raw group=native means native renderer, not trained support; condition_summary.csv explicitly identifies OOD support.')
    findings.extend(['','## Signed orientation and matched cues'])
    for mag in plan['magnitudes']:
        rr=[r for r in summaries if r['group']=='psychometric' and r['delay']==12 and r['magnitude']==mag]
        if rr:findings.append(f'- D12 magnitude{mag}°: accuracy {rr[0]["accuracy"]:.3f}, positive fraction {rr[0]["positive_rate"]:.3f}, n={rr[0]["n"]}. '+('Analysis-only OOD magnitude.' if mag not in (15,30,45) else 'Native magnitude.'))
    findings.extend(['- Matched cue relocation changes the required report target; all pre-overlay scenes and rotations are physically matched. It is not classical valid/invalid cueing. See paired table and Figure2.','- The historical PsychOrientationStream was NOT bit-identical: an extra permutation changes RNG draws. The new sweep passed exact native pixel/label parity before magnitude overrides.','', '## Allocation and memory','- Actual trial/time/scale/head beta, mean8-row alpha, signed coefficients and separately computed absolute summaries are retained in NPZ. Final-read×all-source and source2×all-read axes are distinct. Reconstruction was tested against real pre-output-convolution reads.','- All10 tasks have representative maps; native cued orientation and binding D12 additionally have trial-bootstrap allocation summaries. No spatial cue comparator is assigned to sensory tasks.','- Binding retrocue arrives after retention: eventual-target differences before query cannot be called anticipatory selection. First2 blanks are frame-stack overlap.','', '## Paired functional interventions'])
    descriptive=[f'- {r["dataset"]} beta25 head{r["head"]}, frame{r["frame"]}: target−mean3foils {r["target_minus_meanfoil"]:+.5f} [{r["low"]:+.5f},{r["high"]:+.5f}], n={r["n"]}. This is a gate contrast, not final-decision attribution.' for r in allocation_contrasts if r['scale']==0 and r['quantity']=='beta' and r['frame'] in (2,14)]
    where=findings.index('## Paired functional interventions');findings[where:where]=descriptive+['']
    for r in best[:8]:findings.append(f'- {r["task"]} / {r["condition"]}: Δaccuracy {r["delta_accuracy"]:+.4f} [{r["accuracy_low"]:+.4f}, {r["accuracy_high"]:+.4f}], Δpositive {r["delta_response"]:+.4f}, Δd′ {r["delta_dprime"]:+.4f}, Δcriterion {r["delta_criterion"]:+.4f}; n={r["n"]}.')
    findings.extend([f'- Localizer held-out r={cal["validation_correlation"]:.4f}; feature validation passed={cal["independent_validation_passed"]}. RMS={cal["activation_rms"]:.4f}. Direction represents absolute sin2θ preference, NOT task-aligned change preference.','- Interventions touch only post-output finest KDA emissions. Same-module stored state is intact; downstream/coarser recurrence may carry effects. Feature-map support is not an isolated retinal lesion because convolutions/GroupNorm couple locations.','- Full effect table includes all tested doses, epochs, relative sites and random controls. The list above is exploratory ranking, not multiplicity-corrected significance.','', '## Interpretation and unmeasured comparisons','- No reaction-time policy, calibrated monkey current, neuronal excitation/inhibition identity or biological equivalence is claimed. Signed activation suppression is a functional perturbation.','- No classical cue-validity manipulation exists for this report rule. Foil rotations remain coupled by the native multiset; a fully independent factorial foil sweep and fitted evidence weights were not measured.','- Small causal bins (64TOTAL per condition before any profile reduction) limit psychometric shifts; no threshold/lapse or memory-decay fit is asserted. Trial uncertainty is not between-checkpoint generality.','- Image recognition is partial and omitted from solved-task scope. The two failing cued-motion tasks are outside this study. Sensory motion competence does not establish cued-motion competence.','- Allocation-versus-magnitude and matched-cue allocation contrasts were not collected: allocation maps use native D0 representatives and native D12 primary trials. No magnitude-dependent allocation mechanism is inferred.','- Causal epoch coverage is the pinned finite grid; the source CSV is authoritative. Binding random-direction stimulation was not included; cued orientation has those controls.','', '## Verification',f'- Main model unchanged: {execution["model_immutable"]}; checkpoint bytes/hash unchanged: {execution["checkpoint_immutable"]}.',f'- Saved-logit argmax, correctness and scored count replay passed for all{len(rows)}rows.',f'- {len(atlas.entries)} figures, each PNG/SVG/PDF; offline trial/frame viewer; raw arrays and source CSVs.',f'- Budget origin/deadline: {budget["start_unix"]} / {deadline}; rendering complete elapsed {time.time()-budget["start_unix"]:.1f}s.',f'- Execution status: {execution["status"]}; errors: {execution["errors"]}.'])
    (OUT/'FINDINGS.md').write_text('\n'.join(findings)+'\n')
    (OUT/'README.md').write_text('# Frozen neuroscience atlas\n\nOpen [index.html](index.html) for scientific figures and [viewer.html](viewer.html) for actual per-trial per-timestep maps. [Neuroscience_atlas.pdf](Neuroscience_atlas.pdf) has one figure per page. [FINDINGS.md](FINDINGS.md) states measured outcomes and missing comparisons.\n\n## Data and reproduction\n- `data/trials.jsonl`: complete logits, labels, predictions, native metadata and pair IDs. `trial_scores.csv` is normalized export.\n- `data/paired_causal_effects.csv`, `condition_summary.csv`, allocation and psychometric CSVs: exact numerical plot sources.\n- `maps/*.npz`: real float32 images/gates/signed coefficients, masks, axes and reconstruction errors; matching JSON metadata.\n- `data/calibration.npz`, `calibration.json`: frozen independent feature localizer and held-out validation.\n- `intervention_grid.json`, `achieved_interventions.jsonl`: sites/epochs/doses and achieved RMS.\n- `budget.json`, `plan.json`, `execution.json`, `completion.json`: fixed cap, profiled/prepinned grid, actual exposure and completion.\n\nCPU tests: `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python -m pytest SecondPass/SpatialReadout/NeuroscienceAnalysis/test_analysis.py -q`. Plot tests use system `python3 -m pytest .../test_render.py -q`. CPU preparation: same torch interpreter `-m SecondPass.SpatialReadout.NeuroscienceAnalysis.run --prepare`. The completed `--execute` refuses budget renewal; rerunning requires new explicit authorization, not deleting the budget. Regenerating only figures uses existing system `python3 SecondPass/SpatialReadout/NeuroscienceAnalysis/render.py`; do not use this to evade the original cap.\n\nNo files outside this analysis subtree were modified. Reference guide is independently authored. Scientific caveats: not softmax attention, not physiological inhibition/current, not classical cue-validity or reaction-time analysis. See captions and findings.\n')
    artifacts=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in str(p) or p.name in ['artifact_manifest.json','completion.json','progress.jsonl','worker.lock']:continue
        artifacts.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump(OUT/'artifact_manifest.json',artifacts)
    completion=dict(status='complete_pinned_grid' if execution['status']=='measured' else 'partial',all_requested_comparisons_complete=False,scored_rows=len(rows),counts=execution['counts'],figure_count=len(atlas.entries),map_datasets=len(map_manifest),model_immutable=execution['model_immutable'],checkpoint_immutable=execution['checkpoint_immutable'],prediction_replay_verified=True,completed_within_cap=time.time()<=deadline,start_unix=budget['start_unix'],deadline_unix=deadline,completed_unix=time.time(),elapsed_seconds=time.time()-budget['start_unix'],unmeasured=['classical cue validity undefined','reaction time undefined','fully independent foil factorial','allocation vs magnitude/matched cue','binding random pulse control','recognition supplemental omitted'],errors=execution['errors'])
    dump(OUT/'completion.json',completion);print(json.dumps(completion,indent=2))


def make_gallery(entries):
    cards=''.join(f'<article><h2>{html.escape(r["title"])}</h2><a href="figures/{r["id"]}.svg"><img src="figures/{r["id"]}.png"></a><p>{html.escape(r["caption"])}</p><p>'+ ' · '.join(f'<a href="figures/{r["id"]}.{ext}">{ext.upper()}</a>' for ext in ['png','svg','pdf'])+f' · Source: {html.escape(r["source"])}</p></article>' for r in entries)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen observer neuroscience atlas</title><style>body{font:17px/1.6 system-ui;color:#192b3b;max-width:1200px;margin:40px auto;background:white}article{border-top:1px solid #dce3e9;padding:28px 0}img{width:100%}a{color:#176b9b}header{padding:20px;background:#f4f7f9}p{max-width:1050px}</style><header><h1>Frozen observer: cue-conditioned evidence and local perturbations</h1><p>Fresh whole-model ConvGRU lineage · fixed checkpoint35039 · no training. Actual saved trial scores and raw spatial arrays.</p><nav><a href="viewer.html">Interactive individual trial/frame maps</a> · <a href="Neuroscience_atlas.pdf">PDF atlas</a> · <a href="FINDINGS.md">Findings</a> · <a href="completion.json">Completion</a> · <a href="REFERENCE_GUIDE.md">Scientific reference guide</a> · <a href="data/trial_scores.csv">Trial CSV</a></nav></header>'+cards)


def make_viewer(manifest):
    (OUT/'viewer_manifest.json').write_text(json.dumps(manifest,indent=2))
    page='''<!doctype html><meta charset="utf-8"><title>Actual per-trial spatial maps</title><style>body{font:16px/1.5 system-ui;margin:28px;color:#192b3b;background:white}canvas{width:300px;height:300px;image-rendering:pixelated;border:1px solid #dce3e9}.row{display:flex;gap:24px;flex-wrap:wrap}.controls{padding:14px;background:#f3f6f8;display:flex;gap:16px;flex-wrap:wrap}select{max-width:320px}pre{white-space:pre-wrap;max-width:1000px;font-size:13px}h2{font-size:18px}</style><h1>Actual individual trial × frame maps</h1><p><a href="index.html">Scientific atlas</a> · <a id="raw">Raw NPZ</a>. No spatial map reconstructed from regional means. Heads are shown separately, with common scale within each quantity/dataset. Approximate feature-grid-to-image alignment is not retinal localization.</p><div class="controls"><label>Dataset <select id="dataset"></select></label><label>Trial <input id="trial" type="number" value="0" min="0" style="width:65px"></label><label>Frame <input id="frame" type="range" min="0" value="0"><span id="fnum"></span></label><label>Scale <select id="scale"><option value="0">25×25 KDA</option><option value="1">13×13 KDA</option><option value="2">7×7 KDA</option><option value="gru">7×7 ConvGRU</option></select></label><label>Quantity <select id="quantity"></select></label><label><input id="overlay" type="checkbox"> Overlay on actual scene</label></div><div id="timeline" style="display:flex;gap:3px;margin:14px 0;flex-wrap:wrap"></div><p id="meaning"></p><div class="row"><section><h2>Actual frame + task locations</h2><canvas id="scene" width="300" height="300"></canvas></section><section><h2 id="h0">Head0</h2><canvas id="map0" width="300" height="300"></canvas></section><section><h2 id="h1">Head1</h2><canvas id="map1" width="300" height="300"></canvas></section></div><p id="range"></p><pre id="metadata"></pre><script>const MANIFEST=__MANIFEST__;const PARAMS=new URLSearchParams(location.search);let D=null,A={},image=new Image();const $=s=>document.getElementById(s);MANIFEST.forEach((m,i)=>$('dataset').add(new Option(m.name,i)));window.loadDataset=function(d){D=d;A={};Object.entries(d.quantities).forEach(([k,v])=>{let raw=atob(v.data),u=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)u[i]=raw.charCodeAt(i);A[k]={...v,a:new Float32Array(u.buffer)};delete A[k].data;});$('trial').max=d.n-1;$('trial').value=Math.min(d.n-1,+(PARAMS.get('trial')||0));$('frame').max=d.timesteps-1;$('frame').value=Math.min(d.timesteps-1,+(PARAMS.get('frame')||0));$('raw').href='maps/'+d.name+'.npz';updateQuantities();};function load(){let m=MANIFEST[$('dataset').value],s=document.createElement('script');s.src=m.script;document.body.appendChild(s);}function updateQuantities(){let q=$('quantity'),prev=q.value;q.innerHTML='';let values=$('scale').value==='gru'?['write','reset','state_energy']:['beta','alpha_mean8','coeff_final_read','coeff_source2_all_reads'];values.forEach(k=>q.add(new Option(k,k)));if(values.includes(prev))q.value=prev;draw();}function color(v,lo,hi){let z=(v-lo)/(hi-lo||1);z=Math.min(1,Math.max(0,z));if(lo<0){if(z<.5)return [Math.round(40+430*z),Math.round(95+320*z),Math.round(160+190*z)];return [Math.round(255-180*(z-.5)),Math.round(255-390*(z-.5)),Math.round(255-380*(z-.5))];}return [Math.round(50+190*z),Math.round(30+200*z),Math.round(110-70*z)];}function draw(){if(!D)return;let b=Math.max(0,Math.min(D.n-1,+$('trial').value)),t=+$('frame').value,scale=$('scale').value,q=$('quantity').value,key=scale==='gru'?'gru_'+q:'s'+scale+'_'+q,dat=A[key],shape=dat.shape,H=shape[2],W=shape[3],heads=shape.length===5?shape[4]:1,m=D.metadata[b];$('timeline').innerHTML='';for(let tt=0;tt<D.timesteps;tt++){let blank=(m.blank_frames||[]).includes(tt),overlap=(m.blank_frames||[]).slice(0,2).includes(tt),sample=(m.sample_frames||[]).includes(tt),cue=(m.cue_frames||[]).includes(tt),probe=tt===m.probe_frame;let label=probe?'probe':sample?'sample':blank?(overlap?'stack overlap':'pure blank'):cue?'cue/query':'frame';let button=document.createElement('button');button.textContent=tt+' '+label;button.style.cssText='font:11px system-ui;padding:6px;border:1px solid '+(tt===t?'#176b9b':'#ccd4dc')+';background:'+(probe?'#f4dddd':overlap?'#f4dfab':sample?'#dceee8':blank?'#edf0f3':'white');button.onclick=()=>{$('frame').value=tt;draw();};$('timeline').appendChild(button);}$('fnum').textContent=t;$('meaning').textContent=q==='coeff_final_read'?'Slider is SOURCE time; read fixed at final frame '+(D.timesteps-1)+'. Signed local implicit coefficients, not probabilities.':q==='coeff_source2_all_reads'?'Slider is READ time; source fixed at frame '+Math.min(2,D.timesteps-1)+'. Before that source exists coefficients are zero.':scale==='gru'?'Slider is update time. ConvGRU gates/state energy are distinct from KDA memory coefficients.':'Slider is update time. Beta is write/correction gate; alpha is mean of8 retention rows within each head.';$('range').textContent='Fixed display range '+dat.vmin.toPrecision(4)+' to '+dat.vmax.toPrecision(4)+'. Blue circle: target/eventual binding query. Amber: each foil. First two blanks still contain sample frames in stack3. Sensory tasks have no spatial cue.';$('metadata').textContent=JSON.stringify({trial:b,frame:t,task:D.task,label:m.label,target:m.target_location,cue_sign:m.cue_sign,rotations:m.rotations_degrees,sample_frames:m.sample_frames,blank_frames:m.blank_frames,cue_frames:m.cue_frames,probe_frame:m.probe_frame,cue_timing:m.cue_timing,trial_id:m.trial_id},null,2);image.onload=()=>{let c=$('scene').getContext('2d');c.clearRect(0,0,300,300);c.drawImage(image,0,0,300,300);if(m.target_location!==undefined){[[27,27],[73,27],[27,73],[73,73]].forEach(([x,y],i)=>{c.strokeStyle=i===m.target_location?'#176b9b':'#b17817';c.lineWidth=2;c.beginPath();c.arc(x*3,y*3,36,0,Math.PI*2);c.stroke();});}for(let h=0;h<2;h++){let cv=$('map'+h),ctx=cv.getContext('2d');ctx.clearRect(0,0,300,300);$('h'+h).textContent=heads===1?(h===0?'ConvGRU channel summary':'No second ConvGRU head'):'KDA head'+h;if(h>=heads)continue;if($('overlay').checked)ctx.drawImage(image,0,0,300,300);for(let y=0;y<H;y++)for(let x=0;x<W;x++){let idx=((((b*D.timesteps+t)*H+y)*W+x)*heads+h),v=dat.a[idx],rgb=color(v,dat.vmin,dat.vmax);ctx.fillStyle='rgba('+rgb.join(',')+','+($('overlay').checked?.65:1)+')';ctx.fillRect(x*300/W,y*300/H,300/W+.1,300/H+.1);}}};image.src=D.images[b][t];}['trial','frame','quantity','overlay'].forEach(id=>$(id).addEventListener('input',draw));$('scale').onchange=updateQuantities;$('dataset').onchange=load;let initial=MANIFEST.findIndex(m=>m.name===PARAMS.get('dataset'));if(initial>=0)$('dataset').value=initial;load();</script>'''
    (OUT/'viewer.html').write_text(page.replace('__MANIFEST__',json.dumps(manifest)))

if __name__=='__main__':main()
