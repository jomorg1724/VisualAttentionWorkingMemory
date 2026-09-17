"""Small frozen-state atlas: held-out probes, shared PCA, descriptive tSNE and clusters."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import json,time,base64,html
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score,normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
KINDS=('H','R','A','output');FAMILIES=('orientation_single','orientation_binding','motion_direction')
def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def load(split,f,d):
    stem=f'{split}_{f}_D{d}';return dict(np.load(HERE/(stem+'.npz'))),json.loads((HERE/(stem+'_rows.json')).read_text())
def select(pack,t,kind):
    data,rows=pack;ix=[i for i,r in enumerate(rows) if r['time']==t];return data[kind][ix].astype('float64'),[rows[i] for i in ix]
def axial(theta):return np.c_[np.cos(2*theta),np.sin(2*theta)]
def target(rows,tag):
    if tag in ('sample_angle','right_angle'):return axial(np.array([r[tag] for r in rows]))
    if tag=='counts':
        y=np.array([r['counts'] for r in rows],float)/8;return y-y.mean(1,keepdims=True)
    k=4 if tag in ('label_motion','direction') else 2
    field='label' if tag.startswith('label') else tag
    return np.eye(k)[[r[field] for r in rows]]
def fit(x,y,v,vy):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    a=(x-mean)/scale;b=(v-mean)/scale;best=None
    for lam in (1.,10.,100.):
        m=Ridge(alpha=lam,solver='cholesky').fit(a,y);loss=float(np.mean((m.predict(b)-vy)**2))
        if best is None or loss<best[0]:best=(loss,m,lam)
    return mean,scale,best[1],best[2]
def predict(fit,x):return fit[2].predict((x-fit[0])/fit[1])
def score(y,p,tag):
    if tag in ('sample_angle','right_angle'):
        a=np.arctan2(p[:,1],p[:,0]);b=np.arctan2(y[:,1],y[:,0]);return np.degrees(np.abs(np.angle(np.exp(1j*(a-b))))/2)
    if tag=='counts':return np.mean((y-p)**2,1)
    return (y.argmax(1)==p.argmax(1)).astype(float)
def uncertainty(values):
    rng=np.random.default_rng(731);means=values[rng.integers(len(values),size=(500,len(values)))].mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]
def figure(fig,name):
    fig.savefig(HERE/'figures'/f'{name}.svg',bbox_inches='tight',facecolor=fig.get_facecolor());fig.savefig(HERE/'figures'/f'{name}.png',dpi=155,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
def main(deadline=None):
    started=time.time();(HERE/'figures').mkdir(exist_ok=True);rng=np.random.default_rng(839)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfaf5','axes.facecolor':'#fbfaf5'})
    def check():
        if deadline and time.time()>deadline-40:raise TimeoutError('Analysis cap reached; saved partial artifacts preserved')
    cache={(s,f,d):load(s,f,d) for s in ('train','val','test') for f in FAMILIES for d in (0,24)}
    probes=[];predictions={};temporal={};behavior=[]
    for f in FAMILIES:
        rows=json.loads((HERE/f'test_{f}_behavior.json').read_text())
        for d in (0,24):
            r=[v for v in rows if v['delay']==d];y=np.array([x['label'] for x in r]);p=np.array([x['prediction'] for x in r]);behavior.append(dict(family=f,delay=d,n=len(r),accuracy=float((y==p).mean()),balanced_accuracy=float(balanced_accuracy_score(y,p))))
    for f in FAMILIES:
        for d in (0,24):
            last=(10 if f=='motion_direction' else 4)+d;before=last-2 if f!='motion_direction' else last-1
            specs=[('sample_angle',2),('sample_angle',before),('label',last)] if f!='motion_direction' else [('direction',5),('label_motion',9),('label_motion',last),('counts',9),('counts',last)]
            if f=='orientation_binding':specs.insert(2,('right_angle',before))
            specs=list(dict.fromkeys(specs))
            for kind in KINDS:
                for tag,t in specs:
                    sets=[select(cache[s,f,d],t,kind) for s in ('train','val','test')];xs=[z[0] for z in sets];ys=[target(z[1],tag) for z in sets]
                    fitted=fit(xs[0],ys[0],xs[1],ys[1]);p=predict(fitted,xs[2]);values=score(ys[2],p,tag)
                    record=dict(family=f,delay=d,kind=kind,target=tag,time=t,n=len(values),metric='axial_MAE_degrees' if 'angle' in tag else 'MSE_count_fraction' if tag=='counts' else 'accuracy',value=float(values.mean()),CI95=uncertainty(values),ridge_alpha=fitted[3])
                    if tag=='counts':
                        # Nuisance baseline conditions on the final direction at fixed elapsed time.
                        final=np.array([r['final_direction'] for r in sets[0][1]]);means=np.stack([ys[0][final==k].mean(0) if np.any(final==k) else ys[0].mean(0) for k in range(4)])
                        residual=[y-means[[r['final_direction'] for r in z[1]]] for y,z in zip(ys,sets)]
                        ff=fit(xs[0],residual[0],xs[1],residual[1]);pp=predict(ff,xs[2]);base=float(np.mean(residual[2]**2));err=float(np.mean((residual[2]-pp)**2));record.update(final_direction_baseline_mse=base,conditional_residual_mse=err,conditional_R2=1-err/max(base,1e-9))
                    probes.append(record);predictions[f'{f}_D{d}_{kind}_{tag}_t{t}']=p
                    check()
    # A single decoder fitted at each time and evaluated at every other time.
    for f,tag,times in [('orientation_single','sample_angle',[2,8,14,26,27]),('motion_direction','label_motion',[3,5,9,15,27,33])]:
        for kind in ('R','A'):
            matrix=[]
            for train_t in times:
                a,ar=select(cache['train',f,24],train_t,kind);v,vr=select(cache['val',f,24],train_t,kind);fitted=fit(a,target(ar,tag),v,target(vr,tag));row=[]
                for test_t in times:
                    x,r=select(cache['test',f,24],test_t,kind);row.append(float(score(target(r,tag),predict(fitted,x),tag).mean()))
                matrix.append(row)
            temporal[f'{f}_{kind}']=dict(times=times,values=matrix,metric='degrees' if 'angle' in tag else 'accuracy');check()
    # Common train-fitted population bases; label information is never used to fit PCA.
    atlas=[];basis_info={};cluster_info={}
    for kind in KINDS:
        trains=np.concatenate([cache['train',f,d][0][kind] for f in FAMILIES for d in (0,24)]).astype('float32')
        ix=rng.choice(len(trains),min(6000,len(trains)),replace=False);fitx=trains[ix];mean=fitx.mean(0);scale=fitx.std(0);scale[scale<1e-5]=1
        pca=PCA(n_components=24,svd_solver='randomized',random_state=83);pca.fit((fitx-mean)/scale);basis_info[kind]=dict(explained_variance=pca.explained_variance_ratio_.tolist(),fit_rows=len(fitx),fit='training episodes only')
        np.savez_compressed(HERE/f'pca_{kind}.npz',mean=mean,scale=scale,components=pca.components_,explained_variance=pca.explained_variance_ratio_)
        embed=[];metadata=[]
        for f in FAMILIES:
            data,rs=cache['test',f,24];take=[i for i,r in enumerate(rs) if r['episode']<24]
            embed.append(pca.transform((data[kind][take].astype('float32')-mean)/scale));metadata.extend(rs[i] for i in take)
        z=np.concatenate(embed);trainz=pca.transform((fitx-mean)/scale);km=KMeans(n_clusters=6,random_state=83,n_init=5).fit(trainz);clusters=km.predict(z)
        cluster_info[kind]=dict(k=6,fit='train-only PCA states; fixed exploratory K',phase_NMI=float(normalized_mutual_info_score([r['phase'] for r in metadata],clusters)),family_NMI=float(normalized_mutual_info_score([r['family'] for r in metadata],clusters)))
        # tSNE has no held-out transform: these are pooled descriptive test trajectories only.
        ts=TSNE(n_components=2,perplexity=30,learning_rate=200,n_iter=700,init='pca',random_state=83).fit_transform(z)
        for i,r in enumerate(metadata):atlas.append(dict(r,kind=kind,pca=z[i,:2].tolist(),tsne=ts[i].tolist(),cluster=int(clusters[i])))
        fig,axs=plt.subplots(1,2,figsize=(11,4));phases=sorted(set(r['phase'] for r in metadata));colors=plt.cm.tab10(np.linspace(0,1,len(phases)))
        for name,col in zip(phases,colors):
            mask=np.array([r['phase']==name for r in metadata]);axs[0].scatter(z[mask,0],z[mask,1],s=4,alpha=.55,color=col,label=name);axs[1].scatter(ts[mask,0],ts[mask,1],s=4,alpha=.55,color=col)
        axs[0].set_title(f'{kind}: shared train-fitted PCA');axs[1].set_title('tSNE: descriptive view');axs[0].legend(fontsize=7,ncol=2);figure(fig,'embedding_'+kind);check()
    # Held-out channel tuning: single-feature sinusoid explains channel activity at fixed phase.
    tuning={};fig,axs=plt.subplots(1,2,figsize=(12,3.7))
    for ax,t in zip(axs,(2,26)):
        a,ar=select(cache['train','orientation_single',24],t,'channel_R');b,br=select(cache['test','orientation_single',24],t,'channel_R');x=target(ar,'sample_angle');v=target(br,'sample_angle');m=Ridge(alpha=1.).fit(x,a);p=m.predict(v);den=np.mean((b-a.mean(0))**2,0);r2=1-np.mean((b-p)**2,0)/np.maximum(den,1e-8);tuning[str(t)]=r2.tolist();ax.bar(np.arange(64),r2,color=['#227e84']*51+['#c97649']*13);ax.axhline(0,color='#333',lw=.7);ax.set_title('Sample' if t==2 else 'After 24 blanks');ax.set_xlabel('E channels 0–50 / I channels 51–63');ax.set_ylabel('Held-out angle tuning R²')
    figure(fig,'channel_tuning')
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for row,t in enumerate((2,26)):
        for col,kind in enumerate(('H','R','A')):
            _,rs=select(cache['test','orientation_single',24],t,kind);truth=np.degrees([r['sample_angle'] for r in rs]);p=predictions[f'orientation_single_D24_{kind}_sample_angle_t{t}'];decoded=np.degrees(np.mod(np.arctan2(p[:,1],p[:,0])/2,np.pi));ax=axs[row,col];ax.scatter(truth,decoded,s=13,c=truth,cmap='hsv',vmin=0,vmax=180);ax.plot([0,180],[0,180],color='#666',lw=.8);ax.set_xlim(0,180);ax.set_ylim(0,180);ax.set_title(f'{kind}: '+('sample' if t==2 else '24 blanks'));ax.set_xlabel('True axial orientation (degrees)');ax.set_ylabel('Held-out decoded orientation')
    figure(fig,'orientation_decoding')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axs,['orientation_single_R','motion_direction_R']):
        data=temporal[key];mat=np.array(data['values']);im=ax.imshow(mat,cmap='magma_r' if data['metric']=='degrees' else 'viridis',vmin=0,vmax=45 if data['metric']=='degrees' else 1);ax.set_xticks(range(len(data['times'])),data['times']);ax.set_yticks(range(len(data['times'])),data['times']);ax.set_xlabel('Test frame');ax.set_ylabel('Fit frame');ax.set_title('Orientation: axial error (degrees)' if data['metric']=='degrees' else 'Motion winner: accuracy');fig.colorbar(im,ax=ax,shrink=.8)
    figure(fig,'temporal_generalization')
    # Spatial site map is descriptive, averaged within independent angle bins at one fixed time.
    data,rs=cache['test','orientation_single',24];fig,axs=plt.subplots(1,4,figsize=(12,3));maps=[]
    for q,ax in enumerate(axs):
        ids=[i for i,r in enumerate(rs) if r['time']==26 and int(r['sample_angle']/np.pi*4)%4==q];v=data['site_R'][ids].mean(0).reshape(13,13);maps.append(v);ax.set_title(f'{45*q}–{45*(q+1)}°; n={len(ids)}');ax.axis('off')
    lo=float(np.min(maps));hi=float(np.max(maps))
    for ax,v in zip(axs,maps):im=ax.imshow(v,cmap='viridis',vmin=lo,vmax=hi)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.65,label='Mean rate, shared scale')
    figure(fig,'spatial_maps')
    result=dict(status='completed',checkpoint_step=8400,behavior=behavior,probes=probes,temporal_generalization=temporal,pca=basis_info,clusters=cluster_info,channel_tuning_R2=tuning,analysis_seconds=time.time()-started,limitations=['3x3 pooling discards within-bin feature detail; negative probes do not establish erasure.','Cue appearance and task phase are coupled by existing tasks. No independent semantic cue claim.','tSNE is descriptive and fitted on displayed test states, not a held-out classifier.','Cluster associations use correlated time points and are descriptive; no cell-type claim.','Evidence conditional R2 removes training-estimated final-direction means at fixed elapsed time but other schedule confounds remain.','One checkpoint and small held-out pilot; successful decoding is accessibility, not causal use.'])
    save('results.json',result);save('atlas_points.json',atlas);np.savez_compressed(HERE/'probe_predictions.npz',**predictions)
    render(result,atlas)
def render(result,atlas):
    def embed(name):return 'data:image/svg+xml;base64,'+base64.b64encode((HERE/'figures'/f'{name}.svg').read_bytes()).decode()
    rows=''.join(f"<tr><td>{p['family'].replace('orientation_','')} D{p['delay']}</td><td>{p['kind']}</td><td>{p['target']}, frame {p['time']}</td><td>{p['value']:.3f}</td><td>{p['CI95'][0]:.3f}–{p['CI95'][1]:.3f}</td></tr>" for p in result['probes'])
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Latent Dynamics | Visual Working Memory</title><style>
    :root{--ink:#17323c;--teal:#227e84;--paper:#fbfaf5}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}header{padding:65px 8vw 40px;background:#17323c;color:#f8f5eb}header p{max-width:800px;color:#c4dadb}h1{font:clamp(40px,6vw,74px)/1.06 Georgia,serif;font-weight:400;margin:12px 0 24px}h2{font:34px Georgia,serif;font-weight:400}main{max-width:1200px;margin:auto;padding:35px 35px 80px}.kicker{letter-spacing:.18em;font-size:12px;text-transform:uppercase}.card{background:white;padding:25px;border:1px solid #d6deda;border-radius:12px;margin:24px 0}img{width:100%;height:auto}select,button{font:inherit;padding:7px;border:1px solid #bacbc6;border-radius:5px;background:white;margin:4px}canvas{width:100%;height:480px;display:block}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e0e5de}.scroll{overflow:auto;max-height:600px}a{color:var(--teal)}.note{color:#657b7e;font-size:14px}math{font-size:1.15em}code{background:#eef1eb;padding:2px 5px}.legend{min-height:32px;font-size:13px}</style>
    <header><span class="kicker">Laboratory atlas · frozen checkpoint 8400</span><h1>What is held<br>in a memory state?</h1><p>Follow sensory fields, recurrent activity and adaptation through encoding, blank intervals and decisions. These are measurements of an existing trained network, with no new teaching or weight changes.</p></header><main>
    <h2>A state space, several questions</h2><p>We distinguish the current visual field <i>H</i>, firing rates <i>R</i>, adaptation <i>A</i>, and the final 128-dimensional decision representation. Each full spatial field has shape 64 × 13 × 13. For population analysis we preserve a coarse 3 × 3 spatial grid per channel: 576 features. The ordinary model still uses its complete state.</p>
    <div class="card"><math display="block"><msub><mi>R</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><msub><mi>R</mi><mi>t</mi></msub><mo>+</mo><mi>α</mi><mtext>ReLU</mtext><mo>(</mo><mi>W</mi><msub><mi>z</mi><mi>t</mi></msub><mo>+</mo><mi>K</mi><mo>∗</mo><msub><mi>R</mi><mi>t</mi></msub><mo>−</mo><mi>g</mi><msub><mi>A</mi><mi>t</mi></msub><mo>+</mo><mi>b</mi><mo>)</mo></math><p class="note">Adaptation is an internal dynamical variable. Decoding it here does not make it available to the deployed classifier.</p></div>
    <h2>Explore the population geometry</h2><p>Every point is a frame from an independent held-out episode; lines connect a few example trajectories. Color the same geometry by phase, remembered orientation or motion direction. PCA is fitted only on training episodes. tSNE describes the displayed trajectories and has no held-out transform.</p>
    <div class="card"><label>State <select id="kind"><option>R</option><option>H</option><option>A</option><option>output</option></select></label><label>Projection <select id="projection"><option value="pca">PCA</option><option value="tsne">tSNE</option></select></label><label>Color <select id="color"><option>phase</option><option>family</option><option>sample_angle</option><option>direction</option><option>label</option><option>cluster</option><option>time</option></select></label><label>Task <select id="family"><option value="all">All</option><option>orientation_single</option><option>orientation_binding</option><option>motion_direction</option></select></label><canvas id="plot"></canvas><div id="legend" class="legend"></div><p class="note">PCA preserves global linear geometry more directly. tSNE island sizes and distances do not measure memory capacity. Future change labels before the probe are not an observable signal.</p></div>
    <h2>Can the same decoder read memory later?</h2><p>Rows fit a linear readout at one frame; columns test it at another. Orientation uses the axial target (cos 2θ, sin 2θ): 0° and 180° are equivalent. Low cross-time error supports a stable accessible code; a good diagonal with worse off-diagonal entries suggests changing readout geometry.</p><img src="TEMPORAL">
    <h2>Spatial and channel organization</h2><p>Rate maps average across channels after 24 blanks, grouped by remembered angle. Channel tuning fits a sinusoid on training episodes and measures explained activity variance on held-out episodes. Negative R² is retained. Channel indices reflect the implemented E/I sign constraint, not discovered biological cell types.</p><img src="SPATIAL"><img src="TUNING">
    <h2>Readout evidence</h2><p>Below are held-out linear probes. Angle values are mean absolute error in degrees (uniform-guess reference 45°); label/direction values are accuracy; counts use mean squared error of centered duration fractions. Intervals resample independent episodes within each cell. No statistical inference treats the many frames as independent animals or trials.</p><div class="card scroll"><table><thead><tr><th>Task</th><th>State</th><th>Target</th><th>Value</th><th>95% interval</th></tr></thead><tbody>ROWS</tbody></table></div>
    <h2>What this can establish</h2><p>Recoverable information need not be used by the deployed decision. A failed coarse linear probe does not demonstrate erasure. Existing cues are coupled to phase, so cue-semantic selectivity cannot be isolated here. Duration evidence is tested at fixed elapsed time, with an additional final-direction-conditioned regression in the saved results. Six exploratory state clusters are fitted without labels; associations are descriptive.</p><p><a href="report.md">Full report</a> · <a href="RESEARCH.md">Neuroscience and methods</a> · <a href="results.json">Numerical results</a> · <a href="config.json">Protocol and checkpoint</a></p></main>
    <script>const points=POINTS;const colors=['#227e84','#c97649','#775ba5','#bca438','#42936c','#b95162','#457dae','#8b7067'];function draw(){let k=document.getElementById('kind').value,pr=document.getElementById('projection').value,c=document.getElementById('color').value,f=document.getElementById('family').value;let p=points.filter(x=>x.kind===k&&(f==='all'||x.family===f));const cv=document.getElementById('plot'),ctx=cv.getContext('2d');cv.width=cv.clientWidth*devicePixelRatio;cv.height=480*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);let W=cv.clientWidth,H=480;ctx.clearRect(0,0,W,H);let xs=p.map(x=>x[pr][0]),ys=p.map(x=>x[pr][1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);function xy(q){return [25+(q[pr][0]-xmin)/(xmax-xmin||1)*(W-50),H-25-(q[pr][1]-ymin)/(ymax-ymin||1)*(H-50)]}let cats=[...new Set(p.map(x=>x[c]))].sort();let continuous=['sample_angle','time'].includes(c);let groups={};p.filter(q=>q.episode<3).forEach(q=>(groups[q.group]??=[]).push(q));ctx.strokeStyle='#17323c35';for(let g of Object.values(groups)){g.sort((a,b)=>a.time-b.time);ctx.beginPath();g.forEach((q,i)=>{let a=xy(q);i?ctx.lineTo(...a):ctx.moveTo(...a)});ctx.stroke()}for(let q of p){let a=xy(q),v=q[c];ctx.fillStyle=v===null?'#c7cbc5':continuous?`hsl(${(c==='sample_angle'?v/Math.PI:v/34)*300},65%,43%)`:colors[cats.indexOf(v)%colors.length];ctx.globalAlpha=.68;ctx.beginPath();ctx.arc(...a,2.3,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;document.getElementById('legend').innerText=continuous?(c==='sample_angle'?'Orientation hue: 0 to 180 degrees; gray = not applicable':'Time hue: instruction to final report'):cats.map((v,i)=>v===null?'not applicable':String(v)).join(' · ')}document.querySelectorAll('select').forEach(x=>x.onchange=draw);window.onresize=draw;draw();</script></html>'''
    doc=doc.replace('<h2>Readout evidence</h2>','<h2>Orientation identifiability</h2><p>Each dot is one held-out episode. The same actual rendered orientation is compared against an independently fitted axial readout. The 0/180° boundary wraps: near-boundary points should be read using circular distance.</p><img src="'+embed('orientation_decoding')+'"><h2>Readout evidence</h2>')
    doc=doc.replace('<option>time</option>','<option>time</option><option>evidence_margin</option>')
    doc=doc.replace('<canvas id="plot">','<label>Phase <select id="phase"><option value="all">All</option><option>sample</option><option>blank</option><option>query</option><option>probe</option><option>moving</option><option>report</option></select></label><canvas id="plot">')
    doc=doc.replace("let p=points.filter(x=>x.kind===k&&(f==='all'||x.family===f));","let stage=document.getElementById('phase').value;let p=points.filter(x=>x.kind===k&&(f==='all'||x.family===f)&&(stage==='all'||x.phase===stage));if(!p.length){document.getElementById('legend').innerText='No frames for this task and phase.';return;}")
    doc=doc.replace("['sample_angle','time'].includes(c)","['sample_angle','time','evidence_margin'].includes(c)")
    doc=doc.replace("c==='sample_angle'?v/Math.PI:v/34","c==='sample_angle'?v/Math.PI:c==='evidence_margin'?v:v/34")
    doc=doc.replace("'Time hue: instruction to final report'","c==='evidence_margin'?'Duration margin hue: top count minus second count, divided by eight':'Time hue: instruction to final report'")
    doc=doc.replace('TEMPORAL',embed('temporal_generalization')).replace('SPATIAL',embed('spatial_maps')).replace('TUNING',embed('channel_tuning')).replace('ROWS',rows).replace('POINTS',json.dumps(atlas,separators=(',',':')))
    (HERE/'index.html').write_text(doc,encoding='utf-8')
    highlights=[]
    for p in result['probes']:
        if p['kind']=='R' and p['delay']==24 and p['time'] in (26,33,28):highlights.append(f"| {p['family']} | {p['target']} | {p['time']} | {p['value']:.4f} | {p['CI95']} |")
    report='# Frozen latent-state atlas\n\nMeasured attention checkpoint 8400. No main-model training or new task teaching.\n\n[Open interactive atlas](index.html) · [Research basis](RESEARCH.md) · [Complete results](results.json) · [Protocol](config.json)\n\n## Rate-state held-out results at long delay\n\n| Task | Target | Frame | Value | 95% episode interval |\n|---|---|---:|---:|---|\n'+'\n'.join(highlights)+'\n\nAngles are axial mean absolute errors in degrees; classification values are accuracy; evidence values are MSE. See results.json for all states, conditions, selected ridge penalties, conditional evidence residual R², behavioral accuracy and temporal generalization.\n\n## Interpretation limits\n\n'+'\n'.join('- '+s for s in result['limitations'])+'\n\nUMAP is not installed; PCA and tSNE were used without changing the active training environment. Exploratory K=6 state segmentation was fitted on training PCA coordinates. Channel/site summaries complement the coarse spatial pooling but cannot reconstruct fine within-bin patterns.\n'
    (HERE/'report.md').write_text(report,encoding='utf-8')
if __name__=='__main__':main()
