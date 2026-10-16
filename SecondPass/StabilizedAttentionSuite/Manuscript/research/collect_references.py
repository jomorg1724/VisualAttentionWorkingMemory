"""Retrieve bibliographic metadata; never treat metadata as full-text verification."""
import concurrent.futures,datetime,difflib,json,re,time
from pathlib import Path
import requests

HERE=Path(__file__).parent
ITEMS='''posner1980|1980|Orienting of attention
desimone1995|1995|Neural mechanisms of selective visual attention
moran1985|1985|Selective attention gates visual processing in the extrastriate cortex
reynolds1999|1999|Competitive mechanisms subserve attention in macaque areas V2 and V4
reynolds2009|2009|The normalization model of attention
moore2003|2003|Selective gating of visual signals by microstimulation of frontal cortex
muller2005|2005|Microstimulation of the superior colliculus focuses attention without moving the eyes
cavanaugh2004|2004|Subcortical modulation of attention counters change blindness
zenon2012|2012|Attention deficits without cortical neuronal deficits
krauzlis2013|2013|Superior colliculus and visual spatial attention
herman2018|2018|A neural correlate of perceptual choice in the primate superior colliculus
sridharan2014|2014|Does the superior colliculus control perceptual sensitivity or choice bias during attention? Evidence from a multialternative decision framework
fecteau2006|2006|Salience, relevance, and firing: a priority map for target selection
bisley2010|2010|Attention, intention, and priority in the parietal lobe
adelson1985|1985|Spatiotemporal energy models for the perception of motion
watson1985|1985|Model of human visual-motion sensing
simoncelli1998|1998|A model of neuronal responses in visual area MT
newsome1988|1988|A selective impairment of motion perception following lesions of the middle temporal visual area (MT)
salzman1990|1990|Cortical microstimulation influences perceptual judgements of motion direction
britten1992|1992|The analysis of visual motion: a comparison of neuronal and psychophysical performance
pack2001|2001|Temporal dynamics of a neural solution to the aperture problem in visual area MT of macaque brain
rust2006|2006|How MT cells analyze the motion of visual patterns
fuster1971|1971|Neuron activity related to short-term memory
funahashi1989|1989|Mnemonic coding of visual space in the monkey's dorsolateral prefrontal cortex
miller2001|2001|An integrative theory of prefrontal cortex function
compte2000|2000|Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model
wang2001|2001|Synaptic reverberation underlying mnemonic persistent activity
mongillo2008|2008|Synaptic theory of working memory
stokes2013|2013|Dynamic coding for cognitive control in prefrontal cortex
rose2016|2016|Reactivation of latent working memories with transcranial magnetic stimulation
lundqvist2016|2016|Gamma and beta bursts underlie working memory
harrison2009|2009|Decoding reveals the contents of visual working memory in early visual areas
christophel2017|2017|The distributed nature of working memory
heeger1992|1992|Normalization of cell responses in cat striate cortex
carandini2012|2012|Normalization as a canonical neural computation
rao1999|1999|Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects
friston2005|2005|A theory of cortical responses
horn1981|1981|Determining optical flow
lucas1981|1981|An iterative image registration technique with an application to stereo vision
barron1994|1994|Performance of optical flow techniques
black1996|1996|The robust estimation of multiple motions: parametric and piecewise-smooth flow fields
brox2004|2004|High accuracy optical flow estimation based on a theory for warping
farneback2003|2003|Two-frame motion estimation based on polynomial expansion
sun2014|2014|A quantitative analysis of current practices in optical flow estimation and the principles behind them
ilg2017|2017|FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
sun2018|2018|PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
teed2020|2020|RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
shi2015|2015|Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
hochreiter1997|1997|Long short-term memory
cho2014|2014|Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
vaswani2017|2017|Attention Is All You Need
dosovitskiy2021|2021|An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
ba2016|2016|Layer Normalization
pascanu2013|2013|On the difficulty of training recurrent neural networks
kingma2015|2015|Adam: A Method for Stochastic Optimization
jaderberg2015|2015|Spatial Transformer Networks
arbelaez2011|2011|Contour Detection and Hierarchical Image Segmentation
wu2018|2018|Group Normalization
treisman1980|1980|A feature-integration theory of attention
ma2006|2006|Bayesian inference with probabilistic population codes'''

ARXIV={'shi2015':'1506.04214','cho2014':'1406.1078','vaswani2017':'1706.03762',
 'dosovitskiy2021':'2010.11929','ba2016':'1607.06450','pascanu2013':'1211.5063',
 'kingma2015':'1412.6980','jaderberg2015':'1506.02025','teed2020':'2003.12039','wu2018':'1803.08494'}

def norm(s):return re.sub('[^a-z0-9]','',s.lower())
def retrieve(line):
    key,year,title=line.split('|');out=dict(id=key,expected_year=int(year),requested_title=title)
    try:
        if key in ARXIV:
            from bs4 import BeautifulSoup
            url='https://arxiv.org/abs/'+ARXIV[key]
            response=requests.get(url,timeout=45);response.raise_for_status();soup=BeautifulSoup(response.text,'html.parser')
            actual=soup.select_one('h1.title').get_text(' ',strip=True).removeprefix('Title:').strip()
            authors=soup.select_one('.authors').get_text(' ',strip=True).removeprefix('Authors:').strip()
            abstract=soup.select_one('blockquote.abstract').get_text(' ',strip=True).removeprefix('Abstract:').strip()
            out.update(title=actual,authors=authors,year=int(year),url=url,abstract=abstract,verification='Primary preprint abstract and metadata retrieved',match=difflib.SequenceMatcher(None,norm(title),norm(actual)).ratio())
        else:
            response=requests.get('https://api.crossref.org/works',params={'query.title':title,'rows':4},headers={'User-Agent':'VisualAttentionManuscript/1.0 (local research bibliography)'},timeout=45)
            response.raise_for_status();items=response.json()['message']['items']
            m=max(items,key=lambda m:difflib.SequenceMatcher(None,norm(title),norm(m.get('title',[''])[0])).ratio())
            actual=m.get('title',[''])[0]
            authors='; '.join(' '.join([a.get('given',''),a.get('family','')]).strip() for a in m.get('author',[]))
            date=m.get('published',m.get('issued',{})).get('date-parts',[[int(year)]])[0][0]
            out.update(title=actual,authors=authors,year=date,doi=m['DOI'],url='https://doi.org/'+m['DOI'],journal=m.get('container-title',[''])[0],
                abstract=m.get('abstract',''),verification='Crossref publisher-deposited metadata'+('; abstract available' if m.get('abstract') else '; no abstract in this record'),
                match=difflib.SequenceMatcher(None,norm(title),norm(actual)).ratio(),metadata_url=response.url)
        out['retrieved_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    except Exception as e:out['error']=str(e)
    print(key,out.get('match'),out.get('title',out.get('error')),flush=True)
    (HERE/'reference_records').mkdir(exist_ok=True)
    (HERE/'reference_records'/f'{key}.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    return out

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(retrieve,ITEMS.splitlines()))
    (HERE/'references_retrieved.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
