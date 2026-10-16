import concurrent.futures,datetime,difflib,json,re,time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
HERE=Path(__file__).parent
def norm(s):return re.sub('[^a-z0-9]','',s.lower())
def recover(row):
    if row.get('match',0)>.9:return row
    title=row['requested_title']
    if row['id']=='herman2018':title='Midbrain activity can explain perceptual decisions during an attention task'
    try:
        response=requests.get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',params={'query':'TITLE:"'+title+'"','format':'json','pageSize':5,'resultType':'core'},timeout=50)
        response.raise_for_status();items=response.json()['resultList']['result']
        if not items:raise ValueError('No title match in Europe PMC')
        m=max(items,key=lambda m:difflib.SequenceMatcher(None,norm(title),norm(m.get('title',''))).ratio())
        match=difflib.SequenceMatcher(None,norm(title),norm(m['title'])).ratio()
        row.update(title=m['title'],authors=m.get('authorString',''),year=int(m['pubYear']),doi=m.get('doi'),
            url='https://doi.org/'+m['doi'] if m.get('doi') else 'https://europepmc.org/article/'+m['source']+'/'+m['id'],
            abstract=BeautifulSoup(m.get('abstractText',''),'html.parser').get_text(' ',strip=True),
            verification='Europe PMC bibliographic record and available abstract retrieved; not full text',match=match,
            metadata_url=response.url,retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        row.pop('error',None)
    except Exception as e:row['recovery_error']=str(e)
    (HERE/'reference_records'/f"{row['id']}.json").write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
    print(row['id'],row.get('match',0),flush=True);time.sleep(.3)
    return row
if __name__=='__main__':
    rows=json.loads((HERE/'references_retrieved.json').read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(recover,rows))
    (HERE/'references_retrieved.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
