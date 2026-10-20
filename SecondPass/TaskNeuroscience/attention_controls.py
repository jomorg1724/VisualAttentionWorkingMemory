"""Verify causal map redistribution on a matched subset; central-sample controls."""
import argparse,json,time,os
import numpy as np
import torch
from .design import *
from .collect import Collector,new_stream,draw,save_npz
from SecondPass.PixelContrastiveVAE.train import configure,GPULock

def query_switches(collector):
    for task in ('memory_guided_search','motion_memory_search'):
        folder=OUT/task/'query_switch';folder.mkdir(exist_ok=True)
        for cell in neural_cells(task):
            path=folder/f'{cell}.npz'
            if path.exists():continue
            stream=new_stream('test',NS+7);pp=[];aa=[];blank=[];metadata=[]
            for scene in range(64):
                collector.status(phase='matched_query_switch',task=task,cell=cell,scene=scene,total=64)
                x,y,m=(stream.paired_queries(cell,scene) if task=='memory_guided_search' else stream.paired_motion_queries(cell,scene))
                begin=m[0]['array_onset'];assert torch.equal(y,torch.arange(4))
                assert all(torch.equal(x[0,begin:],x[i,begin:]) for i in range(1,4)), 'Array changed across queries'
                p,a,_=collector.infer(x,task);pp.append(p);aa.append(a);metadata.append(m)
                xb=x[:1].clone()
                for t,phase in enumerate(phases(task,m[0],len(x[0]))):
                    if phase=='sample':xb[0,t].fill_(128/255)
                pb,ab,_=collector.infer(xb,task);blank.append(pb[0])
            save_npz(path,p=np.array(pp),attention=np.array(aa,dtype=np.float16),blank_p=np.array(blank))
            write(path.with_suffix('.json'),dict(metadata=metadata,namespace=NS+7,identical_search_movies_verified=True,scenes=64,queries_per_scene=4))

def weak_vda(collector):
    from SecondPass.StabilizedAttentionVDA.Analysis.design import movie,bias as vbias,masks,PRIMARY_ARMS
    for size in (4,9,16):
        task=f'vda{size}';folder=OUT/task/'weak_counterfactual';folder.mkdir(exist_ok=True)
        for strength in (0,2,4,8):
            for event in (('catch',) if strength==0 else ('cued','uncued')):
                path=folder/f'{strength}_{event}.npz'
                if path.exists():continue
                pp=[];yy=[];mm=[]
                for start in range(0,128,BATCH):
                    rows=[movie(size,i,strength,event,1.,cue=size-1,namespace=NS+8) for i in range(start,start+BATCH)]
                    x=torch.stack([r[0] for r in rows]);meta=[r[1] for r in rows];values=[]
                    for arm in PRIMARY_ARMS:
                        collector.status(phase='weak_vda_followup',task=task,strength=strength,event=event,arm=arm['name'],completed=start)
                        p,_,_=collector.infer(x,task,bias_value=vbias(meta[0],arm,masks(size)));values.append(p)
                    pp.extend(np.stack(values,1));yy.extend([m['label'] for m in meta]);mm+=meta
                save_npz(path,p=np.array(pp),y=np.array(yy))
                write(path.with_suffix('.json'),dict(metadata=mm,arms=[a['name'] for a in PRIMARY_ARMS],strength=strength,event=event,namespace=NS+8,exploratory_followup=True))

def run():
    configure('cuda');lock=GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:
        collector=Collector(hours=3);write(OUT/'attention_control_worker.json',dict(pid=os.getpid(),started=time.time()))
        selected=[a for a in arms() if a['name'] in ('sham','target_-4_evidence','target_+4_evidence','distractor_-4_evidence','distractor_+4_evidence','global_+4_evidence')]
        for task in TASKS:
            folder=OUT/task/'attention_controls';folder.mkdir(parents=True,exist_ok=True)
            for cell in neural_cells(task):
                path=folder/f'{cell}.npz'
                if path.exists():continue
                stream=new_stream('test',NS+3);allp=[];alla=[];allm=[];ally=[];central=[]
                source=np.load(OUT/task/'interventions'/f'{cell}.npz');names=json.loads((OUT/task/'interventions'/f'{cell}.json').read_text())['arms']
                for start in range(0,32,BATCH):
                    x,y,m=draw(stream,BATCH,task,cell);p=[];aa=[]
                    for arm in selected:
                        collector.status(phase='attention_controls',task=task,cell=cell,arm=arm['name'],completed=start)
                        pp,a,_=collector.infer(x,task,bias_value=bias(task,m,len(x[0]),arm));p.append(pp);aa.append(a)
                        # The exact original paired movies must reproduce their original outcomes.
                        assert np.allclose(pp,source['p'][start:start+BATCH,names.index(arm['name'])],atol=2e-5)
                    allp.extend(np.stack(p,1));alla.extend(np.stack(aa,1));allm+=m;ally.extend(y.tolist())
                    if task in ('memory_guided_search','motion_memory_search'):
                        yy,xx=np.meshgrid(np.arange(10)*10+4.5,np.arange(10)*10+4.5,indexing='ij');mask=((xx-50)**2+(yy-50)**2<16**2).flatten();cc=[]
                        for dose in (-4,4):
                            bb=np.zeros((len(x),len(x[0]),100),np.float32)
                            for i,meta in enumerate(m):
                                ix=[j for j,v in enumerate(phases(task,meta,len(x[0]))) if v=='sample'];bb[i,ix]=mask*dose
                            cp,ca,_=collector.infer(x,task,bias_value=bb);cc.append(cp)
                        central.extend(np.stack(cc,1))
                    write(OUT/'attention_control_status.json',dict(task=task,cell=cell,completed=start+BATCH,total=32,time=time.time()))
                save_npz(path,p=np.array(allp),attention=np.array(alla,dtype=np.float16),y=np.array(ally),central_sample_p=np.array(central))
                write(path.with_suffix('.json'),dict(arms=[a['name'] for a in selected],metadata=allm,central_sample_doses=[-4,4],source_pairing_verified=True))
        query_switches(collector)
        weak_vda(collector)
        write(OUT/'attention_controls_complete.json',dict(time=time.time(),presentations=collector.presentations))
    finally:lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-for-collection',action='store_true');a=p.parse_args()
    if a.wait_for_collection:
        deadline=time.time()+16*3600
        while not (OUT/'collection_complete.json').exists():
            if time.time()>deadline or (OUT/'stop_requested.json').exists():return
            if (OUT/'failure.json').exists():raise RuntimeError('Collection failed; inspect failure.json')
            time.sleep(20)
    run()

if __name__=='__main__':main()
