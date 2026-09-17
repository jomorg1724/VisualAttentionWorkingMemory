"""Small sequential PAV comparison with a single durable wall-clock budget."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys
import json
import csv
import time
import random
import hashlib
import threading
import subprocess
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(job):
    deadline=job['deadline'];timer=threading.Timer(max(0,deadline-time.time()),lambda:os._exit(124));timer.daemon=True;timer.start()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PreAttentiveVision.models import build_encoder
    from PreAttentiveVision.decoder import PairClassifier
    from PreAttentiveVision.stimuli import PairStream,FAMILIES
    from PreAttentiveVision.evaluate import calibrate,evaluate_rows
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    cfg=job['config'];out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    for name,digest in job['source_hashes'].items():
        if sha(HERE/name)!=digest:raise RuntimeError('Source changed during fixed run:'+name)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    encoder=build_encoder(cfg['model'])
    # All five decoders begin with identical weights within each training seed.
    torch.manual_seed(seed+500000);torch.cuda.manual_seed_all(seed+500000)
    model=PairClassifier(encoder).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    stream=PairStream(seed+100000,split='train')
    start_step=0
    if job.get('checkpoint'):
        cp=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
        if cp['version']!='pav_checkpoint_v1' or cp['config']!=cfg or cp['source_hashes']!=job['source_hashes']:raise RuntimeError('Checkpoint incompatible')
        model.load_state_dict(cp['model'],strict=True)
        optimizer.load_state_dict(cp['optimizer']);stream.load_state_dict(cp['stream'])
        random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy']);torch.set_rng_state(cp['rng']['torch']);torch.cuda.set_rng_state(cp['rng']['cuda'])
        start_step=cp['step']

    def save(step):
        path=out/f'checkpoint_{step:06d}.pt'
        if path.exists():return str(path)
        state=dict(version='pav_checkpoint_v1',config=cfg,source_hashes=job['source_hashes'],step=step,
            model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
        identity=dict(step=step,file=path.name,sha256=sha(path),fresh_training_pairs=step*cfg['batch_size'])
        with (out/'checkpoint_index.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(identity)+'\n')
        return str(path)

    def batch():
        xs=[];ys=[]
        for family in FAMILIES:
            x,y,_=stream.batch(cfg['batch_size']//4,family=family);xs.append(x);ys.append(y)
        return torch.cat(xs).cuda(),torch.cat(ys).cuda()

    started=time.monotonic();kind=job['kind'];timings=[];step=start_step
    if kind in ('profile','train'):
        model.train();torch.cuda.reset_peak_memory_stats()
        if not job.get('checkpoint'):save(0)
        metrics=out/'metrics.csv'
        with metrics.open('a',newline='',encoding='utf-8') as f:
            fields=['step','fresh_pairs','loss','grad_norm','train_accuracy','step_seconds','wall_utc']
            writer=csv.DictWriter(f,fieldnames=fields)
            if f.tell()==0:writer.writeheader()
            for step in range(start_step+1,job['target']+1):
                if time.time()>deadline-4:step-=1;break
                tick=time.monotonic();x,y=batch();optimizer.zero_grad(set_to_none=True)
                logits=model(x);loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();torch.cuda.synchronize();elapsed=time.monotonic()-tick;timings.append(elapsed)
                writer.writerow(dict(step=step,fresh_pairs=step*cfg['batch_size'],loss=float(loss),grad_norm=float(norm),
                    train_accuracy=float((logits.argmax(1)==y).float().mean()),step_seconds=elapsed,
                    wall_utc=datetime.now(timezone.utc).isoformat()));f.flush()
                if step%128==0:print(json.dumps(dict(step=step,loss=float(loss),fresh_pairs=step*cfg['batch_size'])),flush=True)
                if step%256==0:save(step)
        checkpoint=save(step)
        result=dict(status='completed' if step==job['target'] else 'budget_stopped',step=step,checkpoint=checkpoint,
            worker_seconds=time.monotonic()-started,fresh_pairs=step*cfg['batch_size'])
        if kind=='profile':
            model.eval();inference=[]
            for _ in range(8):
                tick=time.monotonic();x,_=batch()
                with torch.no_grad():model(x)
                torch.cuda.synchronize();inference.append(time.monotonic()-tick)
            result.update(train_step_median=float(np.median(timings[8:])),train_step_p90=float(np.quantile(timings[8:],.9)),
                eval_batch_median=float(np.median(inference[2:])),peak_vram_bytes=torch.cuda.max_memory_allocated(),
                encoder_params=sum(p.numel() for p in model.encoder.parameters()),decoder_params=sum(p.numel() for p in model.decoder.parameters()),
                total_params=sum(p.numel() for p in model.parameters()),trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                profiling_updates=step,profiling_inference_pairs=8*cfg['batch_size'])
    elif kind=='eval':
        model.eval();split=job['split'];eval_stream=PairStream(job['eval_seed'],split=split);rows=[]
        for family in FAMILIES:
            for offset in range(0,job['n_per_family'],cfg['batch_size']):
                if time.time()>deadline-8:raise TimeoutError('Evaluation budget reached')
                x,y,metadata=eval_stream.batch(min(cfg['batch_size'],job['n_per_family']-offset),family=family)
                with torch.no_grad():scores=model(x.cuda()).softmax(1)[:,1].cpu().numpy()
                for score,label,meta in zip(scores,y.tolist(),metadata):
                    rows.append(dict(model=cfg['model'],seed=seed,step=start_step,split=split,
                        family=family,difficulty=meta['difficulty'],label=label,prob_change=float(score),
                        base_id=meta.get('base_id'),trial_id=meta['trial_id'],metadata_json=json.dumps(meta)))
        if split=='val':
            selection=calibrate(rows);threshold=selection['threshold']
        else:selection=job['calibration'];threshold=selection['threshold']
        for row in rows:row.update(pred_default=int(row['prob_change']>=.5),pred_calibrated=int(row['prob_change']>=threshold))
        with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result=evaluate_rows(rows,threshold=threshold,resamples=1000 if split=='test' else 0,seed=77201)
        result['default_threshold']=evaluate_rows(rows,threshold=.5,resamples=0,seed=77201)
        result['engineering_goal_met']=bool(result['macro']['balanced_accuracy']>=.90 and all(v['overall']['balanced_accuracy']>=.85 for v in result['families'].values()))
        result.update(status='completed',step=start_step,split=split,calibration=selection,
            worker_seconds=time.monotonic()-started,predictions=str(out/'predictions.csv'),
            evaluated_pairs=len(rows),checkpoint=job['checkpoint'])
        write(out/'summary.json',result)
    else:raise ValueError(kind)
    write(job['result'],result);timer.cancel()


def orchestrate(root,seconds=3600,batch_size=32):
    import math
    from PreAttentiveVision.models import MODEL_NAMES
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Existing budget cannot silently restart or renew')
    now=time.time();deadline=now+seconds
    hashes={name:sha(HERE/name) for name in ['models.py','stimuli.py','decoder.py','evaluate.py','train.py']}
    base=dict(batch_size=batch_size,lr=.001,weight_decay=.0001,grad_clip=5.,fp32=True,amp=False,
        loss='equal-family mean episode crossentropy',input_normalization='(x-.5)/.5',decoder='symmetric_multiscale_v1',decoder_seed_offset=500000,
        engineering_goal='macro validation BA>=.90 and each family BA>=.85; heuristic goal, no earlystop; locked test confirmation reported')
    ledger=dict(started_unix=now,deadline_unix=deadline,hard_seconds=seconds,source_hashes=hashes,active=None,events=[])
    write(root/'budget.json',ledger)
    aggregate=dict(status='profiling',config=dict(models=MODEL_NAMES,batch_size=batch_size,budget_seconds=seconds),profiles={},runs=[])
    write(root/'aggregate.json',aggregate);write(HERE/'results.json',dict(aggregate,run_root=str(root)))

    def publish():
        write(root/'aggregate.json',aggregate);write(HERE/'results.json',dict(aggregate,run_root=str(root)))

    def launch(kind,cfg,out,**kwargs):
        number=len(ledger['events']);job_path=root/f'job_{number:03d}.json';result=root/f'job_{number:03d}_result.json'
        end=min(deadline,time.time()+180 if kind=='profile' else deadline)
        job=dict(kind=kind,config=cfg,out=str(out),deadline=end,result=str(result),source_hashes=hashes,**kwargs)
        write(job_path,job);log_path=root/f'worker_{number:03d}.log'
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'_worker',str(job_path)]
        start=time.time()
        with log_path.open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            ledger['active']=dict(pid=process.pid,kind=kind,model=cfg['model'],seed=cfg['seed'],started_unix=start,deadline_unix=end,log=str(log_path))
            write(root/'budget.json',ledger)
            try:code=process.wait(timeout=max(0,end-time.time()))
            except subprocess.TimeoutExpired:process.kill();process.wait();code=124
            finally:
                if process.poll() is None:process.kill();process.wait()
        elapsed=time.time()-start
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=elapsed));ledger['active']=None;write(root/'budget.json',ledger)
        if code!=0:raise RuntimeError(f'{kind} {cfg["model"]} failed/limited exit{code}; see {log_path}')
        return read(result),elapsed

    try:
        for model in MODEL_NAMES:
            cfg=dict(base,model=model,seed=991,purpose='accounted_throughput_profile')
            profile,_=launch('profile',cfg,root/'profiles'/model,target=32)
            aggregate['profiles'][model]=profile;publish()
        profiles=aggregate['profiles'];train_cost=sum(max(v['train_step_median']*1.3,v['train_step_p90']) for v in profiles.values())
        eval_cost=sum(v['eval_batch_median']*1.3 for v in profiles.values())
        remaining=deadline-time.time()-90
        def estimate(seeds,updates):
            chunks=math.ceil(updates/512)
            # Worker startup/checkpoint/CI allowance plus measured end-to-end batches.
            return seeds*(updates*train_cost+(chunks*math.ceil(800/batch_size)+math.ceil(3200/batch_size))*eval_cost+chunks*5*6+5*20)
        candidates=[(2,u) for u in [4096,3072,2048,1536]]+[(1,u) for u in [4096,3072,2048,1536,1024,512]]
        chosen=next(((s,u) for s,u in candidates if estimate(s,u)<remaining*.9),None)
        if chosen is None:raise RuntimeError('Measured throughput cannot fit even all-five512updates and evaluations under remaining cap')
        seed_count,updates=chosen;seeds=[20261,20262][:seed_count]
        aggregate['config'].update(seeds=seeds,updates=updates,training_pairs_per_model_seed=updates*batch_size,
            val_pairs_per_family=200,test_pairs_per_family=800,recipe=base,
            estimated_remaining_seconds=estimate(seed_count,updates),
            selection='Checkpoint highest macro validation AUC; exact ties earlieststep. One validation-calibrated common threshold. All final test comparisons shown; test not used to extend training.',
            profiling='32 fresh profiling updates/model, separate preserved states, all charged to hard budget',
            uncertainty='Natural CIFAR source-image cluster bootstrap; procedural pair bootstrap; macro averages independent family bootstrap replicates; training seeds reported separately.')
        aggregate['status']='training';publish()
        write(root/'fixed_config.json',aggregate['config'])
        print('FIXED_ALLOCATION '+json.dumps(aggregate['config']),flush=True)
        runs={}
        for seed in seeds:
            for model in MODEL_NAMES:
                rec=dict(model=model,seed=seed,status='pending',step=0,best_step=None,train_pairs=0,training_seconds=0.,val_curve=[],test=None)
                runs[(model,seed)]=rec;aggregate['runs'].append(rec)
        publish()
        for target in range(512,updates+1,512):
            for seed in seeds:
                for model in MODEL_NAMES:
                    record=runs[(model,seed)];run_dir=root/f'{model}_seed{seed}'
                    cfg=dict(base,model=model,seed=seed,purpose='production_comparison')
                    trained,elapsed=launch('train',cfg,run_dir,target=target,checkpoint=record.get('checkpoint'))
                    record.update(status='running',step=trained['step'],checkpoint=trained['checkpoint'],train_pairs=trained['fresh_pairs'])
                    record['training_seconds']+=elapsed;publish()
                    if trained['step']!=target:raise RuntimeError('Budget stopped before equal update endpoint')
                    validation,_=launch('eval',cfg,run_dir/f'eval_val_{target:06d}',checkpoint=record['checkpoint'],split='val',eval_seed=830001,n_per_family=200)
                    record['val_curve'].append(dict(step=target,macro_auc=validation['macro']['auroc'],macro_balanced_accuracy=validation['macro']['balanced_accuracy'],summary=str(run_dir/f'eval_val_{target:06d}'/'summary.json')))
                    if record.get('best_auc') is None or validation['macro']['auroc']>record['best_auc']:
                        record.update(best_step=target,best_auc=validation['macro']['auroc'],best_checkpoint=record['checkpoint'],calibration=validation['calibration'])
                    publish()
        aggregate['status']='final_test';publish()
        for seed in seeds:
            for model in MODEL_NAMES:
                record=runs[(model,seed)];run_dir=root/f'{model}_seed{seed}'
                cfg=dict(base,model=model,seed=seed,purpose='production_comparison')
                test,_=launch('eval',cfg,run_dir/'eval_test',checkpoint=record['best_checkpoint'],split='test',eval_seed=930001,n_per_family=800,calibration=record['calibration'])
                record.update(test=test,status='completed');publish()
        aggregate['status']='completed';aggregate['wall_seconds']=time.time()-now;publish()
        write(root/'exit.json',dict(status='completed',wall_seconds=time.time()-now,deadline_unix=deadline))
    except BaseException as error:
        aggregate['status']='budget_stopped' if time.time()>=deadline-5 else 'failed'
        aggregate['error']=repr(error);aggregate['wall_seconds']=time.time()-now;publish()
        write(root/'exit.json',dict(status=aggregate['status'],error=repr(error),wall_seconds=time.time()-now));raise


if __name__=='__main__':
    if sys.argv[1]=='_worker':worker(read(sys.argv[2]))
    elif sys.argv[1]=='run':orchestrate(sys.argv[2],seconds=int(sys.argv[3]) if len(sys.argv)>3 else 3600)
    else:raise SystemExit('Usage: train.py run RUN_ROOT [TOTAL_SECONDS]')
