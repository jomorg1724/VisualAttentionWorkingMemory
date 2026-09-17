"""Two separately interpreted arms; no explicit scalar attention logit priors."""
import numpy as np
from WorkingMemory.TrainingExposure.protocol import recipe as old_recipe,assess,VAL_SEED,TEST_SEED
from PreAttentiveVision.evaluate_multitask import summarize
SPATIAL_VAL_SEED=63973001
SPATIAL_TEST_SEED=64973001

def recipe(arm):
    cfg=old_recipe('control_10');cfg['battery']='old';cfg['updates']=4000
    if arm=='old_unbiased':return cfg
    if arm!='spatial_unbiased':raise ValueError(arm)
    from WorkingMemory.SpatialTaskBattery.stimuli import TASK_CLASSES,TRAIN_CONDITIONS,EVAL_CONDITIONS
    cfg.update(battery='spatial',task_classes=TASK_CLASSES,train_seed=61973001,scheduler_seed=62973001,model_seed=41973001,cells=EVAL_CONDITIONS,train_names={})
    for task,conditions in TRAIN_CONDITIONS.items():
        names=[]
        for condition in conditions:
            matching=[n for n,c in EVAL_CONDITIONS.items() if c['task']==task and c['condition']==condition]
            assert len(matching)==1,(task,condition);names.extend(matching)
        cfg['train_names'][task]=names
    cfg['cycle']=list(cfg['cells'])
    return cfg

def summarize_spatial(rows):
    cells={};task_metrics={}
    for task,name in sorted({(r['task'],r['condition']) for r in rows}):
        selected=[r for r in rows if r['task']==task and r['condition']==name]
        metric=summarize(selected,0);cells[task+'/'+name]=dict(family=task,condition=name,overall=metric)
        if metric['balanced_accuracy'] is not None and metric['macro_ovr_auc'] is not None:
            task_metrics.setdefault(task,[]).append(metric)
    task_scores={task:dict(normalized_ba=float(np.mean([(m['balanced_accuracy']-m['chance_accuracy'])/(1-m['chance_accuracy']) for m in ms])),mean_auc=float(np.mean([m['macro_ovr_auc'] for m in ms]))) for task,ms in task_metrics.items()}
    assert len(task_scores)==5
    return dict(cells=cells,task_scores=task_scores,rank=[min(v['normalized_ba'] for v in task_scores.values()),float(np.mean([v['mean_auc'] for v in task_scores.values()]))],episodes=len(rows),interpretation='Equal tasks after within-task condition means. Empty recognition set has accuracy/specificity only and is excluded from balanced ranking. Fixed argmax, no calibration.')
