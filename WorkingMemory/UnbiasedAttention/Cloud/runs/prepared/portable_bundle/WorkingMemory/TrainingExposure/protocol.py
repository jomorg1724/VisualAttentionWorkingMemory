"""Fixed exposure comparison and validation-only capability-preservation screen."""
import numpy as np
from WorkingMemory.PreUpdateAttention.remote_sweep import recipe as parent_recipe
VAL_SEED=53973001
TEST_SEED=54973001
TARGETS=[9200,10000,10800,11600,12400]
def recipe(arm):
    if arm not in ('control_10','focused_50'):raise ValueError(arm)
    cfg=parent_recipe();cfg['cycle']=[name for name in cfg['cells'] for _ in range((4 if name.startswith('motion') else 9) if arm=='control_10' else (20 if name.startswith('motion') else 5))]
    assert len(cfg['cycle'])==80
    return cfg
def cell_metrics(evaluation):
    return {c['condition']:dict(ba=c['overall']['balanced_accuracy'],auc=c['overall']['macro_ovr_auc']) for c in evaluation['cells'].values() if not c['condition'].endswith('_locations')}
def assess(evaluation,baseline):
    values=cell_metrics(evaluation);base=cell_metrics(baseline);assert set(values)==set(base) and len(values)==10
    changes={name:values[name]['ba']-base[name]['ba'] for name in values}
    eligible=all(v>=-.02-1e-12 for v in changes.values())
    key=(min(values[n]['ba'] for n in ('motion_D0','motion_D24')),float(np.mean([values[n]['auc'] for n in ('motion_D0','motion_D24')])))
    return dict(eligible=eligible,ba_changes_vs_parent=changes,rank=list(key),interpretation='All ten cells BA screen at -2pp; then minimum motion BA and mean motion AUC; ties earlier. Screen is not equivalence proof.')
