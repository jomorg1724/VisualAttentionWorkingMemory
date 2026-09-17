"""Focused cells retain their own classes, counts and uncertainty."""
import numpy as np
from WorkingMemory.evaluate import evaluate_rows as summarize_cells


def evaluate_rows(rows,resamples=0):
    result=summarize_cells(rows,resamples)
    result['family_protocol_group_mean_auc']=result['selection_mean_auc']
    result['selection_mean_auc']=float(np.mean([c['overall']['macro_ovr_auc'] for c in result['cells'].values()]))
    result['interpretation']='Equal six-cell mean OVR-AUC selects validation checkpoints; argmax BA and confusion matrices remain per cell. A reset removes the new recurrent history while leaving the opponent traces intact.'
    return result
