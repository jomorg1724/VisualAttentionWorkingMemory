"""Saved-score WM summaries; task/condition cells remain distinct."""
import numpy as np
from PreAttentiveVision.evaluate_multitask import summarize


def evaluate_rows(rows,resamples=0):
    cells={}
    for family,condition in sorted({(r['task'],r['condition']) for r in rows}):
        subset=[r for r in rows if r['task']==family and r['condition']==condition]
        key=family+'/'+condition
        cells[key]=dict(family=family,condition=condition,protocol=subset[0]['protocol'],
            overall=summarize(subset,resamples),
            strata={field:{str(value):summarize([r for r in subset if str(r['metadata'].get(field))==str(value)],0)
                           for value in sorted({str(r['metadata'].get(field)) for r in subset if r['metadata'].get(field) is not None})}
                    for field in ('target_serial_slot','target_age_at_report','mismatch','binding_lure','difficulty','threshold_level','count_margin')})
    # Equal weight protocol x family groups; then equal condition weight within.
    grouped={}
    for cell in cells.values():
        grouped.setdefault((cell['family'],cell['protocol']),[]).append(cell['overall'])
    values=[np.mean([x['macro_ovr_auc'] for x in group]) for group in grouped.values()]
    norm=[np.mean([(x['balanced_accuracy']-x['chance_accuracy'])/(1-x['chance_accuracy']) for x in group]) for group in grouped.values()]
    return dict(cells=cells,selection_mean_auc=float(np.mean(values)),
        mean_normalized_ba=float(np.mean(norm)),worst_cell_ba=min(v['overall']['balanced_accuracy'] for v in cells.values()),
        episodes=len(rows),interpretation='Equal family/protocol groups for validation selection; conditions averaged within group. Binary and four-way raw accuracies are not pooled. Source-photo clusters for natural images; paired histories are not independent repeats.')
