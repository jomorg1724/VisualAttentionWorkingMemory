# Fresh single-layer sixteen-head sequence KDA training
Updates 4216; episodes 134912; selected 2108; stop planned_complete.
Provisional reporting criteria: selected-model fresh final BA at least 0.70 in every condition for acquisition; 0.90 in every condition for the solved target.

## Terminal

|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|
|---|---:|---:|---:|---:|---:|---:|
|B12|200|0.5|0.4261780905752754|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B20|200|0.5|0.4714657282741738|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B28|200|0.5|0.44484394124847004|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`

## Selected

|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|
|---|---:|---:|---:|---:|---:|---:|
|B12|200|0.5|0.43426152590779277|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B20|200|0.5|0.5240718074255406|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B28|200|0.5|0.47070073439412485|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
