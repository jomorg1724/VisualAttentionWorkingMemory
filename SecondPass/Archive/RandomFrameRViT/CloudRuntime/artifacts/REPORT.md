# Fresh two-frame CNN recurrent visual transformer training
Updates 3960; presentations 120000; selected 3500; stop planned_complete.
Provisional reporting criteria: selected-model fresh final BA at least 0.70 in every condition for acquisition; 0.90 in every condition for the solved target.

## Terminal

|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|
|---|---:|---:|---:|---:|---:|---:|
|B12|200|0.5|0.48878008975928194|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B20|200|0.5|0.4541003671970624|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B28|200|0.5|0.49439004487964094|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`

## Selected

|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|
|---|---:|---:|---:|---:|---:|---:|
|B12|200|0.5|0.4769736842105263|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B20|200|0.5|0.5024479804161567|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B28|200|0.5|0.4854141166870665|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`

Generated unique movies: 12000; unique movies presented: 12000; replay presentations: 120000.
