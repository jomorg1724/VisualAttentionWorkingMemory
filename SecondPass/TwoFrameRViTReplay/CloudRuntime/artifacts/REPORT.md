# Fresh two-frame CNN recurrent visual transformer training
Updates 2310; presentations 70000; selected 1250; stop planned_complete.
Provisional reporting criteria: selected-model fresh final BA at least 0.70 in every condition for acquisition; 0.90 in every condition for the solved target.

## Terminal

|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|
|---|---:|---:|---:|---:|---:|---:|
|B12|200|0.4711342309261526|0.48235414116687064|0.5701754385964912|0.6206896551724138|0.6428571428571429|

Events: `{"catch": {"confusion": [[10, 18], [0, 0]], "false_positive_rate": 0.6428571428571429, "hit_rate": null, "n": 28, "positive_rate": 0.6428571428571429, "specificity": 0.35714285714285715, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[22, 36], [0, 0]], "false_positive_rate": 0.6206896551724138, "hit_rate": null, "n": 58, "positive_rate": 0.6206896551724138, "specificity": 0.3793103448275862, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [49, 65]], "false_positive_rate": null, "hit_rate": 0.5701754385964912, "n": 114, "positive_rate": 0.5701754385964912, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B20|200|0.5060179518563852|0.5105059159526724|0.5701754385964912|0.5517241379310345|0.5714285714285714|

Events: `{"catch": {"confusion": [[12, 16], [0, 0]], "false_positive_rate": 0.5714285714285714, "hit_rate": null, "n": 28, "positive_rate": 0.5714285714285714, "specificity": 0.42857142857142855, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[26, 32], [0, 0]], "false_positive_rate": 0.5517241379310345, "hit_rate": null, "n": 58, "positive_rate": 0.5517241379310345, "specificity": 0.4482758620689655, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [49, 65]], "false_positive_rate": null, "hit_rate": 0.5701754385964912, "n": 114, "positive_rate": 0.5701754385964912, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B28|200|0.457874337005304|0.48368013055895553|0.5087719298245614|0.6206896551724138|0.5357142857142857|

Events: `{"catch": {"confusion": [[13, 15], [0, 0]], "false_positive_rate": 0.5357142857142857, "hit_rate": null, "n": 28, "positive_rate": 0.5357142857142857, "specificity": 0.4642857142857143, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[22, 36], [0, 0]], "false_positive_rate": 0.6206896551724138, "hit_rate": null, "n": 58, "positive_rate": 0.6206896551724138, "specificity": 0.3793103448275862, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [56, 58]], "false_positive_rate": null, "hit_rate": 0.5087719298245614, "n": 114, "positive_rate": 0.5087719298245614, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`

## Selected

|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|
|---|---:|---:|---:|---:|---:|---:|
|B12|200|0.5|0.4662382700938392|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B20|200|0.5|0.5089759281925744|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`
|B28|200|0.5|0.42645858833129335|1.0|1.0|1.0|

Events: `{"catch": {"confusion": [[0, 28], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 28, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"confusion": [[0, 58], [0, 0]], "false_positive_rate": 1.0, "hit_rate": null, "n": 58, "positive_rate": 1.0, "specificity": 0.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"confusion": [[0, 0], [0, 114]], "false_positive_rate": null, "hit_rate": 1.0, "n": 114, "positive_rate": 1.0, "specificity": null, "target_side_counts": {"0": 57, "1": 57}}}`

Generated unique movies: 7000; unique movies presented: 7000; replay presentations: 70000.
