# Learned spatial-comparison candidate — completed cloud run

All 2,600 additional updates / 83,200 episodes completed. Parent: ConvGRU terminal6760; selected additional1300 (cumulative8060), terminal additional2600 (cumulative9360). Both final evaluations contain all35 unique primary cells.

## Final held-out task results

|Task|Selected BA (%)|Terminal BA (%)|Selected AUC|Terminal AUC|
|---|---:|---:|---:|---:|
|motion_direction|100.00|100.00|1.0000|1.0000|
|orientation|100.00|100.00|1.0000|1.0000|
|contrast|100.00|100.00|1.0000|1.0000|
|spatial_frequency|100.00|100.00|1.0000|1.0000|
|chromatic_increment|100.00|100.00|1.0000|1.0000|
|contour|88.28|89.06|0.9534|0.9666|
|natural_spectrum|99.22|99.22|1.0000|1.0000|
|orientation_ring|48.44|50.78|0.5181|0.4919|
|orientation_cued|50.00|48.44|0.5186|0.5070|
|motion_duration_cued|24.61|24.22|0.4956|0.4940|
|krauzlis_cued_motion|50.00|50.00|0.5190|0.5033|
|spatial_binding|49.80|52.73|0.5181|0.5137|
|image_recognition|51.39|49.22|0.5195|0.4942|

## Selected — every primary cell

|Task|Cell|n|BA|AUC|Specificity|FPR|
|---|---|---:|---:|---:|---:|---:|
|motion_direction|mixed|128|1.0|1.0|None|None|
|orientation|mixed|128|1.0|1.0|1.0|0.0|
|contrast|mixed|128|1.0|1.0|1.0|0.0|
|spatial_frequency|mixed|128|1.0|1.0|1.0|0.0|
|chromatic_increment|mixed|128|1.0|1.0|1.0|0.0|
|contour|mixed|128|0.8828125|0.953369140625|0.84375|0.15625|
|natural_spectrum|mixed|128|0.9921875|1.0|1.0|0.0|
|orientation_ring|D0|128|0.484375|0.51806640625|0.34375|0.65625|
|orientation_cued|D0|128|0.5234375|0.55810546875|0.4375|0.5625|
|orientation_cued|D4|128|0.5|0.486328125|0.21875|0.78125|
|orientation_cued|D12|128|0.5|0.51611328125|0.171875|0.828125|
|orientation_cued|D24|128|0.4765625|0.513916015625|0.09375|0.90625|
|motion_duration_cued|D0|128|0.2265625|0.46883138020833326|None|None|
|motion_duration_cued|D4|128|0.2421875|0.4955240885416667|None|None|
|motion_duration_cued|D12|128|0.2890625|0.5095214843750001|None|None|
|motion_duration_cued|D24|128|0.2265625|0.5086263020833334|None|None|
|krauzlis_cued_motion|B12|200|0.5|0.4972460220318237|0.0|1.0|
|krauzlis_cued_motion|B20|200|0.5|0.5100979192166463|0.0|1.0|
|krauzlis_cued_motion|B28|200|0.5|0.5495716034271726|0.0|1.0|
|spatial_binding|D0|128|0.4921875|0.47509765625|0.25|0.75|
|spatial_binding|D4|128|0.5|0.4765625|0.015625|0.984375|
|spatial_binding|D12|128|0.5|0.611083984375|0.0|1.0|
|spatial_binding|D24|128|0.5|0.509765625|0.0|1.0|
|image_recognition|N0_H3|128|None|None|1.0|0.0|
|image_recognition|N0_H4|128|None|None|1.0|0.0|
|image_recognition|N0_H5|128|None|None|1.0|0.0|
|image_recognition|N4_H3|128|0.515625|0.508056640625|0.234375|0.765625|
|image_recognition|N4_H4|128|0.5078125|0.557373046875|0.28125|0.71875|
|image_recognition|N4_H5|128|0.453125|0.484619140625|0.140625|0.859375|
|image_recognition|N12_H3|128|0.484375|0.456298828125|0.234375|0.765625|
|image_recognition|N12_H4|128|0.5078125|0.4970703125|0.25|0.75|
|image_recognition|N12_H5|128|0.5546875|0.567138671875|0.296875|0.703125|
|image_recognition|N24_H3|128|0.546875|0.550537109375|0.546875|0.453125|
|image_recognition|N24_H4|128|0.5|0.490478515625|0.46875|0.53125|
|image_recognition|N24_H5|128|0.5546875|0.56396484375|0.5|0.5|

Krauzlis B12 event strata: `{"catch": {"n": 28, "positive_rate": 1.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"n": 58, "positive_rate": 1.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"n": 114, "positive_rate": 1.0, "target_side_counts": {"0": 57, "1": 57}}}`

Krauzlis B20 event strata: `{"catch": {"n": 28, "positive_rate": 1.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"n": 58, "positive_rate": 1.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"n": 114, "positive_rate": 1.0, "target_side_counts": {"0": 57, "1": 57}}}`

Krauzlis B28 event strata: `{"catch": {"n": 28, "positive_rate": 1.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"n": 58, "positive_rate": 1.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"n": 114, "positive_rate": 1.0, "target_side_counts": {"0": 57, "1": 57}}}`

## Terminal — every primary cell

|Task|Cell|n|BA|AUC|Specificity|FPR|
|---|---|---:|---:|---:|---:|---:|
|motion_direction|mixed|128|1.0|1.0|None|None|
|orientation|mixed|128|1.0|1.0|1.0|0.0|
|contrast|mixed|128|1.0|1.0|1.0|0.0|
|spatial_frequency|mixed|128|1.0|1.0|1.0|0.0|
|chromatic_increment|mixed|128|1.0|1.0|1.0|0.0|
|contour|mixed|128|0.890625|0.966552734375|0.890625|0.109375|
|natural_spectrum|mixed|128|0.9921875|1.0|1.0|0.0|
|orientation_ring|D0|128|0.5078125|0.491943359375|0.265625|0.734375|
|orientation_cued|D0|128|0.453125|0.4873046875|0.296875|0.703125|
|orientation_cued|D4|128|0.5|0.49853515625|0.46875|0.53125|
|orientation_cued|D12|128|0.484375|0.55517578125|0.46875|0.53125|
|orientation_cued|D24|128|0.5|0.487060546875|0.46875|0.53125|
|motion_duration_cued|D0|128|0.2265625|0.4788411458333333|None|None|
|motion_duration_cued|D4|128|0.2578125|0.49153645833333337|None|None|
|motion_duration_cued|D12|128|0.2421875|0.5043945312499999|None|None|
|motion_duration_cued|D24|128|0.2421875|0.5013020833333334|None|None|
|krauzlis_cued_motion|B12|200|0.5|0.49775601795185637|0.0|1.0|
|krauzlis_cued_motion|B20|200|0.5|0.49530803753569974|0.0|1.0|
|krauzlis_cued_motion|B28|200|0.5|0.5167278661770706|0.0|1.0|
|spatial_binding|D0|128|0.53125|0.485595703125|0.765625|0.234375|
|spatial_binding|D4|128|0.5078125|0.50732421875|0.84375|0.15625|
|spatial_binding|D12|128|0.5703125|0.55712890625|0.640625|0.359375|
|spatial_binding|D24|128|0.5|0.5048828125|0.421875|0.578125|
|image_recognition|N0_H3|128|None|None|1.0|0.0|
|image_recognition|N0_H4|128|None|None|1.0|0.0|
|image_recognition|N0_H5|128|None|None|1.0|0.0|
|image_recognition|N4_H3|128|0.484375|0.51953125|0.5|0.5|
|image_recognition|N4_H4|128|0.5390625|0.56494140625|0.453125|0.546875|
|image_recognition|N4_H5|128|0.5234375|0.456787109375|0.328125|0.671875|
|image_recognition|N12_H3|128|0.484375|0.516357421875|0.15625|0.84375|
|image_recognition|N12_H4|128|0.53125|0.528076171875|0.203125|0.796875|
|image_recognition|N12_H5|128|0.4609375|0.492919921875|0.09375|0.90625|
|image_recognition|N24_H3|128|0.484375|0.45751953125|0.453125|0.546875|
|image_recognition|N24_H4|128|0.4765625|0.438720703125|0.5|0.5|
|image_recognition|N24_H5|128|0.4453125|0.472900390625|0.5625|0.4375|

Krauzlis B12 event strata: `{"catch": {"n": 28, "positive_rate": 1.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"n": 58, "positive_rate": 1.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"n": 114, "positive_rate": 1.0, "target_side_counts": {"0": 57, "1": 57}}}`

Krauzlis B20 event strata: `{"catch": {"n": 28, "positive_rate": 1.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"n": 58, "positive_rate": 1.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"n": 114, "positive_rate": 1.0, "target_side_counts": {"0": 57, "1": 57}}}`

Krauzlis B28 event strata: `{"catch": {"n": 28, "positive_rate": 1.0, "target_side_counts": {"0": 14, "1": 14}}, "foil": {"n": 58, "positive_rate": 1.0, "target_side_counts": {"0": 29, "1": 29}}, "target": {"n": 114, "positive_rate": 1.0, "target_side_counts": {"0": 57, "1": 57}}}`

## Interpretation and provenance

Sensory discrimination remains strong; the difficult spatial and memory tasks remain near chance. This acquisition run does not establish architectural incapacity, information erasure or causal superiority/inferiority to equally exposed ordinary continuation: that control was cancelled. No new task teaching or privileged supervision was introduced. Empty recognition sets are always negative and have separate specificity/FPR; do not interpret these as balanced binary recognition.

CUDA A40 run used the same native13task/35condition stimuli, all learned parameters trainable, fp32/full BPTT and inherited compatible Adam1e-4. A Torch2.8 metadata-only adapter made implicit decoupled_weight_decay=False explicit. Failed pre-update receipts are preserved.

Artifacts: `/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/artifacts`. All53 published manifest files were retrieved and hash/size verified; see `../retrieval_verified.json`. Terminal SHA256: `e47031f3c66fefdb174384ad442688554966f3ce2cc86a9562e2ec99640ae72c`. GPU stopped after recovery; no new training started during retrieval.
