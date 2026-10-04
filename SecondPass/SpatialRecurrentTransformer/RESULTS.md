# Recurrent spatial transformer: completed run

Finished 2026-09-28 17:21:43 UTC. Full 2,600 additional updates / 83,200 episodes (200 task updates / 6,400 episodes per task). Parent comparison terminal cumulative 9,360; compatible weights and named Adam inherited, transformer/CLS/readout initialized fresh. Native 13 tasks / 35 conditions unchanged; all weights trainable. Validation selected update 1,300; terminal is 2,600. Both fresh final tests complete all 35 conditions.

## Held-out balanced accuracy

|Task|Selected 1300 BA|Terminal 2600 BA|Selected AUC|Terminal AUC|
|---|---:|---:|---:|---:|
|motion_direction|46.88%|25.78%|0.9612|0.7598|
|orientation|79.69%|100.00%|0.9697|1.0000|
|contrast|100.00%|100.00%|1.0000|1.0000|
|spatial_frequency|92.97%|100.00%|0.9785|1.0000|
|chromatic_increment|100.00%|99.22%|1.0000|1.0000|
|contour|50.00%|57.03%|0.5024|0.5706|
|natural_spectrum|100.00%|97.66%|1.0000|0.9954|
|orientation_ring|50.00%|50.78%|0.4934|0.5037|
|orientation_cued|51.76%|48.44%|0.5322|0.4650|
|motion_duration_cued|23.63%|25.78%|0.5129|0.5015|
|krauzlis_cued_motion|50.00%|50.19%|0.5090|0.4849|
|spatial_binding|50.39%|52.34%|0.4860|0.5434|
|image_recognition|51.91%|50.00%|0.5161|0.5500|

Chance is 25% for motion_direction and motion_duration_cued, 50% otherwise. Recognition BA excludes empty sets; all three empty-set conditions have 100% specificity at both checkpoints (128 examples each).

## Every primary condition

|Task|Cell|N/checkpoint|Selected BA or specificity|Terminal BA or specificity|
|---|---|---:|---:|---:|
|motion_direction|mixed|128|46.88%|25.78%|
|orientation|mixed|128|79.69%|100.00%|
|contrast|mixed|128|100.00%|100.00%|
|spatial_frequency|mixed|128|92.97%|100.00%|
|chromatic_increment|mixed|128|100.00%|99.22%|
|contour|mixed|128|50.00%|57.03%|
|natural_spectrum|mixed|128|100.00%|97.66%|
|orientation_ring|D0|128|50.00%|50.78%|
|orientation_cued|D0|128|50.00%|42.97%|
|orientation_cued|D4|128|50.00%|50.78%|
|orientation_cued|D12|128|48.44%|50.78%|
|orientation_cued|D24|128|58.59%|49.22%|
|motion_duration_cued|D0|128|22.66%|27.34%|
|motion_duration_cued|D4|128|24.22%|23.44%|
|motion_duration_cued|D12|128|22.66%|26.56%|
|motion_duration_cued|D24|128|25.00%|25.78%|
|krauzlis_cued_motion|B12|200|50.00%|50.00%|
|krauzlis_cued_motion|B20|200|50.00%|50.58%|
|krauzlis_cued_motion|B28|200|50.00%|50.00%|
|spatial_binding|D0|128|51.56%|53.91%|
|spatial_binding|D4|128|50.00%|57.03%|
|spatial_binding|D12|128|50.00%|49.22%|
|spatial_binding|D24|128|50.00%|49.22%|
|image_recognition|N0_H3|128|100.00%|100.00%|
|image_recognition|N0_H4|128|100.00%|100.00%|
|image_recognition|N0_H5|128|100.00%|100.00%|
|image_recognition|N4_H3|128|51.56%|50.00%|
|image_recognition|N4_H4|128|53.91%|50.00%|
|image_recognition|N4_H5|128|53.91%|50.00%|
|image_recognition|N12_H3|128|52.34%|50.00%|
|image_recognition|N12_H4|128|50.00%|50.00%|
|image_recognition|N12_H5|128|59.38%|50.00%|
|image_recognition|N24_H3|128|47.66%|50.00%|
|image_recognition|N24_H4|128|48.44%|50.00%|
|image_recognition|N24_H5|128|50.00%|50.00%|

## Interpretation and limitations

The difficult spatial/memory tasks were not acquired under this exposure. Several basic sensory tasks are accurate, but motion-direction accuracy is poorer at the terminal checkpoint and contour remains weak. This does not establish that the architecture cannot learn these tasks. Only two validation looks and no matched continuation control were run; architecture replacement and newly initialized decision computations confound a causal comparison against the predecessor. High motion AUC is not high argmax decision accuracy. No additional training or diagnostic was launched.

## Artifacts and billing state

All 40 published artifact sizes/hashes were verified locally, including selected.pt, terminal.pt, report.json and both final test files. Runtime: `/Users/jonathanmorgan/VAWMRuntime/cloud_transformer_01/artifacts`. Full per-condition records, Krauzlis event subgroup statistics and predictions remain in those original artifacts. A bounded CPU-only restart recovered files after GPU capacity was unavailable; no optimizer restarted. Pod `txmfzvhbc9zm3b` was verified EXITED again after retrieval. GPU compute is stopped; retained storage can still bill.
