# Motion validation trajectories

Both branches start at the same checkpoint10000. Each optimizer update exposes eight motion episodes, so the horizontal variable below is added motion episodes, not total five-task episodes. Validation draws differ between branches; these are descriptive trajectories, not paired accuracy differences.

| Branch | Step | Added motion episodes | D0 | D4 | D12 | D24 |
|---|---:|---:|---:|---:|---:|---:|
| Cloud five tasks | 10000 | 0 | 28.12% | 23.44% | 25.00% | 25.00% |
| Local motion only | 10000 | 0 | 26.56% | 28.12% | 24.22% | 25.00% |
| Local motion only | 10440 | 3520 | 22.66% | 25.00% | 25.00% | 25.78% |
| Cloud five tasks | 10800 | 6400 | 26.56% | 23.44% | 25.00% | 25.00% |
| Local motion only | 10880 | 7040 | 23.44% | 25.00% | 25.00% | 25.00% |
| Local motion only | 11320 | 10560 | 25.00% | 24.22% | 27.34% | 25.00% |
| Cloud five tasks | 11600 | 12800 | 25.00% | 23.44% | 25.00% | 25.00% |
| Local motion only | 11760 | 14080 | 25.00% | 24.22% | 25.00% | 25.00% |
| Local motion only | 12200 | 17600 | 21.09% | 25.00% | 23.44% | 25.00% |

The local branch uses full motion CE; the cloud averages five task losses and combines their gradients before clipping/Adam. These results cannot isolate gradient interference. Cloud points beyond the local exposure range are outside the matched exposure overlap. Missing later cloud checkpoints may still be training; this report reads only completed scheduled validations. The local final parent/selected/terminal test remains the paired evidence for local acquisition.