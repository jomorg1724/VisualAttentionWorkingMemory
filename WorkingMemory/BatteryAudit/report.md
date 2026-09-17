# Ideal-observer audit of the five-task battery

Generated 2026-09-16T21:13:35-0700 from `WorkingMemory/SpatialTaskBattery/stimuli.py` (SHA256 `16b7efd4575eba27...`), validation stream seed 63973001. CPU only, no training. Full numbers in `results.json`.

## Orientation (cued sign, D0)

| n | BA | AUC | BA 15deg | BA 30deg | BA 45deg | BA top row | BA bottom row | glyph decode | angle err median / p95 (deg) | no-cue multiset TV raw / sign-normalised |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 768 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.059 / 0.168 | 0.065 |

## Motion duration (cued patch, D0)

| n | BA hard vote | BA soft sum | per-frame direction agreement | ring decode | uncued patch vs own winner | uncued patch vs cued label |
|---:|---:|---:|---:|---:|---:|---:|
| 768 | 0.954 | 0.879 | 0.982 | 1.000 | 0.944 | 0.258 |

| step (px) | BA soft | BA hard |
|---:|---:|---:|
| 0.8 | 0.960 | 1.000 |
| 1.2 | 0.949 | 1.000 |
| 1.6 | 0.734 | 0.869 |

| count margin (winner minus runner-up frames) | trials | BA soft |
|---:|---:|---:|
| 1 | 472 | 0.816 |
| 2 | 224 | 0.972 |
| 3 | 50 | 1.000 |
| 4 | 17 | 1.000 |
| 5 | 2 | 1.000 |
| 6 | 3 | 1.000 |

## Krauzlis cued motion change

| baseline | n | events t/f/c | AUC abs(dtheta) | BA at 13deg | BA best thr (deg) | est abs(dtheta) target | est abs(dtheta) unchanged (sd) | dprime | signed err mean (sd) | pre-angle err med / p95 | ring decode |
|---:|---:|---|---:|---:|---|---:|---|---:|---|---|---:|
| B12 | 256 | 144/74/38 | 0.999 | 0.973 | 0.986 (17.084) | 27.657 | 5.082 (4.038) | 4.558 | -0.508 (5.667) | 2.568 / 8.448 | 1.000 |
| B20 | 256 | 145/77/34 | 0.999 | 0.979 | 0.988 (13.945) | 27.360 | 5.287 (3.876) | 4.371 | 0.602 (6.106) | 2.282 / 6.687 | 1.000 |
| B28 | 256 | 146/73/37 | 0.999 | 0.987 | 0.997 (14.771) | 27.398 | 4.852 (3.428) | 4.722 | -0.007 (5.678) | 1.907 / 6.201 | 1.000 |

## Spatial binding (retrocue)

| delay | n | BA | ring decode | swapped pair recovered |
|---:|---:|---:|---:|---:|
| D0 | 256 | 1.000 | 1.000 | 1.000 |
| D24 | 128 | 1.000 | 1.000 | 1.000 |

No-cue leak: P(label=1 | swapped pair) over all binding trials, expected 0.5 everywhere.

| pair | n | P(label=1) |
|---|---:|---:|
| (0, 1) | 67 | 0.552 |
| (0, 2) | 61 | 0.443 |
| (0, 3) | 61 | 0.639 |
| (1, 2) | 56 | 0.429 |
| (1, 3) | 76 | 0.447 |
| (2, 3) | 63 | 0.492 |

## Image recognition

| condition | n | accuracy | BA | labels | probe hold identical | study unique | blanks gray | metadata hashes match |
|---|---:|---:|---|---|---:|---:|---:|---:|
| N0_H3 | 128 | 1.000 | n/a (all negative) | {'0': 128} | 1.000 | 1.000 | 1.000 | 1.000 |
| N4_H3 | 128 | 1.000 | 1.000 | {'0': 64, '1': 64} | 1.000 | 1.000 | 1.000 | 1.000 |
| N24_H5 | 128 | 1.000 | 1.000 | {'1': 64, '0': 64} | 1.000 | 1.000 | 1.000 | 1.000 |

Manifest: {'test': 200, 'train': 200, 'val': 100}; sha256 disjoint across splits: True; base_id disjoint: True.

## Frame roles per task (extreme conditions)

**orientation_cued {'delay': 0}**: tensor frames 4, frame_count 4, indices in range True, last referenced 3.  
Fields: cue_frames=[0, 1, 2], sample_frames=[1, 2], probe_frame=3, blank_frames=[].  
Non-gray pixels per frame: [144, 28644, 28638, 28929]

**orientation_cued {'delay': 24}**: tensor frames 28, frame_count 28, indices in range True, last referenced 27.  
Fields: cue_frames=[0, 1, 2], sample_frames=[1, 2], probe_frame=27, blank_frames=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26].  
Non-gray pixels per frame: [168, 28662, 28665, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 28929]

**spatial_binding {'delay': 0}**: tensor frames 5, frame_count 5, indices in range True, last referenced 4.  
Fields: cue_frames=[3], sample_frames=[1, 2], probe_frame=4, blank_frames=[].  
Non-gray pixels per frame: [108, 28908, 28899, 504, 28929]

**spatial_binding {'delay': 24}**: tensor frames 29, frame_count 29, indices in range True, last referenced 28.  
Fields: cue_frames=[27], sample_frames=[1, 2], probe_frame=28, blank_frames=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26].  
Non-gray pixels per frame: [108, 28905, 28908, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 504, 28929]

**motion_duration_cued {'delay': 0}**: tensor frames 11, frame_count 11, indices in range True, last referenced 10.  
Fields: cue_frames=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], reference_frame=1, moving_frames=[2, 3, 4, 5, 6, 7, 8, 9], blank_frames=[], report_frame=10.  
Non-gray pixels per frame: [480, 852, 849, 846, 852, 846, 846, 843, 846, 852, 120]

**motion_duration_cued {'delay': 24}**: tensor frames 35, frame_count 35, indices in range True, last referenced 34.  
Fields: cue_frames=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], reference_frame=1, moving_frames=[2, 3, 4, 5, 6, 7, 8, 9], blank_frames=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], report_frame=34.  
Non-gray pixels per frame: [480, 852, 849, 849, 858, 846, 852, 840, 852, 855, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 120]

**image_recognition {'load': 0, 'probe_hold': 3}**: tensor frames 7, frame_count 7, indices in range True, last referenced 6.  
Fields: study_frames=[], blank_frames=[1, 2, 3], probe_frames=[4, 5, 6].  
Non-gray pixels per frame: [108, 0, 0, 0, 30000, 30000, 30000]

**image_recognition {'load': 24, 'probe_hold': 5}**: tensor frames 33, frame_count 33, indices in range True, last referenced 32.  
Fields: study_frames=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], blank_frames=[25, 26, 27], probe_frames=[28, 29, 30, 31, 32].  
Non-gray pixels per frame: [108, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000, 0, 0, 0, 30000, 30000, 30000, 30000, 30000]

**krauzlis_cued_motion {'baseline_transitions': 12}**: tensor frames 29, frame_count 29, indices in range True, last referenced 28.  
Fields: cue_frames=[0, 1], fixation_only_frames=[2, 3, 4, 5, 6], reference_frame=7, first_postchange_frame=20, virtual_event_frame=20, report_frame=28.  
Non-gray pixels per frame: [252, 252, 36, 36, 36, 36, 36, 348, 357, 342, 348, 351, 342, 360, 375, 393, 360, 381, 381, 372, 378, 390, 375, 366, 396, 396, 375, 390, 36]

**krauzlis_cued_motion {'baseline_transitions': 28}**: tensor frames 45, frame_count 45, indices in range True, last referenced 44.  
Fields: cue_frames=[0, 1], fixation_only_frames=[2, 3, 4, 5, 6], reference_frame=7, first_postchange_frame=36, virtual_event_frame=36, report_frame=44.  
Non-gray pixels per frame: [252, 252, 36, 36, 36, 36, 36, 363, 354, 378, 375, 390, 369, 366, 372, 372, 384, 390, 390, 390, 369, 372, 366, 366, 363, 375, 351, 363, 375, 372, 387, 369, 384, 375, 363, 393, 363, 366, 363, 345, 357, 354, 354, 348, 36]
