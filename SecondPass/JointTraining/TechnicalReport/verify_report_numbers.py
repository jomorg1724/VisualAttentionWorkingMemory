"""Cross-check the report's displayed task table against its frozen audit."""
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
snapshot = json.loads((HERE / 'metrics_snapshot.json').read_text())
text = (HERE / 'architecture_microstimulation.md').read_text()
old = next(e['result'] for e in snapshot['evaluations'] if e['role'] == 'v2_final_terminal')
new = next(e['result'] for e in snapshot['evaluations'] if e['role'] == 'v3_validation' and e['step'] == 2314)
names = {
    'Motion direction': 'motion_direction', 'Signed orientation': 'orientation',
    'Contrast': 'contrast', 'Spatial frequency': 'spatial_frequency',
    'Chromatic increment': 'chromatic_increment', 'Contour grouping': 'contour',
    'Natural spectral detail': 'natural_spectrum', 'Ring-cued orientation': 'orientation_ring',
    'Spatially cued orientation': 'orientation_cued', 'Cued motion duration': 'motion_duration_cued',
    'Krauzlis target/foil change': 'krauzlis_cued_motion', 'Spatial orientation binding': 'spatial_binding',
    'Scene recognition, nonempty': 'image_recognition',
}
checked = []
for line in text.splitlines():
    cells = [v.strip() for v in line.strip('|').split('|')]
    if len(cells) != 4 or cells[0] not in names or '%' not in cells[1]:
        continue
    task = names[cells[0]]
    a, b = old['summary']['tasks'][task], new['summary']['tasks'][task]
    percent = lambda value: f"{(Decimal(str(value)) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}%"
    expected = [percent(a['balanced_accuracy']), percent(b['balanced_accuracy']), f"{b['auc']:.3f}"]
    assert cells[1:] == expected, (task, cells[1:], expected)
    checked.append(task)
assert len(checked) == len(set(checked)) == 13
assert len(re.findall(r'^# \d+\.', text, re.M)) == 10
assert f"{snapshot['architecture']['total_parameters']:,}" in text
assert snapshot['architecture']['total_parameters'] == sum(snapshot['architecture']['parameter_groups'].values())
assert 32 * 2647 == 84704 and 32 * 2860 == 91520
result = {'all_13_task_rows_match_snapshot': True, 'displayed_metric_values_verified': 39,
          'all_10_sections_present': True, 'parameter_total_verified': True,
          'no_model_calls': True, 'snapshot_completed_utc': snapshot['snapshot_completed_utc']}
(HERE / 'report_numeric_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
