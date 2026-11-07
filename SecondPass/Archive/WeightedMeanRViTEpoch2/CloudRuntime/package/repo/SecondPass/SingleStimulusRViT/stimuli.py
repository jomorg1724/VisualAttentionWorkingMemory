"""Project native Krauzlis movies onto one existing stimulus without a cue.

Keep original target pixels, timeline and labels. Remove cue rings and the other
patch. Thus native foil-only trials become ordinary no-change trials. Renderer
metadata is analysis-only and never enters the model.
"""
from __future__ import annotations

import copy

from SecondPass.TwoFrameRViT.worker import FreshStream

ADAPTER_VERSION = 'single_stimulus_no_cue_native_target_projection_v1'


def single_stimulus_movies(images, labels, metadata):
    projected = images.clone()
    records = copy.deepcopy(metadata)
    for index, record in enumerate(records):
        side = int(record['target_location'])
        centers = record['positions_xy']
        cx, cy = map(int, centers[1-side])
        # A generous box contains all bilinear dot support in the discarded
        # 8.125px-radius aperture, without touching fixation or the kept patch.
        projected[index, :, :, cy-12:cy+13, cx-12:cx+13] = .5
        # Preserve the two original cue-time frames as fixation-only frames.
        projected[index, :2] = images[index, 2].unsqueeze(0)
        changed = bool(labels[index])
        record.update(
            task_variant=ADAPTER_VERSION,
            source_event_type=record['event_type'],
            source_changed_patch=record['changed_patch'],
            source_positions_xy=centers,
            source_signed_change_degrees=record['signed_change_degrees'],
            scheduled_event_magnitude_degrees=record['event_magnitude_degrees'],
            positions_xy=[centers[side]], stimulus_count=1,
            stimulus_location_xy=centers[side], cue_frames=[], source_cue_frames=[0,1],
            fixation_only_frames=list(range(7)),
            event_type='target' if changed else 'catch',
            event_name='change' if changed else 'no_change',
            changed_patch=side if changed else None,
            signed_change_degrees=record['signed_change_degrees'] if changed else 0,
            event_magnitude_degrees=record['event_magnitude_degrees'] if changed else 0,
            baseline_means_degrees=[record['baseline_means_degrees'][side]],
            postevent_means_degrees=[record['postevent_means_degrees'][side]],
            event_proportions_exact_per_100_trials={'change':.57,'no_change':.43},
            label_semantics='1=direction change in the sole visible patch; 0=no direction change',
            adaptation='Native target-patch pixels retained exactly; cue rings and other patch removed; all frame counts/times unchanged',
            suite_trial_id=ADAPTER_VERSION+'/'+record['suite_trial_id'],
        )
    return projected, labels, records


class SingleStimulusStream(FreshStream):
    def batch(self, n, task, cell):
        return single_stimulus_movies(*super().batch(n, task, cell))

    def state_dict(self):
        return dict(super().state_dict(), stimulus_adapter=ADAPTER_VERSION)

    def load_state_dict(self, state):
        if state.get('stimulus_adapter') != ADAPTER_VERSION:
            raise ValueError('single-stimulus stream policy mismatch')
        super().load_state_dict(state)
