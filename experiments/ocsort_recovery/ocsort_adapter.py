"""I/O adapter for the official, externally installed OC-SORT tracker."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class OCSortAdapter:
    """Keep official OC-SORT logic and frozen defaults unchanged."""

    def __init__(self, args, frame_rate=30):
        root = os.environ.get("OCSORT_ROOT")
        if not root:
            raise RuntimeError("Set OCSORT_ROOT to the pinned OC_SORT checkout")
        root = str(Path(root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from trackers.ocsort_tracker.ocsort import OCSort
        self.tracker = OCSort(det_thresh=0.6, max_age=30, min_hits=3,
                              iou_threshold=0.3, delta_t=3, asso_func="iou",
                              inertia=0.2, use_byte=False)
        self.last_g3 = {"b_t": 0, "alarm": False, "parked": 0}
        self.cg_stats = {}

    def update(self, output_results, img_info, img_size, current_frame=None):
        if output_results is None:
            detections = np.empty((0, 5), dtype=np.float32)
        else:
            detections = output_results.detach().cpu().numpy()[:, :5]
        predictions = self.tracker.update(detections, img_info, img_size)
        scores = {int(track.id + 1): float(track.last_observation[4])
                  for track in self.tracker.trackers if track.last_observation is not None}
        output = []
        for x1, y1, x2, y2, track_id in predictions:
            width, height = x2 - x1, y2 - y1
            if width * height <= 100 or width / max(height, 1e-12) > 1.6:
                continue
            track_id = int(track_id)
            output.append(SimpleNamespace(
                track_id=track_id, tlwh=np.asarray([x1, y1, width, height], dtype=float),
                score=scores.get(track_id, 1.0)))
        return output
