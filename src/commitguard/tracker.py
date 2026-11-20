"""CommitGuard's auxiliary-review extension of the frozen ByteTrack tracker."""

import numpy as np

from ._vendor.byte_tracker import BYTETracker, _tlbr_iou_matrix
from ._vendor.precommit_logger import PreCommitLogger


class CommitGuardTracker(BYTETracker):
    """Final E19 auxiliary gates; no ReID, ground truth, or future frames."""

    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.aux_filter_track_iou = float(getattr(
            args, "aux_filter_track_iou", self.aux_track_iou))
        self.aux_fn_track_iou = float(getattr(
            args, "aux_fn_track_iou", self.aux_track_iou))
        self.aux_fn_lost_score_floor = float(getattr(
            args, "aux_fn_lost_score_floor", 0.0))
        self.aux_filter_score_delta = float(getattr(
            args, "aux_filter_score_delta", -1.0))
        self.aux_filter_score_min = float(getattr(
            args, "aux_filter_score_min", 0.0))
        self.aux_filter_alarm_only = bool(getattr(
            args, "aux_filter_alarm_only", False))
        self.aux_filter_pretrack_only = bool(getattr(
            args, "aux_filter_pretrack_only", False))
        self.cg_stats["aux_fn_lost_score_floored"] = 0

    def _apply_aux_review(self, output_results, bboxes, scores, scale):
        """Use independent existing-track gates for filtering and FN addition."""
        self.cg_stats["aux_review_frames"] += 1
        aux = self._aux_candidates()
        reviewed_aux = aux[aux[:, 4] >= self.aux_conf]
        reviewed_boxes = reviewed_aux[:, :4].copy()
        if len(reviewed_boxes):
            reviewed_boxes[:, 2:] += reviewed_boxes[:, :2]
        original_n = len(scores)
        self.cg_stats["aux_original_detections"] += original_n
        track_boxes = [t.tlbr for t in list(self.tracked_stracks) + list(self.lost_stracks)]
        lost_boxes = [t.tlbr for t in self.lost_stracks]
        filter_now = self.aux_filter_mode and (
            not self.aux_filter_alarm_only or self.frame_id <= self._cg_last_window_frame)
        if filter_now:
            if original_n and len(reviewed_aux):
                support = _tlbr_iou_matrix(bboxes, reviewed_boxes).max(axis=1)
            else:
                support = np.zeros(original_n, dtype=np.float64)
            filter_tracks = list(self.tracked_stracks) + list(self.lost_stracks)
            if self.aux_filter_pretrack_only:
                filter_tracks = [t for t in filter_tracks if t.start_frame <= 5]
            filter_track_boxes = [t.tlbr for t in filter_tracks]
            if original_n and filter_track_boxes:
                track_support = _tlbr_iou_matrix(
                    bboxes, np.asarray(filter_track_boxes)).max(axis=1)
            else:
                track_support = np.zeros(original_n, dtype=np.float64)
            track_keep = track_support >= self.aux_filter_track_iou
            if self.aux_filter_score_delta >= 0.0 and original_n:
                track_keep = np.zeros(original_n, dtype=bool)
                active = list(self.tracked_stracks)
                if active:
                    active_boxes = np.asarray([t.tlbr for t in active])
                    active_iou = _tlbr_iou_matrix(bboxes, active_boxes)
                    best = active_iou.argmax(axis=1)
                    for i, j in enumerate(best):
                        track_keep[i] = (
                            active_iou[i, j] >= self.aux_filter_track_iou and
                            scores[i] >= self.aux_filter_score_min and
                            abs(float(scores[i]) - float(active[j].score)) <=
                            self.aux_filter_score_delta)
            keep = np.logical_or(support >= self.aux_iou, track_keep)
            removed = int((~keep).sum())
            self.cg_stats["aux_filter_retained"] += int(keep.sum())
            self.cg_stats["aux_filter_removed"] += removed
            self.cg_stats["aux_filter_track_protected"] += int(np.logical_and(
                support < self.aux_iou, track_keep).sum())
            output_results = output_results[keep]
            bboxes = bboxes[keep]
            scores = scores[keep]
        else:
            removed = 0
        added = []
        if self.aux_fn_mode:
            for det in aux[aux[:, 4] >= self.aux_add_conf]:
                x, y, w, h = det[:4] / scale
                box = np.asarray([x, y, x + w, y + h])
                track_support = (_tlbr_iou_matrix(np.asarray([box]), np.asarray(track_boxes)).max()
                                 if track_boxes else 0.0)
                if track_support < self.aux_fn_track_iou:
                    continue
                self.cg_stats["aux_fn_candidates"] += 1
                if len(bboxes) and _tlbr_iou_matrix(np.asarray([box]), bboxes).max() >= 0.5:
                    self.cg_stats["aux_fn_duplicate_suppressed"] += 1
                    continue
                score = float(det[4])
                if self.aux_fn_lost_score_floor > 0.0 and lost_boxes:
                    lost_support = _tlbr_iou_matrix(
                        np.asarray([box]), np.asarray(lost_boxes)).max()
                    if lost_support >= self.aux_fn_track_iou:
                        score = max(score, self.aux_fn_lost_score_floor)
                        self.cg_stats["aux_fn_lost_score_floored"] += 1
                added.append([box[0], box[1], box[2], box[3], score])
        if added:
            extra = np.asarray(added, dtype=np.float64)
            extra_output = np.column_stack([
                extra[:, :4], extra[:, 4], np.ones(len(extra)), np.zeros(len(extra))])
            output_results = np.concatenate([output_results, extra_output], axis=0)
            bboxes = np.concatenate([bboxes, extra[:, :4]], axis=0)
            scores = np.concatenate([scores, extra[:, 4]], axis=0)
        self.cg_stats["aux_fn_added"] += len(added)
        PreCommitLogger.get().log(
            "aux_review", self.frame_id, raw_n=int(original_n),
            aux_n=int(len(reviewed_aux)), removed=removed, added=int(len(added)),
            conf=float(self.aux_conf), iou=float(self.aux_iou),
            add_conf=float(self.aux_add_conf),
            filter_track_iou=float(self.aux_filter_track_iou),
            fn_track_iou=float(self.aux_fn_track_iou))
        return output_results, bboxes, scores
