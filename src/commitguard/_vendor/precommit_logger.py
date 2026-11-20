# -*- coding: utf-8 -*-
"""E3-2 Pre-commit observability logger (read-only instrumentation).

Env-gated: set PRECOMMIT_LOG_DIR to enable; otherwise every call is a no-op.
One JSONL file per run-condition x segment: {cond}__{seg}.jsonl
All logged values are read-only copies; logging never mutates tracker state.
"""
import json
import os

import numpy as np


def _round(x, nd=6):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


def _tlwh_to_xyah(tlwh):
    return [tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0,
            tlwh[2] / max(tlwh[3], 1e-9), tlwh[3]]


class PreCommitLogger(object):
    _inst = None

    def __init__(self):
        self.dir = os.environ.get('PRECOMMIT_LOG_DIR', '')
        self.enabled = bool(self.dir)
        self.run = None
        self.seg = None
        self._f = None
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = PreCommitLogger()
        return cls._inst

    def set_run(self, cond):
        if self.enabled:
            self.run = cond

    def set_segment(self, seg_name):
        if not self.enabled:
            return
        self.seg = seg_name
        if self._f is not None:
            self._f.close()
        self._f = open(os.path.join(self.dir, '%s__%s.jsonl' % (self.run, seg_name)),
                       'w', buffering=1)

    def log(self, etype, frame, **kw):
        if not self.enabled or self._f is None:
            return
        rec = {'t': etype, 'seg': self.seg, 'f': int(frame)}
        rec.update({k: _round(v) for k, v in kw.items()})
        self._f.write(json.dumps(rec) + '\n')

    def log_stage(self, name, frame, tracks, dets, iou_mat, fused_mat,
                  matches, u_track, u_det):
        """One association stage: candidate sets, raw IoU-dist matrix, fused cost,
        committed matches with cost/margin/KF-innovation, unmatched sets."""
        if not self.enabled or self._f is None:
            return
        match_rows = []
        for i, j in matches:
            cost = float(fused_mat[i, j])
            row = fused_mat[i]
            second = float(np.partition(row, 1)[1]) if row.size > 1 else None
            innov = None
            if tracks[i].mean is not None:
                z = _tlwh_to_xyah(list(dets[j].tlwh))
                innov = [float(z[k] - tracks[i].mean[k]) for k in range(4)]
            match_rows.append({'track_id': int(tracks[i].track_id),
                               'det_idx': int(j),
                               'det_score': float(dets[j].score),
                               'cost': cost,
                               'margin2nd': second,
                               'iou_dist': float(iou_mat[i, j]),
                               'innov_xyah': innov})
        self.log(name, frame,
                 track_ids=[int(t.track_id) for t in tracks],
                 track_states=[int(t.state) for t in tracks],
                 track_means_xyah=[_round(t.mean[:4]) if t.mean is not None else None
                                   for t in tracks],
                 det_scores=[float(d.score) for d in dets],
                 det_tlwhs=[_round(list(d.tlwh)) for d in dets],
                 iou_dist=iou_mat, fused=fused_mat,
                 matches=match_rows,
                 u_track=[int(k) for k in u_track],
                 u_det=[int(k) for k in u_det])

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
