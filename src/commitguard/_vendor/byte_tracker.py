import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
from .precommit_logger import PreCommitLogger
import time
import random
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, args, image=None, scale=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.args = args
        self.scale = scale
        self.tlwh_mem = [self._tlwh.copy()]
        self.images_mem = []
        if image is not None:
            self.images_mem.append(image)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if not self.args.disable_strict_birth:
            if frame_id == 1:
                self.is_activated = True
        else:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, update_mems=True):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.scale = new_track.scale
        if update_mems:
            self.tlwh_mem.extend(new_track.tlwh_mem)
            self.images_mem.extend(new_track.images_mem)

    def update(self, new_track, frame_id, update_mems=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.scale = new_track.scale
        if update_mems:
            self.tlwh_mem.extend(new_track.tlwh_mem)
            self.images_mem.extend(new_track.images_mem)

    def apply_camera_motion(self, warp_matrix):
        """Apply camera motion to the tracklet center in detector coordinates."""
        if self.mean is None:
            pos = self._tlwh[:2].copy()
        else:
            pos = self.mean[:2].copy()
        pos = pos * self.scale
        new_pos = BYTETracker.warp_pos(pos=pos, warp_matrix=warp_matrix)
        new_pos = new_pos / self.scale
        if self.mean is None:
            self._tlwh[:2] = new_pos
        else:
            self.mean[:2] = new_pos

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    _L1_CONT = {}  # E4-4: cont 模式跨段状态存储（进程内）

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # E4-4 L1 状态机：cold 随 tracker 实例重置；cont 跨段存续（按主视频名）
        _l1_fresh = lambda: {'hist': [], 'raws': [], 'state': 'WARMUP', 'alarm_run': 0, 'drift': 0}
        if getattr(args, 'l1_mode', 'off') == 'cont':
            _key = getattr(args, 'current_seg', '').rsplit('-', 1)[0]
            self._l1 = BYTETracker._L1_CONT.setdefault(_key, _l1_fresh())
        else:
            self._l1 = _l1_fresh()
        # E5-4 S1 Survival 状态机：随 tracker 实例重置（每 30 帧段独立）
        self._s1 = {'state': 'WARMUP', 'recovery_left': 0, 'intervene': False}
        self._s1_thr = float(getattr(args, 's1_thr', 0.0) or 0.0)
        # E9 audit-gated 关联收紧状态机：随 tracker 实例重置（每段独立）。
        # tighten 供下一帧 stage1 匹配使用（1 帧因果滞后：b_t 匹配后才可得，与 S1 同构）。
        self._gated = {'state': 'WARMUP', 'recovery_left': 0, 'tighten': False}
        self._gated_sc = bool(getattr(args, 'gated_sc', 0))
        self._gated_thr = 8   # 冻结：b_t>=8 触发（fixed8 同阈值）
        self._gated_r = 3     # 冻结：报警结束后 R=3 恢复窗
        self._gated_lo = 0.7  # 冻结：收紧目标 match_thresh
        #self.det_thresh = args.track_thresh
        # E8 S-hiconf：出生候选分数门槛覆盖（默认 -1 = track_thresh + 0.1）
        _dt = float(getattr(args, 'det_thresh', -1.0) or -1.0)
        self.det_thresh = _dt if _dt > 0 else args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        BaseTrack._count = 0  # added by F&F attack, reset track id for each sub video sequence

        # E17: default-off suspect lifecycle and local conditional association.
        self.suspect_mode = bool(getattr(args, 'suspect_mode', False))
        self.suspect_hits = int(getattr(args, 'suspect_hits', 3))
        self.suspect_recovery = int(getattr(args, 'suspect_recovery', 2))
        self.local_assoc_mode = bool(getattr(args, 'local_assoc_mode', False))
        self.local_match_thresh = float(getattr(args, 'local_match_thresh', 0.80))
        self.local_recovery = int(getattr(args, 'local_recovery', 2))
        self.fn_recovery_mode = bool(getattr(args, 'fn_recovery_mode', False))
        self.fn_recovery_iou = float(getattr(args, 'fn_recovery_iou', 0.50))
        self.fn_recovery_center = float(getattr(args, 'fn_recovery_center', 0.35))
        self.fn_recovery_max_lost = int(getattr(args, 'fn_recovery_max_lost', 3))
        self.fn_recovery_window = int(getattr(args, 'fn_recovery_window', 2))
        # E18: causal heterogeneous-detector review, default off.
        self.aux_review_mode = bool(getattr(args, 'aux_review_mode', False))
        self.aux_filter_mode = bool(getattr(args, 'aux_filter_mode', False))
        self.aux_fn_mode = bool(getattr(args, 'aux_fn_mode', False))
        self.aux_candidate_file = str(getattr(args, 'aux_candidate_file', ''))
        self.aux_condition = str(getattr(args, 'aux_condition', 'A'))
        self.aux_conf = float(getattr(args, 'aux_conf', 0.10))
        self.aux_iou = float(getattr(args, 'aux_iou', 0.30))
        self.aux_add_conf = float(getattr(args, 'aux_add_conf', 0.25))
        self.aux_track_iou = float(getattr(args, 'aux_track_iou', 0.10))
        self.aux_recovery = int(getattr(args, 'aux_recovery', 2))
        self._aux_review_active = False
        if self.suspect_mode or self.local_assoc_mode:
            self._quarantine = []
        self._cg_last_window_frame = -1000000
        self._cg_window_active = False
        self._cg_assoc_window_active = False
        self.cg_stats = {
            'trigger_frames': 0, 'alarm_state_frames': 0,
            'intervention_frames': 0, 'window_frames': 0,
            'suspect_parked': 0, 'suspect_confirmed': 0,
            'suspect_discarded': 0, 'suspect_expired': 0,
            'confirmation_delays': [],
            'confirmed_priority_blocked_pairs': 0,
            'local_assoc_blocked_pairs': 0,
            'affected_tracks_marked': 0,
            'fn_recovery_eligible_lost_track_frames': 0,
            'fn_recovery_low_score_detections': 0,
            'fn_recovery_candidate_pairs': 0,
            'fn_recovery_active_conflict_detections': 0,
            'fn_recovery_matches': 0,
            'aux_review_frames': 0,
            'aux_birth_review_frames': 0,
            'aux_original_detections': 0,
            'aux_filter_retained': 0,
            'aux_filter_removed': 0,
            'aux_filter_track_protected': 0,
            'aux_fn_candidates': 0,
            'aux_fn_added': 0,
            'aux_fn_duplicate_suppressed': 0,
            'aux_birth_rejected': 0,
        }

        self.use_busca = bool(getattr(args, 'use_busca', False))
        self.busca_thresh = float(getattr(args, 'busca_thresh', 0.8))
        self.use_camera_motion_compensation = bool(
            getattr(args, 'use_camera_motion_compensation', False))
        self.last_image = None
        self.busca_tracker = None
        if self.use_busca:
            from busca.network import BUSCA
            busca_args = self.args.transformer
            busca_args.device = torch.device("cuda" if self.args.device == "gpu" else "cpu")
            self.busca_tracker = BUSCA(busca_args).to(busca_args.device)
            self.busca_tracker.load_pretrained(self.args.busca_ckpt, ignore_reid_fc=True)
            self.busca_tracker.eval()

        self.trackshield = None
        if getattr(args, 'trackshield_mode', 'off') in ('audit', 'calibrate', 'ts_cold', 'ts_warm'):
            from .trackshield import TrackShieldController
            self.trackshield = TrackShieldController(self)

    def _l1_update(self, b_t):
        """E4-4 冻结状态机 v1.0：W=5 k=3 r=3 c_min=8 A_max=5（E4-3R 规格）。
        因果：只用 hist（过去帧），b_t 为当前帧观测；返回 (alarm, thr, state_in)。"""
        l1 = self._l1
        hist, raws = l1['hist'], l1['raws']
        thr = None
        med = iqr = None
        if hist:
            med = float(np.median(hist))
            iqr = float(np.percentile(hist, 75) - np.percentile(hist, 25))
            thr = max(med + 3.0 * iqr, 3.0 * med, 8.0)
        state_in = l1['state']
        if l1['state'] == 'WARMUP' and len(hist) >= 5:
            l1['state'] = 'NORMAL'
        if l1['state'] == 'NORMAL' and thr is not None and b_t >= thr:
            l1['state'] = 'ALARM'
        alarm = l1['state'] == 'ALARM'
        if alarm:
            l1['alarm_run'] += 1
            if b_t < thr:
                l1['state'] = 'NORMAL'
                l1['alarm_run'] = 0
            elif l1['alarm_run'] >= 5:
                r_t = float(np.median(raws[-5:])) if len(raws) >= 5 else 0.0
                l1['drift'] += 1
                if r_t >= thr:
                    l1['hist'] = list(raws[-5:])  # DRIFT_RESET（记污染风险）
                l1['state'] = 'NORMAL'
                l1['alarm_run'] = 0
        raws.append(b_t)
        if not alarm:
            hist.append(b_t)
            if len(hist) > 5:
                hist.pop(0)
        return alarm, thr, med, iqr, state_in

    def _fixed8_update(self, b_t):
        """E4-4R G3 固定阈值负对照：帧 1–5 WARMUP 不报警不干预；帧≥6 trigger=b_t>=8，
        alarm_pre=state==ALARM，intervene=alarm_pre or trigger（退出帧仍干预一次，与 G2 共享）。
        无 median/IQR/drift；返回 (intervene, thr, med, iqr, state_in) 对齐 _l1_update 签名。"""
        l1 = self._l1
        state_in = l1['state']
        if self.frame_id <= 5:
            l1['state'] = 'WARMUP'
            return False, 8.0, None, None, state_in
        trigger = (b_t >= 8)
        alarm_pre = (l1['state'] == 'ALARM')
        intervene = alarm_pre or trigger
        l1['state'] = 'ALARM' if trigger else 'NORMAL'
        if trigger:
            l1['alarm_run'] += 1
        else:
            l1['alarm_run'] = 0
        return intervene, 8.0, None, None, state_in

    def _s1_update(self, s_t, n_active):
        """E5-4 v1.1 S1 Survival 状态机：帧≤5 WARMUP；帧>5 alarm=(n_active>=3 且 s_t>=thr)。
        ALARM→(报警结束)→RECOVERY(R=3)→NORMAL。self._s1['intervene'] 供下一帧 priority pass 使用
        （state_in 语义，1 帧因果滞后：s_t 在 stage1+stage2 后才可得，与 G3 b_t 的 step4 可得不同）。
        返回 (alarm_this_frame, state_in)。"""
        s1 = self._s1
        state_in = s1['state']
        if self.frame_id <= 5:
            s1['state'] = 'WARMUP'
            s1['recovery_left'] = 0
            s1['intervene'] = False
            return False, state_in
        alarm_cond = (n_active >= 3) and (s_t >= self._s1_thr)
        if s1['state'] == 'ALARM':
            if not alarm_cond:
                s1['state'] = 'RECOVERY'
                s1['recovery_left'] = 3  # R=3 恢复窗（冻结）
        elif s1['state'] == 'RECOVERY':
            if alarm_cond:
                s1['state'] = 'ALARM'
            else:
                s1['recovery_left'] -= 1
                if s1['recovery_left'] <= 0:
                    s1['state'] = 'NORMAL'
        else:  # NORMAL
            if alarm_cond:
                s1['state'] = 'ALARM'
        s1['intervene'] = s1['state'] in ('ALARM', 'RECOVERY')
        return alarm_cond, state_in

    def _gated_update(self, b_t):
        """E9 audit-gated 关联收紧状态机：帧≤5 WARMUP 不收紧；帧>5 trigger=b_t>=8。
        ALARM→(报警结束)→RECOVERY(R=3)→NORMAL，与 S1 同构，触发信号为 G3 的 b_t。
        self._gated['tighten'] 供下一帧 stage1 使用（1 帧因果滞后）。返回 tighten_this_frame。"""
        g = self._gated
        if self.frame_id <= 5:
            g['state'] = 'WARMUP'
            g['recovery_left'] = 0
            g['tighten'] = False
            return False
        trigger = (b_t >= self._gated_thr)
        if g['state'] == 'ALARM':
            if not trigger:
                g['state'] = 'RECOVERY'
                g['recovery_left'] = self._gated_r
        elif g['state'] == 'RECOVERY':
            if trigger:
                g['state'] = 'ALARM'
            else:
                g['recovery_left'] -= 1
                if g['recovery_left'] <= 0:
                    g['state'] = 'NORMAL'
        else:  # NORMAL / WARMUP
            g['state'] = 'NORMAL'  # 帧>5 首次退出 WARMUP（纯语义标签；WARMUP/NORMAL 均 tighten=False）
            if trigger:
                g['state'] = 'ALARM'
        g['tighten'] = g['state'] in ('ALARM', 'RECOVERY')
        return g['tighten']

    def _birth_scorer_cfg(self):
        # E4-1: 冻结出生评分器（只读缓存）
        if not hasattr(self, '_bsc'):
            import json as _json
            self._bsc = {}
            p = getattr(self.args, 'birth_scorer', '')
            if p and os.path.exists(p):
                self._bsc = _json.load(open(p))
        return self._bsc

    def _birth_thr(self):
        cfg = self._birth_scorer_cfg().get(getattr(self.args, 'current_seg', ''), {})
        return cfg.get('thr_p95', float('inf'))

    def _birth_risk_z(self, det_track, n_birth_frame):
        # E4-1: 与 E3-2R 冻结定义一致（det_score↓ dup_iou↑ n_births↑，max z）
        cfg = self._birth_scorer_cfg().get(getattr(self.args, 'current_seg', ''), {})
        if not cfg:
            return None
        dup = 0.0
        for t in list(self.tracked_stracks) + list(self.lost_stracks):
            v = _tlbr_iou(det_track.tlbr, t.tlbr)
            if v > dup:
                dup = v
        feats = {'det_score': float(det_track.score), 'dup_iou': dup,
                 'n_births_same_frame': float(n_birth_frame)}
        dirs = {'det_score': -1, 'dup_iou': 1, 'n_births_same_frame': 1}
        return max(dirs[k] * (feats[k] - cfg['median'][k]) / cfg['iqr'][k] for k in feats)

    def _randq_cfg(self):
        # E8 S-randq：配额/park 概率（离线预计算，键=段名）。缓存只读。
        if not hasattr(self, '_randq_map'):
            import json as _json
            self._randq_map = {}
            p = getattr(self.args, 'randq_quota_json', '')
            if p and os.path.exists(p):
                self._randq_map = _json.load(open(p))
        return self._randq_map

    def _randq_init(self):
        # E8 S-randq：每段首次调用时初始化（tracker 每段新建，但 current_seg 在 update 时已就绪）
        if not hasattr(self, '_randq_seg'):
            cfg = self._randq_cfg().get(getattr(self.args, 'current_seg', ''), {})
            self._randq_seg = getattr(self.args, 'current_seg', '')
            self._randq_quota = int(cfg.get('quota', 0))
            self._randq_p = float(cfg.get('p', 0.0))
            self._randq_parked = 0
            self._randq_seen = 0
            self._randq_rng = random.Random(int(getattr(self.args, 'randq_seed', 0)))

    def _oracle_max_iou(self, det_track):
        # E8 Oracle（上限，GT 辅助，仅离线参考）：返回候选与有效 GT 的 max IoU（<0.5 判定假出生）。
        gt = getattr(self.args, 'current_gt', None)
        if not gt:
            return 0.0
        best = 0.0
        a = det_track.tlbr
        for cx, cy, w, h in gt:
            b = [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
            v = _tlbr_iou(a, b)
            if v > best:
                best = v
        return best

    def _cg_association_guard(self, dists, strack_pool, detections, base_thresh):
        """Protect confirmed tracks and tighten only suspect/affected pairs."""
        if not (self.suspect_mode or self.local_assoc_mode) or \
                not strack_pool or not detections:
            return dists
        f = self.frame_id
        suspect = [bool(getattr(t, '_cg_suspect_until', -1) >= f)
                   for t in strack_pool]
        affected = [bool(getattr(t, '_cg_affected_until', -1) >= f)
                    for t in strack_pool]
        if self.local_assoc_mode and self._cg_assoc_window_active and self._quarantine:
            for i, track in enumerate(strack_pool):
                if affected[i]:
                    continue
                affected[i] = any(
                    _tlbr_iou(track.tlbr, q['tlbr']) > 0.1
                    for q in self._quarantine)
        priority_blocks = 0
        if self.suspect_mode and any(suspect):
            confirmed = [i for i, flag in enumerate(suspect) if not flag]
            for i, flag in enumerate(suspect):
                if not flag:
                    continue
                for j in range(len(detections)):
                    if dists[i, j] > base_thresh:
                        continue
                    if any(dists[k, j] <= base_thresh for k in confirmed):
                        dists[i, j] = 1e6
                        priority_blocks += 1
        local_blocks = 0
        if self.local_assoc_mode and self._cg_assoc_window_active:
            for i, track in enumerate(strack_pool):
                if not (suspect[i] or affected[i]):
                    continue
                for j in range(len(detections)):
                    if self.local_match_thresh < dists[i, j] <= base_thresh:
                        dists[i, j] = 1e6
                        local_blocks += 1
        self.cg_stats['confirmed_priority_blocked_pairs'] += priority_blocks
        self.cg_stats['local_assoc_blocked_pairs'] += local_blocks
        if priority_blocks or local_blocks:
            PreCommitLogger.get().log(
                'cg_assoc', f, confirmed_priority_blocks=int(priority_blocks),
                local_assoc_blocks=int(local_blocks),
                suspect_tracks=int(sum(suspect)), affected_tracks=int(sum(affected)),
                local_thresh=float(self.local_match_thresh))
        return dists

    def _aux_candidates(self):
        """Load frozen, image-derived detections; no GT or future frames are used."""
        if not hasattr(self, '_aux_map'):
            if not self.aux_candidate_file:
                raise ValueError('aux_candidate_file is required when aux_review_mode=True')
            import json as _json
            self._aux_map = {}
            with open(self.aux_candidate_file, encoding='utf-8') as f:
                for line in f:
                    rec = _json.loads(line)
                    frame = int(rec['f'])
                    if frame in (6, 7, 8):
                        expected_kind = 'adversarial' if self.aux_condition == 'A' else 'clean_reference'
                    else:
                        expected_kind = 'clean_recovery'
                    if rec.get('input_kind') != expected_kind or frame < 6:
                        raise ValueError(f'Invalid E18 candidate input for frame {frame}')
                    key = (rec['seg'], frame)
                    if key in self._aux_map:
                        raise ValueError(f'Duplicate auxiliary candidate record: {key}')
                    self._aux_map[key] = rec['dets']
        key = (getattr(self.args, 'current_seg', ''), self.frame_id)
        if key not in self._aux_map:
            raise KeyError(f'Missing auxiliary candidates: {key}')
        return np.asarray(self._aux_map[key], dtype=np.float64).reshape(-1, 5)

    def _apply_aux_review(self, output_results, bboxes, scores, scale):
        """Filter by detector consensus and add only track-supported FN candidates."""
        self.cg_stats['aux_review_frames'] += 1
        aux = self._aux_candidates()
        reviewed_aux = aux[aux[:, 4] >= self.aux_conf]
        reviewed_boxes = reviewed_aux[:, :4].copy()
        if len(reviewed_boxes):
            reviewed_boxes[:, 2:] += reviewed_boxes[:, :2]
        original_n = len(scores)
        self.cg_stats['aux_original_detections'] += original_n
        track_boxes = [t.tlbr for t in list(self.tracked_stracks) + list(self.lost_stracks)]
        if self.aux_filter_mode:
            if original_n and len(reviewed_aux):
                support = _tlbr_iou_matrix(bboxes, reviewed_boxes).max(axis=1)
            else:
                support = np.zeros(original_n, dtype=np.float64)
            if original_n and track_boxes:
                track_support = _tlbr_iou_matrix(bboxes, np.asarray(track_boxes)).max(axis=1)
            else:
                track_support = np.zeros(original_n, dtype=np.float64)
            # Detector disagreement alone must not erase an observation of an existing track.
            keep = np.logical_or(support >= self.aux_iou, track_support >= self.aux_track_iou)
            removed = int((~keep).sum())
            self.cg_stats['aux_filter_retained'] += int(keep.sum())
            self.cg_stats['aux_filter_removed'] += removed
            self.cg_stats['aux_filter_track_protected'] += int(np.logical_and(
                support < self.aux_iou, track_support >= self.aux_track_iou).sum())
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
                if track_support < self.aux_track_iou:
                    continue
                self.cg_stats['aux_fn_candidates'] += 1
                if len(bboxes) and _tlbr_iou_matrix(np.asarray([box]), bboxes).max() >= 0.5:
                    self.cg_stats['aux_fn_duplicate_suppressed'] += 1
                    continue
                added.append([box[0], box[1], box[2], box[3], float(det[4])])
        if added:
            extra = np.asarray(added, dtype=np.float64)
            extra_output = np.column_stack([
                extra[:, :4], extra[:, 4], np.ones(len(extra)), np.zeros(len(extra))])
            output_results = np.concatenate([output_results, extra_output], axis=0)
            bboxes = np.concatenate([bboxes, extra[:, :4]], axis=0)
            scores = np.concatenate([scores, extra[:, 4]], axis=0)
        self.cg_stats['aux_fn_added'] += len(added)
        PreCommitLogger.get().log(
            'aux_review', self.frame_id, raw_n=int(original_n),
            aux_n=int(len(reviewed_aux)), removed=removed, added=int(len(added)),
            conf=float(self.aux_conf), iou=float(self.aux_iou),
            add_conf=float(self.aux_add_conf), track_iou=float(self.aux_track_iou))
        return output_results, bboxes, scores

    def update(self, output_results, img_info, img_size, current_frame=None):
        if self.use_busca and current_frame is None:
            raise ValueError("current_frame must be provided when use_busca=True")
        self.frame_id += 1
        if self.suspect_mode or self.local_assoc_mode or self.fn_recovery_mode or self.aux_review_mode:
            recovery = max(
                self.suspect_recovery if self.suspect_mode else 0,
                self.local_recovery if self.local_assoc_mode else 0,
                self.fn_recovery_window if self.fn_recovery_mode else 0,
                self.aux_recovery if self.aux_review_mode else 0)
            # b_t is observed after association; all review uses only prior-frame state.
            self._cg_assoc_window_active = (
                self.frame_id - self._cg_last_window_frame <= recovery)
        self._aux_review_active = bool(self.aux_review_mode and self._cg_window_active)
        if self.trackshield is not None:
            self.trackshield.begin_frame()
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # E14: 检测器无输出时等价于零检测，仍完整推进 tracker/G3 状态。
        if output_results is None:
            output_results = torch.zeros((0, 6))
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        if self._aux_review_active:
            output_results, bboxes, scores = self._apply_aux_review(
                output_results, bboxes, scores, scale)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        inds_all_considered = np.logical_or(remain_inds, inds_second)
        bboxes_all_considered = bboxes[inds_all_considered]
        scores_all_considered = scores[inds_all_considered]

        if self.use_busca and self.busca_thresh > 0:
            images_second = self.busca_tracker.get_image_crops(
                image=current_frame, bboxes=dets_second * scale, normalize=False)
            images_first = self.busca_tracker.get_image_crops(
                image=current_frame, bboxes=dets * scale, normalize=False)
            images_all_considered = self.busca_tracker.get_image_crops(
                image=current_frame, bboxes=bboxes_all_considered * scale, normalize=False)
        else:
            images_second = [None] * len(dets_second)
            images_first = [None] * len(dets)
            images_all_considered = [None] * len(bboxes_all_considered)

        pcl = PreCommitLogger.get()  # E3-2 instrumentation (no-op unless PRECOMMIT_LOG_DIR set)
        pcl.log('dets', self.frame_id,
                n_raw=int(output_results.shape[0]),
                high=[[float(x) for x in STrack.tlbr_to_tlwh(b)] + [float(s)]
                      for b, s in zip(dets, scores_keep)],
                second=[[float(x) for x in STrack.tlbr_to_tlwh(b)] + [float(s)]
                        for b, s in zip(dets_second, scores_second)])
        # E9 audit-gated 收紧：本帧 stage1 用上一帧状态机结论（1 帧因果滞后）
        if self._gated_sc:
            match_thresh_eff = self._gated_lo if self._gated['tighten'] else float(self.args.match_thresh)
        else:
            match_thresh_eff = float(self.args.match_thresh)
        pcl.log('cfg', self.frame_id, match_thresh=float(self.args.match_thresh),
                match_thresh_eff=float(match_thresh_eff),
                track_thresh=float(self.args.track_thresh), det_thresh=float(self.det_thresh),
                l1_mode=getattr(self.args, 'l1_mode', 'off'))

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.args, image=im, scale=scale) for
                          (tlbr, s, im) in zip(dets, scores_keep, images_first)]
        else:
            detections = []
        if len(bboxes) > 0:
            all_considered_dets = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.args, image=im, scale=scale)
                                   for (tlbr, s, im) in zip(bboxes_all_considered, scores_all_considered,
                                                            images_all_considered)]
        else:
            all_considered_dets = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        n_active_start = len(tracked_stracks)  # E5-4 S-monitor 分母：帧起始活动 track 数

        # E5-4 v1.1：分块 KF 预测（lost 单独一次，供 priority pass 复用；KF 逐轨迹独立，等价原联合预测）
        STrack.multi_predict(tracked_stracks)
        STrack.multi_predict(self.lost_stracks)

        # E5-4 v1.1 S1 priority pass：报警/恢复窗内，lost 轨迹先于 stage1 认领检测（fused 阈值 1.0 完全放宽）
        s1_on = self._s1_thr > 0.0
        n_priority_matches = 0
        if s1_on and self._s1['intervene'] and len(self.lost_stracks) > 0 and len(detections) > 0:
            lost_pool = list(self.lost_stracks)
            pdist = matching.iou_distance(lost_pool, detections)
            pdist_raw = pdist
            if not self.args.mot20:
                pdist = matching.fuse_score(pdist, detections)
            # E5-4 v1.2 ⑤c 零 IoU 护栏：IoU=0（iou_dist==1.0）的对设为禁止匹配。
            # "认领返回观测"隐含空间相邻；零重叠认领会错认无关检测并制造 IDSW。
            pdist[pdist_raw >= 1.0] = 1e9
            pmatches, pu_lost, pu_det = matching.linear_assignment(pdist, thresh=1.0)
            matched_lost_ids = set()
            matched_det_idx = set()
            for i, j in pmatches:
                track = lost_pool[i]
                det = detections[j]
                track.re_activate(det, self.frame_id, new_id=False,
                                  update_mems=det.score >= self.det_thresh)
                refind_stracks.append(track)
                matched_lost_ids.add(track.track_id)
                matched_det_idx.add(int(j))
                n_priority_matches += 1
                pcl.log('s1_priority_match', self.frame_id, track_id=int(track.track_id),
                        det_idx=int(j), fused=float(pdist[i, j]), iou_dist=float(pdist_raw[i, j]),
                        det_score=float(det.score))
            if matched_det_idx:
                detections = [d for k, d in enumerate(detections) if k not in matched_det_idx]
            if matched_lost_ids:
                self.lost_stracks = [t for t in self.lost_stracks if t.track_id not in matched_lost_ids]

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        if self.trackshield is not None:
            strack_pool, detections = self.trackshield.cap_association('stage1', strack_pool, detections)
        dists = matching.iou_distance(strack_pool, detections)
        dists_raw1 = dists  # E3-2: fuse_score returns a new array; dists stays raw IoU-dist
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if self.suspect_mode or self.local_assoc_mode:
            dists = self._cg_association_guard(
                dists, strack_pool, detections, float(match_thresh_eff))

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thresh_eff)
        if self.trackshield is not None:
            self.trackshield.record_matches('stage1', len(matches))
        pcl.log_stage('stage1', self.frame_id, strack_pool, detections,
                      dists_raw1, dists, matches, u_track, u_detection)


        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            update_mems = det.score >= self.det_thresh
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, update_mems=update_mems)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, update_mems=update_mems)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.args, image=im, scale=scale) for
                                 (tlbr, s, im) in zip(dets_second, scores_second, images_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        r_lost_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state != TrackState.Tracked]
        ts_stage2_excluded = set()
        if self.trackshield is not None:
            r_tracked_stracks, detections_second = self.trackshield.cap_association('stage2', r_tracked_stracks, detections_second)
            ts_stage2_excluded = {id(track) for track in self.trackshield.excluded_tracks['stage2']}
        dists = matching.iou_distance(r_tracked_stracks, detections_second)

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        if self.trackshield is not None:
            self.trackshield.record_matches('stage2', len(matches))
        pcl.log_stage('stage2', self.frame_id, r_tracked_stracks, detections_second,
                      dists, dists, matches, u_track, u_detection_second)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            update_mems = not bool(getattr(self.args, 'transformer_update_mems_only_first_round', False))
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, update_mems=update_mems)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, update_mems=update_mems)
                refind_stracks.append(track)

        # E17 conservative FN recovery: only existing, still-unmatched low-score
        # detections and recently lost, previously observed tracks are eligible.
        if self.fn_recovery_mode and self._cg_assoc_window_active:
            remaining_second = [detections_second[j] for j in u_detection_second]
            eligible_lost = []
            for track in r_lost_stracks:
                age = self.frame_id - track.end_frame
                if (track.state == TrackState.Lost and track.is_activated and
                        track.mean is not None and len(track.tlwh_mem) >= 3 and
                        float(track.score) >= self.det_thresh and
                        0 < age <= self.fn_recovery_max_lost):
                    eligible_lost.append(track)
            self.cg_stats['fn_recovery_eligible_lost_track_frames'] += len(eligible_lost)
            self.cg_stats['fn_recovery_low_score_detections'] += len(remaining_second)
            active_tracks = [t for t in strack_pool
                             if t.state == TrackState.Tracked and t.is_activated]
            usable_detections = []
            for det in remaining_second:
                conflict = any(_tlbr_iou(track.tlbr, det.tlbr) > 0.5
                               for track in active_tracks)
                self.cg_stats['fn_recovery_active_conflict_detections'] += int(conflict)
                if not conflict:
                    usable_detections.append(det)
            if eligible_lost and usable_detections:
                cost = np.ones((len(eligible_lost), len(usable_detections)), dtype=float)
                diagnostics = {}
                for i, track in enumerate(eligible_lost):
                    predicted = track.tlwh
                    predicted_center = predicted[:2] + predicted[2:] / 2.0
                    last = track.tlwh_mem[-1]
                    last_center = last[:2] + last[2:] / 2.0
                    predicted_motion = predicted_center - last_center
                    predicted_norm = float(np.linalg.norm(predicted_motion))
                    box_scale = max(float(np.sqrt(abs(predicted[2] * predicted[3]))), 1.0)
                    for j, det in enumerate(usable_detections):
                        iou = 1.0 - float(matching.iou_distance([track], [det])[0, 0])
                        det_tlwh = det.tlwh
                        det_center = det_tlwh[:2] + det_tlwh[2:] / 2.0
                        center_delta = float(np.linalg.norm(det_center - predicted_center)) / box_scale
                        observed_motion = det_center - last_center
                        denom = predicted_norm * float(np.linalg.norm(observed_motion))
                        motion_cosine = (float(np.dot(predicted_motion, observed_motion)) / denom
                                         if denom > 1e-12 else (1.0 if predicted_norm <= 1e-12 else 0.0))
                        if (iou >= self.fn_recovery_iou and
                                center_delta <= self.fn_recovery_center and motion_cosine >= 0.0):
                            cost[i, j] = 1.0 - iou
                            diagnostics[(i, j)] = (iou, center_delta, motion_cosine)
                self.cg_stats['fn_recovery_candidate_pairs'] += len(diagnostics)
                recovery_matches, _, _ = matching.linear_assignment(
                    cost, thresh=1.0 - self.fn_recovery_iou + 1e-9)
                for i, j in recovery_matches:
                    track = eligible_lost[i]
                    det = usable_detections[j]
                    iou, center_delta, motion_cosine = diagnostics[(i, j)]
                    lost_age = self.frame_id - track.end_frame
                    track.re_activate(det, self.frame_id, new_id=False, update_mems=False)
                    refind_stracks.append(track)
                    self.cg_stats['fn_recovery_matches'] += 1
                    pcl.log('fn_recovery', self.frame_id, action='reassociated',
                            id=int(track.track_id), score=float(det.score), iou=float(iou),
                            center_delta=float(center_delta), motion_cosine=float(motion_cosine),
                            lost_age=int(lost_age))

        unassigned_stracks = r_tracked_stracks
        if self.use_busca and self.busca_thresh > 0:
            unassigned_stracks = joint_stracks(
                [r_tracked_stracks[it] for it in u_track], r_lost_stracks)
            u_track = list(range(len(unassigned_stracks)))
            reliable_thresh = getattr(self.args, 'reliable_thresh', None)
            if reliable_thresh is not None and not self.is_reliable(
                    current_frame=current_frame, active_stracks=self.tracked_stracks,
                    p=reliable_thresh):
                third_round_stracks = []
            else:
                third_round_stracks = unassigned_stracks
                if self.use_camera_motion_compensation:
                    self.camera_motion_compensation(
                        track_pool=third_round_stracks, current_frame=current_frame)
                extra_kalman_candidates = self.get_extra_kalman_candidates(
                    strack_pool=third_round_stracks, frame_img=current_frame)
                third_round_matches, third_round_u_track = self.third_round_association(
                    strack_pool=third_round_stracks, considered_dets=all_considered_dets,
                    extra_kalman_candidates=extra_kalman_candidates,
                    asoc_thresh=self.busca_thresh)
                for itracked, prob in third_round_matches:
                    track = third_round_stracks[itracked]
                    det = extra_kalman_candidates[itracked]
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id, update_mems=False)
                        activated_starcks.append(track)
                u_track = third_round_u_track

        for it in u_track:
            track = unassigned_stracks[it]
            if id(track) in ts_stage2_excluded:
                continue
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                pcl.log('lifecycle', self.frame_id, ev='lost', id=int(track.track_id))

        if self.local_assoc_mode and self._cg_assoc_window_active:
            for it in u_track:
                track = unassigned_stracks[it]
                track._cg_affected_until = self.frame_id + self.local_recovery
                self.cg_stats['affected_tracks_marked'] += 1

        # E5-4 S-monitor：s_t = 未获匹配活动 track / 起始活动 track（stage1+stage2 后统计）
        if s1_on:
            n_unmatched_active = sum(
                1 for it in u_track if unassigned_stracks[it].state == TrackState.Tracked)
            s_t = (float(n_unmatched_active) / n_active_start) if n_active_start > 0 else 0.0
            alarm, state_in = self._s1_update(s_t, n_active_start)
            pcl.log('s1_state', self.frame_id, s_t=round(s_t, 6), thr=round(self._s1_thr, 6),
                    alarm=int(alarm), state_in=state_in, state_out=self._s1['state'],
                    n_active=int(n_active_start), n_priority_matches=int(n_priority_matches),
                    intervene=int(self._s1['intervene']))

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        if self.trackshield is not None:
            unconfirmed, detections = self.trackshield.cap_association('stage3', unconfirmed, detections)
        dists = matching.iou_distance(unconfirmed, detections)
        dists_raw3 = dists
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        if self.trackshield is not None:
            self.trackshield.record_matches('stage3', len(matches))
        pcl.log_stage('stage3_unconfirmed', self.frame_id, unconfirmed, detections,
                      dists_raw3, dists, matches, u_unconfirmed, u_detection)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            pcl.log('lifecycle', self.frame_id, ev='removed_unconfirmed', id=int(track.track_id))
        if self.trackshield is not None:
            activated_starcks.extend(self.trackshield.excluded_tracks['stage3'])

        if self.trackshield is not None:
            self.trackshield.after_association(len(u_detection), int(output_results.shape[0]))

        """ Step 4: Init new stracks"""
        # ---- E4-1 出生隔离（仅改出生入口；确认/丢弃语义见 E4_1_protocol.md） ----
        qmode = getattr(self.args, 'quarantine_mode', 'off')
        l1_on = getattr(self.args, 'l1_mode', 'off') != 'off'
        l1_iv = bool(getattr(self.args, 'l1_intervene', 0))
        if (qmode != 'off' or l1_iv) and not hasattr(self, '_quarantine'):
            self._quarantine = []
        u_detection = list(u_detection)
        n_skip_birth = 0
        if qmode != 'off' or l1_iv:
            if not self.suspect_mode:
                for q in self._quarantine:  # 原 G3：上一帧候选在本帧确认或丢弃
                    best_iou, best_j = 0.5, -1
                    for j in u_detection:
                        d = detections[j]
                        if d.score < self.det_thresh:
                            continue
                        v = _tlbr_iou(q['tlbr'], d.tlbr)
                        if v > best_iou:
                            best_iou, best_j = v, j
                    if best_j >= 0:
                        det = detections[best_j]
                        det.activate(self.kalman_filter, self.frame_id)
                        activated_starcks.append(det)
                        pcl.log('birth_q', self.frame_id, action='confirmed', id=int(det.track_id),
                                score=float(det.score), iou=float(best_iou),
                                parked_frame=int(q['frame']), tlwh=[float(x) for x in det.tlwh])
                        pcl.log('lifecycle', self.frame_id, ev='birth', id=int(det.track_id),
                                score=float(det.score), tlwh=[float(x) for x in det.tlwh],
                                activated=bool(det.is_activated))
                        u_detection.remove(best_j)
                    else:
                        pcl.log('birth_q', self.frame_id, action='discarded',
                                score=float(q['score']), parked_frame=int(q['frame']),
                                tlbr=[float(x) for x in q['tlbr']])
                self._quarantine = []
            else:
                # E17: 2/3 个连续帧观测才确认；任一帧无匹配即丢弃。
                # 候选仍在高置信度关联之后处理，不能抢占本帧已有轨迹的检测。
                next_queue = []
                for q in self._quarantine:
                    if not self._cg_assoc_window_active:
                        self.cg_stats['suspect_expired'] += 1
                        pcl.log('birth_q', self.frame_id, action='expired',
                                score=float(q['score']), parked_frame=int(q['frame']),
                                hits=int(q.get('hits', 1)), tlbr=[float(x) for x in q['tlbr']])
                        continue
                    best_iou, best_j = 0.5, -1
                    for j in u_detection:
                        d = detections[j]
                        if d.score < self.det_thresh:
                            continue
                        v = _tlbr_iou(q['tlbr'], d.tlbr)
                        if v > best_iou:
                            best_iou, best_j = v, j
                    if best_j < 0:
                        self.cg_stats['suspect_discarded'] += 1
                        pcl.log('birth_q', self.frame_id, action='discarded',
                                score=float(q['score']), parked_frame=int(q['frame']),
                                hits=int(q.get('hits', 1)), tlbr=[float(x) for x in q['tlbr']])
                        continue
                    det = detections[best_j]
                    hits = int(q.get('hits', 1)) + 1
                    u_detection.remove(best_j)
                    if hits >= self.suspect_hits:
                        det.activate(self.kalman_filter, self.frame_id)
                        det._cg_suspect_until = self.frame_id + self.suspect_recovery
                        activated_starcks.append(det)
                        delay = self.frame_id - int(q['frame'])
                        self.cg_stats['suspect_confirmed'] += 1
                        self.cg_stats['confirmation_delays'].append(delay)
                        pcl.log('birth_q', self.frame_id, action='confirmed', id=int(det.track_id),
                                score=float(det.score), iou=float(best_iou), hits=hits,
                                delay_frames=delay, parked_frame=int(q['frame']),
                                tlwh=[float(x) for x in det.tlwh])
                        pcl.log('lifecycle', self.frame_id, ev='birth', id=int(det.track_id),
                                score=float(det.score), tlwh=[float(x) for x in det.tlwh],
                                activated=bool(det.is_activated))
                    else:
                        q.update(tlbr=det.tlbr.copy(), score=float(det.score),
                                 hits=hits, last_frame=self.frame_id)
                        next_queue.append(q)
                        pcl.log('birth_q', self.frame_id, action='continued',
                                score=float(det.score), iou=float(best_iou), hits=hits,
                                parked_frame=int(q['frame']), tlwh=[float(x) for x in det.tlwh])
                self._quarantine = next_queue
        cand_n = sum(1 for i in u_detection if detections[i].score >= self.det_thresh)
        # E4-4：L1 状态更新在出生处理之前（因果）；G2 报警帧全隔离
        l1_alarm = False
        if l1_on:
            if self.args.l1_mode == 'fixed8':
                l1_alarm, l1_thr, l1_med, l1_iqr, l1_sin = self._fixed8_update(cand_n)
            else:
                l1_alarm, l1_thr, l1_med, l1_iqr, l1_sin = self._l1_update(cand_n)
            pcl.log('l1_state', self.frame_id, b_t=int(cand_n),
                    thr=l1_thr if l1_thr is not None else -1,
                    med=l1_med if l1_med is not None else -1,
                    iqr=l1_iqr if l1_iqr is not None else -1,
                    state_in=l1_sin, state_out=self._l1['state'],
                    alarm_run=int(self._l1['alarm_run']), drift_evals=int(self._l1['drift']))
        if self.suspect_mode or self.local_assoc_mode or self.fn_recovery_mode or self.aux_review_mode:
            state_out = self._l1['state'] if l1_on else 'OFF'
            if l1_on and self.args.l1_mode == 'fixed8':
                trigger = self.frame_id > 5 and cand_n >= 8
            else:
                trigger = l1_on and state_out == 'ALARM' and l1_sin != 'ALARM'
            self.cg_stats['trigger_frames'] += int(trigger)
            self.cg_stats['alarm_state_frames'] += int(state_out == 'ALARM')
            self.cg_stats['intervention_frames'] += int(l1_alarm)
            if l1_alarm:
                self._cg_last_window_frame = self.frame_id
            recovery = max(
                self.suspect_recovery if self.suspect_mode else 0,
                self.local_recovery if self.local_assoc_mode else 0,
                self.fn_recovery_window if self.fn_recovery_mode else 0,
                self.aux_recovery if self.aux_review_mode else 0)
            self._cg_window_active = bool(
                l1_alarm or self.frame_id - self._cg_last_window_frame <= recovery)
            self.cg_stats['window_frames'] += int(self._cg_window_active)
        # E9 audit-gated 收紧状态更新（b_t=cand_n 匹配后可得；结论供下一帧 stage1 使用）
        if self._gated_sc:
            gtight = self._gated_update(cand_n)
            pcl.log('gated_state', self.frame_id, b_t=int(cand_n),
                    state_out=self._gated['state'], recovery_left=int(self._gated['recovery_left']),
                    tighten=int(gtight))
        if self.trackshield is not None:
            u_detection = self.trackshield.cap_births(detections, u_detection)
        park_all_l1 = l1_iv and (
            self._cg_window_active if self.suspect_mode else l1_alarm) and self.frame_id > 1
        n_parked_l1 = 0
        aux_birth_boxes = None
        if self.aux_review_mode and self.aux_filter_mode and l1_alarm:
            aux_birth = self._aux_candidates()
            aux_birth = aux_birth[aux_birth[:, 4] >= self.aux_conf]
            aux_birth_boxes = aux_birth[:, :4].copy()
            if len(aux_birth_boxes):
                aux_birth_boxes[:, 2:] += aux_birth_boxes[:, :2]
            self.cg_stats['aux_birth_review_frames'] += 1
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                n_skip_birth += 1
                continue
            if aux_birth_boxes is not None:
                support = (_tlbr_iou_matrix(np.asarray([track.tlbr]), aux_birth_boxes).max()
                           if len(aux_birth_boxes) else 0.0)
                if support < self.aux_iou:
                    self.cg_stats['aux_birth_rejected'] += 1
                    pcl.log('aux_birth', self.frame_id, action='rejected',
                            score=float(track.score), max_iou=float(support),
                            tlwh=[float(x) for x in track.tlwh])
                    continue
            park = False
            z = None
            if self.frame_id > 1:  # E4-1 v1.1: 冷启动豁免（frame 1 不隔离）
                if qmode == 'all':
                    park = True
                elif qmode == 'risk':
                    z = self._birth_risk_z(track, cand_n)
                    park = z is not None and z > self._birth_thr()
                elif qmode == 'randq':
                    # E8 S-randq：warmup 后（帧>5）候选，按离线配额+概率+种子随机停放
                    self._randq_init()
                    if self.frame_id > 5:
                        self._randq_seen += 1
                        if self._randq_parked < self._randq_quota and \
                                self._randq_rng.random() < self._randq_p:
                            park = True
                            self._randq_parked += 1
                elif qmode == 'oracle':
                    # E8 Oracle（上限，GT 辅助，仅离线参考）：停放 det 与有效 GT max IoU<0.5 的假出生
                    z = self._oracle_max_iou(track)
                    park = z < 0.5
                elif park_all_l1:
                    park = True
            if park:
                n_parked_l1 += 1 if park_all_l1 else 0
                queue_entry = {'tlbr': track.tlbr.copy(), 'score': float(track.score),
                               'frame': self.frame_id}
                if self.suspect_mode:
                    queue_entry.update(hits=1, last_frame=self.frame_id)
                    self.cg_stats['suspect_parked'] += 1
                self._quarantine.append(queue_entry)
                pcl.log('birth_q', self.frame_id, action='parked', det_idx=int(inew),
                        score=float(track.score), z=z, n_births_same_frame=cand_n,
                        trigger='fixed8' if (park_all_l1 and self.args.l1_mode == 'fixed8') else ('l1' if park_all_l1 else qmode),
                        tlwh=[float(x) for x in track.tlwh])
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            pcl.log('lifecycle', self.frame_id, ev='birth', id=int(track.track_id),
                    score=float(track.score), tlwh=[float(x) for x in track.tlwh],
                    activated=bool(track.is_activated))
            if qmode == 'oracle' and z is not None:
                pcl.log('oracle_birth', self.frame_id, score=float(track.score),
                        max_iou=float(z), park=0, tlwh=[float(x) for x in track.tlwh])
        self.last_g3 = {'b_t': int(cand_n), 'alarm': bool(l1_alarm), 'parked': int(n_parked_l1)}
        if n_skip_birth:
            pcl.log('lifecycle', self.frame_id, ev='birth_skipped_low_score', count=n_skip_birth)
        if qmode == 'randq':
            self._randq_init()
            pcl.log('randq_state', self.frame_id, quota=self._randq_quota, p=self._randq_p,
                    parked=self._randq_parked, seen=self._randq_seen)
        if qmode == 'oracle':
            pcl.log('oracle_gt', self.frame_id,
                    gt=[[float(x) for x in g] for g in (getattr(self.args, 'current_gt', None) or [])])

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                pcl.log('lifecycle', self.frame_id, ev='removed_timeout', id=int(track.track_id))


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        _pre_dup = {t.track_id for t in self.tracked_stracks} | {t.track_id for t in self.lost_stracks}
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        _post_dup = {t.track_id for t in self.tracked_stracks} | {t.track_id for t in self.lost_stracks}
        for _d in (_pre_dup - _post_dup):
            pcl.log('lifecycle', self.frame_id, ev='removed_duplicate', id=int(_d))
        if self.trackshield is not None:
            self.tracked_stracks, self.lost_stracks, ts_removed = self.trackshield.prune(
                self.tracked_stracks, self.lost_stracks)
            self.removed_stracks.extend(ts_removed)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        if self.use_busca:
            self.last_image = np.copy(current_frame)

        return output_stracks

    def is_reliable(self, current_frame, active_stracks, p):
        det_cov = self.get_detection_coverage(
            frame=current_frame, active_stracks=active_stracks, inactive_stracks=[])
        return det_cov['area_covered'] > det_cov['area_covered_per_obj'] * p[0] + p[1]

    def get_extra_kalman_candidates(self, strack_pool, frame_img, det_conf=0.10000001):
        extra_kalman_candidates = []
        for track in strack_pool:
            det_tlwh = track.tlwh
            det_score = np.float32(det_conf)
            det_img = self.busca_tracker.get_image_crops(
                image=frame_img, bboxes=[track.tlbr * track.scale], normalize=False)[0]
            det = STrack(tlwh=det_tlwh, score=det_score, args=self.args,
                         image=det_img, scale=track.scale)
            extra_kalman_candidates.append(det)
        return extra_kalman_candidates

    def third_round_association(self, strack_pool, considered_dets,
                                extra_kalman_candidates, asoc_thresh):
        if asoc_thresh <= 0.0 or len(strack_pool) == 0:
            return [], list(range(len(strack_pool)))

        with torch.no_grad():
            if len(considered_dets) > 0:
                from busca.tracking import center_distance
                tracks_dets_dists = center_distance(strack_pool, considered_dets)
            else:
                tracks_dets_dists = np.zeros((len(strack_pool), 0), dtype=np.float64)
            embedding_probs, reliable_predictions = self.busca_tracker.associate_embeddings(
                tracks_embeddings=strack_pool,
                dets_embeddings=considered_dets,
                dists_matrix=tracks_dets_dists,
                seq_len=self.args.seq_len,
                num_candidates=self.args.num_candidates,
                use_broader_memory=self.args.use_broader_memory,
                extra_kalman_candidates=extra_kalman_candidates,
                select_highest_candidate=self.args.select_highest_candidate,
                highest_candidate_minimum_thresh=getattr(
                    self.args, 'highest_candidate_minimum_thresh', None),
                plot_results=bool(getattr(self.args, 'online_visualization', False)),
                normalize_ims=True)

        if embedding_probs is None:
            return [], list(range(len(strack_pool)))

        num_all_dets = len(considered_dets)
        matches = []
        u_track = []
        for i in range(len(strack_pool)):
            prob = embedding_probs[i, i + num_all_dets]
            if reliable_predictions[i] and prob > asoc_thresh:
                matches.append([i, prob])
            else:
                u_track.append(i)
        return matches, u_track

    def get_detection_coverage(self, frame, active_stracks, inactive_stracks):
        import cv2
        base_frame = np.zeros_like(frame).astype(np.uint8)
        num_objs = 0
        max_bbox_area = 0.0
        bbox_areas = []
        for track in joint_stracks(active_stracks, inactive_stracks):
            target_bbox = np.array(track.tlbr) * track.scale
            cv2.rectangle(base_frame,
                          (int(target_bbox[0]), int(target_bbox[1])),
                          (int(target_bbox[2]), int(target_bbox[3])),
                          (255, 255, 255), thickness=-1)
            num_objs += 1
            bbox_area = max(min(
                ((target_bbox[2] - target_bbox[0]) / base_frame.shape[0]) *
                ((target_bbox[3] - target_bbox[1]) / base_frame.shape[1]), 1.0), 0.0)
            max_bbox_area = max(max_bbox_area, bbox_area)
            bbox_areas.append(bbox_area)

        base_frame = base_frame[:, :, 0]
        num_non_black_pixels = np.count_nonzero(base_frame)
        percentage_covered = num_non_black_pixels / (base_frame.shape[0] * base_frame.shape[1])
        if num_objs > 0:
            avg_area_covered = percentage_covered / num_objs
            average_bbox_area = np.sqrt(np.array(bbox_areas)).mean() ** 2
        else:
            avg_area_covered = 0.0
            average_bbox_area = 0.0
        return {'area_covered': percentage_covered,
                'area_covered_per_obj': avg_area_covered,
                'max_bbox_area': max_bbox_area,
                'average_bbox_area': average_bbox_area,
                'bbox_areas': bbox_areas}

    def camera_motion_compensation(self, track_pool, current_frame,
                                   number_of_iterations=100, termination_eps=0.00001,
                                   warp_mode='MOTION_EUCLIDEAN'):
        import cv2
        cc = 1.0
        if self.frame_id > 1:
            im1_gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
            im2_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            if warp_mode == 'MOTION_EUCLIDEAN':
                warp_mode = cv2.MOTION_EUCLIDEAN
            elif warp_mode == 'MOTION_AFFINE':
                warp_mode = cv2.MOTION_AFFINE
            else:
                raise ValueError('Invalid warp_mode: {}'.format(warp_mode))
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        number_of_iterations, termination_eps)
            cc, warp_matrix = cv2.findTransformECC(
                templateImage=im1_gray, inputImage=im2_gray, warpMatrix=warp_matrix,
                motionType=warp_mode, criteria=criteria)
            warp_matrix = torch.from_numpy(warp_matrix)
            for track in track_pool:
                track.apply_camera_motion(warp_matrix)
        return cc

    @staticmethod
    def warp_pos(pos, warp_matrix):
        p1 = torch.Tensor([pos[0], pos[1], 1]).view(3, 1)
        p1_n = torch.mm(warp_matrix, p1).view(1, 2)
        return p1_n[0]


def _tlbr_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def _tlbr_iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
