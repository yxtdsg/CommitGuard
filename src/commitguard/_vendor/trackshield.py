import json
import math
from collections import deque
from pathlib import Path

import numpy as np


class TrackShieldController:
    """TrackShield v1 algorithm adapted to the F&F ByteTrack lifecycle.

    The monitor, L1 birth cap, L2 pruning, and L3 association caps follow the
    published rules. Warm mode seeds only from disjoint same-video clean prefixes.
    """

    SIGNALS = ("M", "dM", "U", "W", "RT", "RD")
    STAGES = ("stage1", "stage2", "stage3")

    def __init__(self, tracker):
        self.tracker = tracker
        self.args = tracker.args
        self.mode = self.args.trackshield_mode
        self.segment = self.args.current_seg
        self.video = self.segment.split("-")[1]
        self.calibration_key = getattr(self.args, 'trackshield_calibration_key', self.video)
        self.tau = 6.0
        self.alpha = 0.02
        self.bmax = 15
        self.l_init = 20
        self.window = 5
        self.q = 3
        self.lam = 0.05
        self.votes = deque(maxlen=self.window)
        self.caps = {}
        self.hist = {k: deque(maxlen=120) for k in self.SIGNALS}
        self.mu = {k: None for k in self.SIGNALS}
        self.n = {k: 0 for k in self.SIGNALS}
        self.level = 0
        self.score = 0.0
        self.cap_triggered = False
        self.stage_ratios = []
        self.m_previous = None
        self.log_path = Path(self.args.external_log_dir) / f"{self.segment}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_calibration()

    def _load_calibration(self):
        if self.mode == "audit":
            return
        path = Path(self.args.trackshield_calibration)
        if not path.exists():
            raise FileNotFoundError(f"TrackShield calibration missing: {path}")
        data = json.load(open(path, encoding="utf-8"))[self.calibration_key]
        self.caps = data["caps"]
        if self.mode == "ts_warm":
            for signal in self.SIGNALS:
                item = data["monitor"][signal]
                self.hist[signal] = deque(item["history"], maxlen=120)
                self.mu[signal] = None if item["mu"] is None else float(item["mu"])
                self.n[signal] = int(item["n"])

    def log(self, **record):
        record.update({"seg": self.segment, "f": self.tracker.frame_id})
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def begin_frame(self):
        self.cap_triggered = False
        self.stage_ratios = []
        self.match_counts = {}
        self.excluded_tracks = {}
        self.m_start = len(self.tracker.tracked_stracks) + len(self.tracker.lost_stracks)
        self.delta_m = 0 if self.m_previous is None else self.m_start - self.m_previous
        self.m_previous = self.m_start

    @staticmethod
    def _track_order(track):
        return (-int(track.tracklet_len), int(track.start_frame), -float(track.score), int(track.track_id))

    def cap_association(self, stage, tracks, detections):
        original_tracks = list(tracks)
        tracks = list(original_tracks)
        detections = list(detections)
        if self.mode == "audit":
            kt = kd = 10 ** 9
        else:
            cfg = self.caps[stage]
            kt = max(1, int(cfg["KT"]))
            kd = max(1, int(cfg["KD"]))
        nt, nd = len(tracks), len(detections)
        capped = nt > kt or nd > kd or nt * nd > kt * kd
        if capped:
            self.cap_triggered = True
            track_indices = sorted(range(nt), key=lambda i: self._track_order(tracks[i]))[:kt]
            det_indices = sorted(range(nd), key=lambda i: (-float(detections[i].score), i))[:kd]
            tracks = [tracks[i] for i in track_indices]
            detections = [detections[i] for i in det_indices]
        admitted = {id(track) for track in tracks}
        self.excluded_tracks[stage] = [track for track in original_tracks if id(track) not in admitted]
        self.stage_ratios.append((stage, nt, nd, len(tracks), len(detections)))
        self.log(event="L3", stage=stage, tracks_before=nt, dets_before=nd,
                 tracks_after=len(tracks), dets_after=len(detections), KT=kt, KD=kd,
                 triggered=capped)
        return tracks, detections

    def record_matches(self, stage, n_matches):
        self.match_counts[stage] = int(n_matches)

    @staticmethod
    def _scale(history, signal):
        if not history:
            return 1.0 if signal in ("M", "dM", "U", "W") else 0.02
        med = float(np.median(history))
        mad = float(np.median(np.abs(np.asarray(history) - med)))
        floor = 0.02 if signal in ("RT", "RD") else 1.0
        return max(1.4826 * mad + 0.05 * abs(med), floor) + 1e-6

    def after_association(self, unmatched_high, n_detections):
        valid_rt = valid_rd = True
        rt_values = []
        rd_values = []
        for stage, _, _, kept_tracks, kept_dets in self.stage_ratios:
            if kept_tracks > 0 and kept_dets > 0:
                n_matches = self.match_counts.get(stage, 0)
                rt_values.append(n_matches / float(kept_tracks))
                rd_values.append(n_matches / float(kept_dets))
            else:
                valid_rt = valid_rd = False
        rt = float(np.mean(rt_values)) if valid_rt and rt_values else 1.0
        rd = float(np.mean(rd_values)) if valid_rd and rd_values else 1.0
        values = {
            "M": float(self.m_start),
            "dM": float(self.delta_m),
            "U": float(unmatched_high),
            "W": float(self.m_start * n_detections),
            "RT": float(rt),
            "RD": float(rd),
        }
        z_values = {}
        directed = {}
        for signal in self.SIGNALS:
            if self.n[signal] < self.l_init or self.mu[signal] is None:
                z = 0.0
            else:
                z = float(np.clip((values[signal] - self.mu[signal]) / self._scale(self.hist[signal], signal), -12, 12))
            if (signal == "RT" and not valid_rt) or (signal == "RD" and not valid_rd):
                z = 0.0
            z_values[signal] = z
            if signal in ("RT", "RD"):
                directed[signal] = max(-z, 0.0)
            else:
                directed[signal] = max(z, 0.0)
        raw_score = sum(directed[s] for s in ("M", "dM", "U", "W", "RT")) + 0.5 * directed["RD"]
        warmup = self.mode == "ts_warm" and self.tracker.frame_id <= 5
        self.score = 0.0 if warmup else float(raw_score)
        vote = self.score > self.tau
        self.votes.append(int(vote))
        if not warmup and len(self.votes) == self.window and sum(self.votes) >= self.q:
            self.level = 2
        elif vote:
            self.level = 1
        else:
            self.level = 0
        self.log(event="monitor", M=values["M"], dM=values["dM"], U=values["U"], W=values["W"],
                 RT=rt, RD=rd, z=z_values, score=self.score, raw_score=float(raw_score), vote=vote,
                 warmup=warmup, votes_recent=sum(self.votes), level=self.level, cap_any=self.cap_triggered)

        if warmup or (self.score <= self.tau and not self.cap_triggered):
            for signal in self.SIGNALS:
                if (signal == "RT" and not valid_rt) or (signal == "RD" and not valid_rd):
                    continue
                value = values[signal]
                if self.mu[signal] is None:
                    self.mu[signal] = value
                else:
                    self.mu[signal] = self.alpha * value + (1.0 - self.alpha) * self.mu[signal]
                self.hist[signal].append(value)
                self.n[signal] += 1

    def cap_births(self, detections, unmatched_indices):
        if self.mode not in ("ts_cold", "ts_warm"):
            return list(unmatched_indices)
        if self.level < 1 or len(unmatched_indices) <= self.bmax:
            return list(unmatched_indices)
        ranked = sorted(unmatched_indices, key=lambda i: (-float(detections[i].score), i))
        kept = ranked[:self.bmax]
        self.log(event="L1", before=len(unmatched_indices), after=len(kept), Bmax=self.bmax)
        return kept

    def prune(self, tracked, lost):
        if self.mode not in ("ts_cold", "ts_warm") or self.level != 2:
            return tracked, lost, []
        removed = []

        def risky(track):
            age = self.tracker.frame_id - int(track.start_frame) + 1
            hits = int(track.tracklet_len)
            conf = float(track.score)
            votes = int(age < 3) + int(hits < 2) + int(conf < 0.3)
            return votes >= 2, {"age": age, "hits": hits, "conf": conf, "P": votes}

        kept_tracked = []
        for track in tracked:
            remove, detail = risky(track)
            (removed if remove else kept_tracked).append(track)
            if remove:
                self.log(event="L2", track_id=int(track.track_id), pool="tracked", **detail)
        kept_lost = []
        for track in lost:
            remove, detail = risky(track)
            (removed if remove else kept_lost).append(track)
            if remove:
                self.log(event="L2", track_id=int(track.track_id), pool="lost", **detail)
        for track in removed:
            track.mark_removed()
        return kept_tracked, kept_lost, removed
