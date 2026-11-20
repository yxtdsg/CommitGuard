# -*- coding: utf-8 -*-
"""Final CommitGuard with strict TraSw-tensor auxiliary provenance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from commitguard.tracker import CommitGuardTracker as E19Tracker
from commitguard._vendor.precommit_logger import PreCommitLogger


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class E29Tracker(E19Tracker):
    def __init__(self, args, frame_rate=30):
        if getattr(args, "l1_mode", "off") != "fixed8" or int(args.l1_intervene) != 1:
            raise ValueError("E29 CG 必须使用 fixed8 且 l1_intervene=1")
        args.aux_review_mode = True
        args.aux_filter_mode = True
        args.aux_fn_mode = True
        args.aux_candidate_file = os.environ["E29_AUX_FILE"]
        args.aux_conf = 0.25
        args.aux_iou = 0.40
        args.aux_add_conf = 0.10
        args.aux_track_iou = 0.99
        args.aux_filter_track_iou = 0.99
        args.aux_fn_track_iou = 0.10
        args.aux_recovery = 2
        args.reid_mode = False
        self._e29_expected_kind = os.environ["E29_INPUT_KIND"]
        super().__init__(args, frame_rate)
        PreCommitLogger.get().log(
            "e29_tracker_config", 1, tracker="E19Tracker/e19_fi40",
            aux_candidate_file=self.aux_candidate_file,
            aux_candidate_sha256=_sha256(self.aux_candidate_file),
            input_kind=self._e29_expected_kind,
            aux_conf=self.aux_conf, aux_iou=self.aux_iou,
            aux_add_conf=self.aux_add_conf, aux_track_iou=self.aux_track_iou,
            aux_filter_track_iou=self.aux_filter_track_iou,
            aux_fn_track_iou=self.aux_fn_track_iou,
            aux_recovery=self.aux_recovery, l1_mode=args.l1_mode,
            l1_intervene=int(args.l1_intervene))

    def _load_aux_map(self):
        self._e29_aux_map = {}
        with open(self.aux_candidate_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                frame = int(rec["f"])
                if not 6 <= frame <= 30:
                    raise ValueError("E29 aux 只允许帧 6..30")
                if rec.get("input_kind") != self._e29_expected_kind:
                    raise ValueError("E29 aux input_kind 与当前臂输入不符")
                key = (rec["seg"], frame)
                if key in self._e29_aux_map:
                    raise ValueError("E29 aux 重复记录: " + str(key))
                np.asarray(rec["dets"], dtype=np.float64).reshape(-1, 5)
                self._e29_aux_map[key] = rec

    def _aux_candidates(self):
        frame = self.frame_id
        if not 6 <= frame <= 30:
            raise RuntimeError("E29 不允许在帧 1..5 或段外读取 aux")
        if not hasattr(self, "_e29_aux_map"):
            self._load_aux_map()
        seg = getattr(self.args, "current_seg", "")
        key = (seg, frame)
        if key not in self._e29_aux_map:
            raise KeyError("E29 缺少 aux 记录: " + str(key))
        rec = self._e29_aux_map[key]
        replay_seg = self.args.replay_seg
        if replay_seg != seg or rec["seg"] != replay_seg:
            raise RuntimeError("E29 aux/replay 显式段名不一致")
        seg_root = Path(self.args.replay_adv_dir) / replay_seg
        with open(seg_root / "frame_hashes.json", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["seg"] != replay_seg or manifest["n_frames"] != 30:
            raise RuntimeError("E29 aux replay manifest 不匹配")
        expected = manifest["frames"][str(frame)]
        path = seg_root / ("frame_%d.npy" % frame)
        if _sha256(path) != expected["file_sha256"] or \
                rec["input_file_sha256"] != expected["file_sha256"]:
            raise RuntimeError("E29 aux 输入文件哈希不匹配")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (3, 800, 1440) or \
                not np.isfinite(array).all():
            raise RuntimeError("E29 aux replay 张量非法")
        actual_data_hash = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        if actual_data_hash != expected["data_sha256"] or \
                rec["input_data_sha256"] != actual_data_hash:
            raise RuntimeError("E29 aux 同帧 data_sha256 不匹配")
        PreCommitLogger.get().log(
            "e29_aux_verified", frame, replay_seg=replay_seg,
            input_kind=rec["input_kind"], input_data_sha256=actual_data_hash,
            input_file_sha256=expected["file_sha256"], aux_n=len(rec["dets"]))
        return np.asarray(rec["dets"], dtype=np.float64).reshape(-1, 5)

    def update(self, output_results, img_info, img_size, current_frame=None):
        output = super().update(output_results, img_info, img_size, current_frame)
        stats = {k: v for k, v in self.cg_stats.items()
                 if isinstance(v, (int, float, bool))}
        PreCommitLogger.get().log(
            "e29_cg_state", self.frame_id, state_out=self._l1["state"],
            alarm=bool(self.last_g3["alarm"]), parked=int(self.last_g3["parked"]),
            b_t=int(self.last_g3["b_t"]),
            window_active=bool(self._cg_window_active),
            assoc_window_active=bool(self._cg_assoc_window_active),
            aux_review_active=bool(self._aux_review_active), **stats)
        return output
