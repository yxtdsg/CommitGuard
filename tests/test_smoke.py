import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from commitguard.runner import run
from commitguard._vendor.matching import ious


class CommitGuardSmokeTest(unittest.TestCase):
    def test_compiled_iou(self):
        result = ious(np.array([[0.0, 0.0, 10.0, 10.0]]),
                      np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]]))
        np.testing.assert_allclose(result, [[1.0, 36.0 / 206.0]])

    def test_frozen_detection_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.json"
            detections = root / "detections.jsonl"
            auxiliary = root / "auxiliary.jsonl"
            methods = root / "methods.json"
            protocol = root / "protocol.json"
            images = [{"id": frame, "file_name": f"synth/{frame:06d}.jpg",
                       "width": 1440, "height": 800} for frame in range(1, 31)]
            annotation.write_text(json.dumps({"video_name": "synth", "images": images}),
                                  encoding="utf-8")
            with open(detections, "w", encoding="utf-8") as handle:
                for frame in range(1, 31):
                    dets = [[100.0, 100.0, 50.0, 50.0, 0.9]]
                    if frame == 6:
                        dets += [[200.0 + 100 * index, 300.0, 40.0, 40.0, 0.9]
                                 for index in range(8)]
                    handle.write(json.dumps({"seg": "synth-000", "f": frame, "dets": dets}) + "\n")
            with open(auxiliary, "w", encoding="utf-8") as handle:
                for frame in range(6, 31):
                    record = {"seg": "synth-000", "f": frame,
                              "input_kind": "adversarial" if frame in (6, 7, 8) else "clean_recovery",
                              "dets": [[100.0, 100.0, 50.0, 50.0, 0.9]]}
                    handle.write(json.dumps(record) + "\n")
            methods.write_text(json.dumps({
                "c0": {"base": "c0", "args": {}},
                "audit": {"base": "g3", "args": {"l1_intervene": 0}},
                "e19_fi40": {"base": "g3", "args": {
                    "aux_review_mode": True, "aux_filter_mode": True, "aux_fn_mode": True,
                    "aux_conf": 0.25, "aux_iou": 0.4, "aux_add_conf": 0.1,
                    "aux_track_iou": 0.99, "aux_filter_track_iou": 0.99,
                    "aux_fn_track_iou": 0.1, "aux_recovery": 2}}}), encoding="utf-8")
            protocol.write_text("{}", encoding="utf-8")

            outputs = {}
            for method in ("c0", "audit", "e19_fi40"):
                metadata = run(method, "synth", 1, "A", root / "runs", annotation,
                               detections, methods, protocol,
                               str(auxiliary) if method == "e19_fi40" else "")
                outputs[method] = metadata.parent
            self.assertEqual((outputs["c0"] / "synth-000.txt").read_bytes(),
                             (outputs["audit"] / "synth-000.txt").read_bytes())
            runtime = json.loads((outputs["e19_fi40"] / "runtime.json").read_text(encoding="utf-8"))
            stats = runtime["segments"][0]["cg_stats"]
            self.assertGreater(stats["trigger_frames"], 0)
            self.assertGreater(stats["aux_review_frames"], 0)
            self.assertEqual(runtime["frames"][0]["seg"], "synth-000")


if __name__ == "__main__":
    unittest.main()
