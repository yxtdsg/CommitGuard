import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class FrozenProtocolTest(unittest.TestCase):
    def load(self, experiment, filename):
        with open(REPO / "experiments" / experiment / filename, encoding="utf-8") as handle:
            return json.load(handle)

    def test_final_thresholds_are_frozen(self):
        for experiment in ("mot17_development", "mot17_additional", "dancetrack_main"):
            args = self.load(experiment, "methods.json")["e19_fi40"]["args"]
            self.assertEqual(args["aux_conf"], 0.25)
            self.assertEqual(args["aux_iou"], 0.4)
            self.assertEqual(args["aux_add_conf"], 0.1)
            self.assertEqual(args["aux_filter_track_iou"], 0.99)
            self.assertEqual(args["aux_fn_track_iou"], 0.1)
            self.assertEqual(args["aux_recovery"], 2)

    def test_formal_run_counts(self):
        development = self.load("mot17_development", "protocol.json")
        additional = self.load("mot17_additional", "protocol.json")
        dance = self.load("dancetrack_main", "protocol.json")
        comparison = self.load("method_comparison", "protocol.json")
        recovery = self.load("ocsort_recovery", "protocol.json")
        with open(REPO / "experiments" / "trasw" / "freeze.json", encoding="utf-8") as handle:
            trasw = json.load(handle)
        self.assertEqual(len(development["groups"]) * len(development["videos"]) * len(development["repeats"]), 144)
        self.assertEqual(len(additional["groups"]) * len(additional["videos"]) * len(additional["repeats"]), 48)
        self.assertEqual(len(dance["groups"]) * len(dance["videos"]) * len(dance["repeats"]), 60)
        self.assertEqual(len(comparison["methods"]) * len(comparison["videos"]) * len(comparison["repeats"]), 96)
        recovery_runs = sum((2 if arm == "clean" else 3) * len(scope["videos"]) * len(scope["repeats"])
                            for scope in recovery["scopes"].values() for arm in recovery["conditions"])
        self.assertEqual(recovery_runs, 187)
        self.assertEqual(len(trasw["segments"]) * 4 * 3, 120)

    def test_configs_have_no_private_absolute_paths(self):
        for path in (REPO / "experiments").rglob("*.json"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("e:\\sensor", text, path)
            self.assertNotIn("c:\\users\\yxt", text, path)
            self.assertNotIn("e:\\motattack", text, path)

    def test_formal_video_scopes(self):
        development = self.load("mot17_development", "protocol.json")
        additional = self.load("mot17_additional", "protocol.json")
        dance = self.load("dancetrack_main", "protocol.json")
        self.assertEqual(development["videos"], ["02", "04", "05"])
        self.assertEqual(development["repeats"], [1, 2, 3])
        self.assertEqual(additional["videos"], ["09", "10", "11", "13"])
        self.assertEqual(additional["repeats"], [1, 2])
        self.assertEqual(dance["videos"], ["0004", "0005", "0007", "0010"])
        self.assertEqual(set(dance["segments"].values()), {4})
        self.assertNotIn("0014", dance["videos"])


if __name__ == "__main__":
    unittest.main()
