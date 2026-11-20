# -*- coding: utf-8 -*-
"""Runtime-only tracker selection for the frozen E14 victim harness."""
import argparse
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
FNF = Path(os.environ.get("FNF_ROOT", REPO / "external" / "FnF")).resolve()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(FNF))
parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True,
                    choices=("clean_c0", "attack_c0", "clean_cg", "attack_cg"))
parser.add_argument("--aux-file", required=True)
parser.add_argument("official", nargs=argparse.REMAINDER)
wrapper = parser.parse_args()
official = list(wrapper.official)
if official and official[0] == "--":
    official = official[1:]
if not official:
    raise ValueError("缺少 E14 victim 官方参数")
os.environ["E29_AUX_FILE"] = str(Path(wrapper.aux_file).resolve())
os.environ["E29_INPUT_KIND"] = (
    "trasw_clean_tensor" if wrapper.arm.startswith("clean") else "trasw_attack_tensor")
import yolox.core.joint_attacker_and_tracker as victim_core
if wrapper.arm.endswith("_cg"):
    from yolox.tracker.precommit_logger import PreCommitLogger as ExternalLogger
    from commitguard._vendor.precommit_logger import PreCommitLogger as PublicLogger
    PublicLogger._inst = ExternalLogger.get()
    from tracker import E29Tracker
    victim_core.BYTETracker = E29Tracker
else:
    from yolox.tracker.byte_tracker import BYTETracker as NativeBYTETracker
    victim_core.BYTETracker = NativeBYTETracker
_driver = FNF / "tools" / "joint_attack_and_track.py"
sys.argv = [str(_driver)] + official
runpy.run_path(str(_driver), run_name="__main__")
