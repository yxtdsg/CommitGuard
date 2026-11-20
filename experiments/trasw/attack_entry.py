# -*- coding: utf-8 -*-
"""Run frozen official TraSw without editing its source tree."""
import builtins
import io
import os
from pathlib import Path
import runpy
import sys
import numpy as np

sys.dont_write_bytecode = True
TRASA = Path(os.environ.get("TRASA_ROOT", Path(__file__).resolve().parents[2] / "external" / "TraSA")).resolve()
sys.path.insert(0, str(TRASA))
from yolox.tracker.byte_tracker import BYTETracker

_original_recover_noise = BYTETracker.recoverNoise


def _e29_recover_noise(self, noise, img0):
    if isinstance(noise, np.ndarray):
        return noise
    return _original_recover_noise(self, noise, img0)


BYTETracker.recoverNoise = _e29_recover_noise
_original_open = builtins.open
_original_io_open = io.open
_target = (TRASA / "datasets" / "mot" / "annotations" / "_e5_2_seg.json").resolve()
_redirect = Path(os.environ["E29_TRASA_SEG_ANN"]).resolve()
_redirect.parent.mkdir(parents=True, exist_ok=True)


def _redirected_open(file, *args, **kwargs):
    try:
        is_target = Path(file).resolve() == _target
    except TypeError:
        is_target = False
    return _original_open(_redirect if is_target else file, *args, **kwargs)


def _redirected_io_open(file, *args, **kwargs):
    try:
        is_target = Path(file).resolve() == _target
    except TypeError:
        is_target = False
    return _original_io_open(_redirect if is_target else file, *args, **kwargs)


builtins.open = _redirected_open
io.open = _redirected_io_open
_driver = Path(__file__).resolve().parent / "attack_driver.py"
sys.argv[0] = str(_driver)
runpy.run_path(str(_driver), run_name="__main__")
