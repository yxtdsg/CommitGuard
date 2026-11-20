# Third-party notices

The MIT `LICENSE` applies to CommitGuard's own code. Vendored or overlaid third-party files remain under their original licenses. External dependencies are not relicensed by this project.

## Vendored or overlaid source

### FnF / ByteTrack / YOLOX

- Project page: https://infzhou.github.io/FnF.html
- Source fork used by the experiments: https://github.com/yxtdsg/FnF_Attack
- Recorded base commit: `bec6398bd9d1a784e7044e4419251cdfc28a1dad`
- ByteTrack upstream: https://github.com/ifzhang/ByteTrack
- YOLOX upstream: https://github.com/Megvii-BaseDetection/YOLOX
- License: MIT; see `licenses/ByteTrack-FnF-MIT.txt`.
- Locations: `src/commitguard/_vendor/{byte_tracker,matching,kalman_filter,basetrack,precommit_logger}.py` and `experiments/trasw/overlays/fnf/`.
- Changes: package-relative imports in the public core; the original compiled `cython_bbox` association path is preserved. Overlay files are byte-pinned by `experiments/trasw/protocol.json`.

### TraSw / ByteTrack-attack

- Upstream: https://github.com/DerryHub/ByteTrack-attack
- License: MIT; see `licenses/TraSw-MIT.txt`.
- Locations: `experiments/trasw/overlays/trasa/`.
- Only the small files required for frozen tensor export are overlaid. No upstream repository, frames, weights or tensors are bundled.

### TrackShield adaptation

- Source: the CommitGuard project's E16 comparison controller.
- Location: `src/commitguard/_vendor/trackshield.py`.
- Released under this project's MIT license. The frozen comparison calibration is `experiments/method_comparison/calibration.json`.

## External trackers and evaluation

### OC-SORT

- Upstream: https://github.com/noahcao/OC_SORT
- E24 protocol pin: `5d9b940e259ecc801156b4b05b378ec1fc4ad04d`
- License: MIT; reference copy in `licenses/OC-SORT-MIT.txt`.
- Used through an external adapter; the upstream repository is not bundled.

### BoT-SORT

- Upstream: https://github.com/NirAcker/BoT-SORT
- Recorded commit: `2519852d398900f014991430ef800de393c9d93b`
- License: MIT; reference copy in `licenses/BoT-SORT-MIT.txt`.
- The comparison is explicitly no-ReID. An external FnF-compatible BoT-SORT checkout is required; no BoT-SORT or fast-reid source is bundled.

### BUSCA

- Upstream: https://github.com/chipay/BUSCA
- Recorded commit: `88a9e448f9d69c68b2e9cc9fc51025e60ebcffda`
- License: GNU GPLv3.
- External dependency only. No BUSCA source or weights are redistributed. Running that baseline requires `BUSCA_ROOT` and its official MOT17 configuration and weights.

### TrackEval

- Package/version: `trackeval==1.3.0`
- Repository: https://github.com/kovalp/TrackEval
- Original project: https://github.com/JonathonLuiten/TrackEval
- License: MIT; reference copy in `licenses/TrackEval-MIT.txt`.
- Installed as an external package, not vendored.

## Detector and weights

### YOLOv7

- Upstream: https://github.com/WongKinYiu/yolov7
- License: GNU GPLv3.
- External code and weights only. No YOLOv7 source or weights are bundled.
- Frozen `yolov7.pt` SHA256: `d1ac3c74eb96a3eec77949c0f37a06bc272756606ff011d9353f7abff4e0c71d`.

### ByteTrack weights

- Distributed by the respective ByteTrack/FnF/TraSw upstream projects, not by this repository.
- Frozen `bytetrack_x_mot17.pth.tar` SHA256: `e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba`.

FTHA source with unclear redistribution terms is not included. Only the documented external YOLOv7 dependency and weight hash are retained.

## Data

MOT17 and DanceTrack remain subject to their respective dataset terms. This repository contains neither dataset images nor ground-truth files. Curated CSV/JSON summaries and PDF/PNG figures are included for result validation; raw tracks, model-input tensors, attack tensors, detection caches and large logs are excluded.
