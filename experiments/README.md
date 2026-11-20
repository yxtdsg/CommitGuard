# Experiment input and execution contract

All machine-specific locations are supplied by CLI arguments or environment variables. Outputs should go under the ignored `runs/` directory or another external directory.

## Frozen detection schema

Each JSONL record represents one local frame:

```json
{"seg": "MOT17-02-FRCNN-000", "f": 1, "dets": [[100.0, 50.0, 40.0, 80.0, 0.9]]}
```

`dets` contains `[x, y, width, height, score]` rows in the frozen model-input coordinate system. Every segment has 30 records. No detector is run inside the replay tracker.

Auxiliary candidate JSONL uses the same fields plus `input_kind`:

- `adversarial` on local frames 6–8 for condition `A`;
- `clean_reference` on frames 6–8 for clean runs;
- `clean_recovery` on later frames.

Annotations are COCO-style dictionaries with sorted `images`, and optionally `video_name`. Ground truth follows the MOTChallenge text format.

## Directory templates

Each public protocol defines `paths`. The root passed as `--input-root` is prepended to these relative paths.

| Experiment | Annotation | Detection | Auxiliary candidates |
|---|---|---|---|
| MOT17 development | `annotations/mot17_development/{annotation}` | `detections/mot17_development/det_input_{condition}_{video}_r{rep}.jsonl` | `auxiliary/mot17_development/aux_yolov7_1280_{condition}_all.jsonl` |
| MOT17 additional | `annotations/mot17_additional/{annotation}` | `detections/mot17_additional/det_input_{condition}_{video}_r{rep}.jsonl` | `auxiliary/mot17_additional/aux_yolov7_1280_{condition}_all.jsonl` |
| DanceTrack | `annotations/dancetrack_main/{annotation}` | `detections/dancetrack_main/det_input_{condition}_{video}_r{rep}.jsonl` | `auxiliary/dancetrack_main/aux_yolov7_1280_A_all.jsonl` |
| OC-SORT recovery | `annotations/ocsort_recovery/{annotation}` | `detections/ocsort_recovery/det_input_{condition}_{video}_r{rep}.jsonl` | `auxiliary/ocsort_recovery/aux_yolov7_1280_{condition}_all.jsonl` for CG |
| Method comparison | `annotations/clean2_mot17_{video}.json` | `detections/det_input_A_{video}_r{rep}.jsonl` | `auxiliary/aux_yolov7_1280_A_all.jsonl` |

The exact annotation filenames are in each protocol's `annotations` map. GT paths are `gt/mot17/MOT17-{video}-FRCNN/gt/gt.txt` for MOT17 and `gt/dancetrack/dancetrack{video}/gt/gt.txt` for DanceTrack.

BUSCA also needs `images/`; BoT-SORT no-ReID needs `images/`, `tensors/mot17_additional/`, and `tensors/mot17_additional/frozen_tensor_hashes.json`. These files remain external.

## Frozen final method

The public method name is `e19_fi40`. Across E19, E20 and E26 its CG settings are:

- auxiliary review/filter/FN enabled;
- `aux_conf=0.25`, `aux_iou=0.4`, `aux_add_conf=0.1`;
- `aux_track_iou=0.99`, `aux_filter_track_iou=0.99`;
- `aux_fn_track_iou=0.1`, `aux_recovery=2`.

E19 additionally retains the filter-only, FN-only, IoU=0.3/0.5 and add-confidence=0.25 ablations. E20 has exactly two available repetitions. E26 uses four segments per video and only videos 0004/0005/0007/0010.

## Outputs

Each run writes:

- `<run_name>/<segment>.txt` in MOT format;
- `<run_name>/runtime.json` with input/source hashes, frame timing and CG counters.

Evaluation writes segment, video and macro tables. The public evaluator performs per-segment evaluation and then the same arithmetic aggregation as the frozen protocol. Timing values are environment-dependent and are not expected to match historical timing byte-for-byte.

## E24

The E24 matrix contains clean C0/OC-SORT plus C0/CG/OC-SORT under B, A and C. These condition labels are frozen historical arm names. The evaluator checks 187 runs, 1,012 window rows, 77 identity-video rows and 22 identity-macro rows before writing outputs.

## E29 method comparison

The eight methods are C0, TrackShield-a1, TrackShield-a2, BUSCA, OC-SORT, BoT-SORT no-ReID, Audit+BG, and CG. All 96 runs are regenerated under one output root; no old E20/E24/E25 run directory is reused.

## E29 TraSw

`freeze.json` contains only the sanitized ten-segment protocol and hashes. `install_overlays.py` verifies or installs the seven required third-party source overlays. The execution order is attack → auxiliary candidates → victims → summary → TrackEval/identity analysis. Completion records and replay hashes are validated; the pipeline does not retry until a preferred result appears.

See the root README for the explicit historical FnF source-version limitation.
