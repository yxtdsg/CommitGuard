# CommitGuard

**Pre-Commit Birth Review for Robust Multi-Object Tracking**

CommitGuard adds a causal review stage before birth decisions in ByteTrack. The public implementation contains the frozen **Audit → BG → CG** pipeline and the final `e19_fi40` method. It uses no ground-truth intervention, ReID, Vault, or future-frame information.

This repository is a curated release of the final algorithm and paper experiments. It is not a dump of the development workspace.

## What is included

- Portable tracker and frozen-detection replay runner under `src/commitguard/`.
- Formal experiment configurations and run/evaluation entry points.
- Final analysis CSV/JSON files, paper PDF/PNG figures, and their plotting scripts.
- SHA256 provenance in `results/paper/manifest.json` and `results/supplementary/manifest.json`.
- Synthetic smoke tests and result-aggregation checks.

Raw datasets, model weights, attack tensors, per-frame tracks, detection caches and large logs are **not** bundled.

## Formal experiment map

| Public name | Historical ID | Scope | Runs |
|---|---:|---|---:|
| `mot17_development` | E19 | MOT17 02/04/05, final method and ablations | 144 |
| `mot17_additional` | E20 | MOT17 09/10/11/13, 2 repetitions | 48 |
| `ocsort_recovery` | E24 | C0/CG/OC-SORT window and identity recovery | 187 |
| `dancetrack_main` | E26 | DanceTrack 0004/0005/0007/0010 only | 60 |
| `method_comparison` | E29 | Eight methods, four videos, three repetitions | 96 |
| `trasw` | E29 | Complete TraSw, ten segments, four arms, three repetitions | 120 |

E30 sensitivity analyses, E31 clean timing and E32 DanceTrack-0014 visualization are supplementary. **E32 is never included in E26 statistics.**

## Installation

The formal replay environment is pinned to **Python 3.10**. In particular, the original compiled `cython_bbox` association is required: replacing it with a NumPy IoU can change crowded-scene associations.

```bash
conda create -n commitguard python=3.10
conda activate commitguard
pip install -r requirements.txt
```

For a smaller package install, use:

```bash
pip install -e ".[eval,plot,baselines]"
```

Choose the appropriate CUDA build of PyTorch for your machine. The tracker replay itself does not require a GPU; auxiliary YOLOv7 generation and external baselines do.

## Validate the bundled results — no raw data required

```bash
python -m unittest discover -s tests -v
python scripts/validate_results.py
python scripts/check_release.py
```

The validator checks file hashes, formal video scopes, run counts, Audit/C0 equivalence metadata, and macro-table aggregation. It also checks that TraSw's reported conclusion remains unchanged: overall IDF1/HOTA defense is **not supported** by that result set.

Regenerate the quantitative and principle figures from the public tables:

```bash
python scripts/make_result_figs.py
python scripts/make_principle_figs.py
python scripts/build_result_manifest.py
python scripts/validate_results.py
```

Trajectory renderers are `scripts/plot_mot17_three_way.py` and `scripts/plot_dancetrack_three_way.py`; they require external raw tracks and frames.

## Replay a formal experiment from frozen inputs

Frozen inputs follow the relative templates in each experiment's `protocol.json`. See [experiments/README.md](experiments/README.md) for schemas and the complete input contract.

Example: validate inputs and run one MOT17 segment.

```bash
python scripts/run_experiment.py mot17_development prepare \
  --input-root /path/to/external_inputs \
  --output-root runs/mot17_development

python scripts/run_experiment.py mot17_development run \
  --input-root /path/to/external_inputs \
  --output-root runs/mot17_development \
  --video 02 --rep 1 --method e19_fi40 --condition A \
  --only-segment MOT17-02-FRCNN-000
```

Evaluate a completed matrix with TrackEval 1.3.0:

```bash
python scripts/evaluate_experiment.py mot17_development \
  --input-root /path/to/external_inputs \
  --runs-root runs/mot17_development \
  --output-dir runs/mot17_development/analysis
```

The same three-stage interface applies to `mot17_additional` and `dancetrack_main`. `prepare` verifies and hashes existing frozen inputs; it does not fabricate missing attacks or detections.

### OC-SORT recovery

OC-SORT is an external fixed-version dependency.

```bash
python experiments/ocsort_recovery/run.py prepare \
  --input-root /path/to/external_inputs
python experiments/ocsort_recovery/run.py run \
  --input-root /path/to/external_inputs \
  --output-root runs/ocsort_recovery/results \
  --ocsort-root /path/to/OC_SORT
python experiments/ocsort_recovery/evaluate_recovery.py \
  --runs-root runs/ocsort_recovery/results \
  --gt-root /path/to/external_inputs/gt/mot17 \
  --output-dir runs/ocsort_recovery/analysis
```

Rebuild the full-sequence OC-SORT macro table from its video rows:

```bash
python experiments/ocsort_recovery/summarize_full.py \
  --video-summary results/paper/ocsort_recovery/ocsort_full_video_summary.csv \
  --output runs/ocsort_recovery/ocsort_full_macro_summary.csv
```

### Eight-method comparison

```bash
python experiments/method_comparison/run_all.py \
  --input-root /path/to/method_comparison_inputs \
  --output-root runs/method_comparison \
  --ocsort-root /path/to/OC_SORT \
  --bot-sort-root /path/to/BoT-SORT \
  --dry-run
```

Remove `--dry-run` to execute serially. BUSCA additionally requires `BUSCA_ROOT`; TrackShield uses the included frozen calibration. After completion:

```bash
python experiments/method_comparison/evaluate.py \
  --input-root /path/to/method_comparison_inputs \
  --manifest runs/method_comparison/run_manifest.json \
  --output-dir runs/method_comparison/analysis
```

## Complete TraSw execution

TraSw needs external FnF and TraSw checkouts, frozen annotations, exact ByteTrack/YOLOv7 weights, and GPU dependencies.

1. Obtain the upstreams listed in [NOTICE.md](NOTICE.md).
2. Review the seven small source overlays:

   ```bash
   python experiments/trasw/install_overlays.py \
     --fnf-root /path/to/FnF --trasa-root /path/to/TraSA
   ```

3. Install missing overlays explicitly; `--replace` is required before differing files are overwritten:

   ```bash
   python experiments/trasw/install_overlays.py \
     --fnf-root /path/to/FnF --trasa-root /path/to/TraSA --execute --replace
   ```

4. Set `FNF_ROOT`, `TRASA_ROOT`, `YOLOV7_ROOT`, and optionally `TRASW_PYTHON`, `COMMITGUARD_INPUT_ROOT`, and `COMMITGUARD_TRASW_RUNS`.
5. Put the ten annotations at `annotations/trasw/<seg_name>.json` under the input root and GT files under `gt/mot17/MOT17-XX-FRCNN/gt/gt.txt`.
6. Run the staged serial pipeline:

   ```bash
   python experiments/trasw/run.py check
   python experiments/trasw/run.py attack
   python experiments/trasw/run.py aux
   python experiments/trasw/run.py victims
   python experiments/trasw/run.py summarize
   python experiments/trasw/evaluate.py \
     --runs-root runs/trasw \
     --input-root /path/to/external_inputs \
     --gt-root /path/to/external_inputs/gt/mot17 \
     --output-dir runs/trasw/analysis
   ```

**TraSw provenance limitation:** the exact historical FnF byte versions identified by four old E29 hashes are not preserved in the available snapshots. Those hashes remain recorded in `experiments/trasw/protocol.json`. The included public overlays are pinned to their actual reviewed hashes and support source-overlay re-execution, but a fresh TraSw run is not claimed to be bit-identical to the archived historical harness. The bundled E29 result tables themselves are unchanged and hash-validated.

## Verification performed for this release

- Synthetic clean/attack replay, CG activation, and Audit/C0 trajectory equality.
- Real MOT17-02 segment 000 and DanceTrack-0004 segment 000 replay under the pinned environment; both public track files are byte-identical to the original outputs.
- All curated result hashes and macro aggregations.
- Regeneration of the OC-SORT full macro table and quantitative paper figures.

The complete GPU matrices were not rerun merely for packaging.

## Known boundaries

- External frozen attack tensors and detection logs are prerequisites for ordinary replay; they are not regenerated by `prepare`.
- External baseline licenses and dataset terms continue to apply.
- Raw per-frame outputs and private development diagnostics are intentionally excluded.
- E30 does not claim a significant sensitivity result; E31 is clean timing only; E32 is visualization only.

## License

CommitGuard's own code is MIT-licensed. Vendored and external third-party components retain their original licenses; see [NOTICE.md](NOTICE.md) and `licenses/`.
