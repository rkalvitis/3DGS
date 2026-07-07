# Experiment scripts

All scripts are meant to run **inside the Singularity container** (see `../REPRODUCTION.md` §3 for building it), invoked as `bash /workspace/scripts/<script>.sh` with the code mounted at `/workspace`, data at `/data`, and outputs at `/output`.

There are three experiment groups:

| Group | Scripts | Guide |
|---|---|---|
| **Paper reproduction** — Kerbl et al. 2023 Table 1 on the original benchmark scenes | `run_experiments.sh`, `run_metrics_only.sh` | `../REPRODUCTION.md` |
| **LLFF** — 3DGS on the NeRF LLFF forward-facing dataset | `run_experiments_llff.sh` | — |
| **FineView** — 3DGS on the FineView insect dataset (calibrated multi-camera rig) | `run_fineview_colmap.sh`, `run_fineview_train.sh`, `run_fineview_seeds.sh` | `../RUN_FINEVIEW.md` |

## Paper reproduction

- **`run_experiments.sh`** — the full Table 1 reproduction: trains, renders, and evaluates 13 scenes (Mip-NeRF360, Tanks & Temples, Deep Blending) × 5 seeds = 65 sequential runs (~28–32 h). Writes one `<scene>_seed<N>/` directory per run plus `logs/`.
- **`run_metrics_only.sh`** — post-processing for already-trained runs: renders the 7k-iteration test views (30k already exist after training) and re-runs `metrics.py` so each `results.json` contains both `ours_7000` and `ours_30000`. Use it to backfill metrics without retraining.

## LLFF

- **`run_experiments_llff.sh`** — same train → render → metrics loop as the paper reproduction, but on the 8 NeRF LLFF scenes (fern, flower, fortress, horns, leaves, orchids, room, trex) × 5 seeds at the standard 4× downsampled resolution. Expects `$DATA_DIR/<scene>/sparse/0/` and `images_4/`. Aggregate with `make_llff_results.py` (repo root).

## FineView

- **`run_fineview_colmap.sh`** — dataset preparation: exports the 5 insect species from the FineView release to COLMAP format — masked white-background images, camera poses from the rig calibration h5, and the structured-light point cloud installed as `points3D.ply` (the 3DGS initialization). COLMAP SIFT+triangulation is opt-in via `RUN_COLMAP=1`.
- **`run_fineview_train.sh`** — trains 3DGS on each exported species (30k iterations, native resolution), then renders and computes metrics, including foreground-only metrics from the masks.
- **`run_fineview_seeds.sh`** — utility to (re)install the structured-light `.pcd` point clouds as `points3D.ply` in already-exported scenes, without redoing the full export. Only needed if the seed points were changed or lost.

## Common environment variables

| Variable | Meaning | Default |
|---|---|---|
| `CODE_DIR` | repo mount point | `/workspace` |
| `DATA_DIR` | dataset root | `/data` (LLFF: `/data/llff`) |
| `OUTPUT_DIR` | output root | `/output` (LLFF: `/output/llff`) |
| `DEVICE` | CUDA device index | `1` (FineView scripts: `0`) |
| `EXP_NAME` | optional experiment name — outputs and logs are grouped under `$OUTPUT_DIR/$EXP_NAME/` so repeated launches don't mix | unset (no subdirectory) |
| `RAW_DATA` | FineView raw data mount (colmap/seeds scripts only) | `/raw_data` |

After any experiment, aggregate results with `python /workspace/collect_results.py [output_dir]`.
