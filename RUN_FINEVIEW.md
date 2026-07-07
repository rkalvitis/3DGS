# FineView → 3DGS

Converts FineView insect captures (calibrated multi-camera rig, 8 lenses × 40 rotations) to COLMAP format — masked images on white background, poses from the rig calibration, structured-light point cloud as initialization — then trains 3DGS per species.

**Species:** 009-Neophasia_Menapia-001, 072-Colias_Eurytheme-002, 110-Nymphalis_l_album-001, 184-Speyeria_Hydaspe-001, 195-Lycaena_Arota-002
**Time:** ~1–2 h export + ~1 h training per species
**Container:** the same `3dgs.sif` as the Table 1 reproduction (it includes COLMAP, h5py, Pillow) — see `REPRODUCTION.md` §3 for building it.

---

## 1 — Clone the repo

```bash
git clone --recursive -b fineview https://github.com/rkalvitis/3DGS.git ~/gaussian-splatting
```

---

## 2 — Prepare the raw data

The pipeline expects one directory (`$RAW_DATA`) with the FineView release layout:

```
$RAW_DATA/
├── camera_parameters.h5            # rig calibration (poses + intrinsics)
├── crop_undistort/                 # undistorted RGB crops, one folder per species
│   └── <species>/…                 # several layouts auto-detected, e.g. camera1/00.png
├── crop_mask_undistort/            # binary foreground masks, same layout
└── correspondence_undistort/       # structured-light point clouds (.pcd per species)
```

Copy the data to the machine, e.g.:

```bash
rsync -avz --progress /local/fineview_data/ user@server:/path/to/raw_data/
```

---

## 3 — Export to COLMAP format

Writes one COLMAP scene per species: masked white-background images, `cameras.txt`/`images.txt` from the h5 calibration, and the structured-light cloud installed as `points3D.ply` (the 3DGS init). COLMAP SIFT+triangulation is **off by default** — the structured-light cloud is used instead; set `RUN_COLMAP=1` only if you need a triangulated cloud for comparison.

```bash
screen -S fineview -U

export CODE_DIR=~/gaussian-splatting
export RAW_DATA=/path/to/raw_data
export DATA_DIR=/path/to/colmap_scenes
mkdir -p "$DATA_DIR"

singularity exec --nv --cleanenv --contain \
    --bind "$CODE_DIR:/workspace" \
    --bind "$RAW_DATA:/raw_data" \
    --bind "$DATA_DIR:/data" \
    ~/containers/3dgs.sif \
    bash /workspace/scripts/run_fineview_colmap.sh
```

Result, per species:

```
$DATA_DIR/<species>/
├── sparse/0/            cameras.txt, images.txt, points3D.ply
└── images/              camera1/00.png … camera8/39.png  (white background)
```

---

## 4 — Train

Trains, renders, and computes metrics for all 5 species sequentially (~5 h total):

```bash
export OUTPUT_DIR=/path/to/fineview_output
export TORCH_CACHE=/path/to/torch_cache   # torchvision weights cache for LPIPS
mkdir -p "$OUTPUT_DIR" "$TORCH_CACHE"

singularity exec --nv --cleanenv --contain \
    --env DEVICE=${DEVICE:-0} \
    --bind "$CODE_DIR:/workspace" \
    --bind "$DATA_DIR:/data" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$TORCH_CACHE:/torch_cache" \
    ~/containers/3dgs.sif \
    bash /workspace/scripts/run_fineview_train.sh
```

Optional: `EXP_NAME=my_experiment` nests all outputs under `$OUTPUT_DIR/my_experiment/`.

Training settings (see `scripts/run_fineview_train.sh`):

| Setting | Value | Reason |
|---|---|---|
| `--iterations` | 30000 | Standard 3DGS schedule, paper defaults |
| `--test/save_iterations` | 30000 | Evaluate and save only the final model |
| `--resolution` | 1 | Train at native image resolution |
| `--data_device` | cpu | Keep images in RAM, not VRAM |
| `--seed` | 1 | Fixed for reproducibility |
| `--sh_degree` | 3 | Full SH for view-dependent wing appearance |

Because the images are RGBA-masked, `render.py` also saves per-view foreground masks and `metrics.py` reports additional foreground-only metrics (`PSNR_fg`, `SSIM_fg`, `LPIPS_fg`).

---

## 5 — Monitor and collect results

```bash
tail -f "$OUTPUT_DIR"/logs/009-Neophasia_Menapia-001.log   # live log
nvidia-smi                                                 # GPU utilisation
```

After all species finish:

```bash
singularity exec --nv --cleanenv --contain \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$CODE_DIR:/workspace" \
    ~/containers/3dgs.sif \
    python /workspace/collect_results.py
```

Per-species metrics are in `$OUTPUT_DIR/<species>/results.json`; each model directory also contains `hyperparams.txt`, `training_time.txt`, the trained point cloud, and rendered test views.
