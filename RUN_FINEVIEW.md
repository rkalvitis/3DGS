# FineView → 3DGS

Converts FineView insect captures (calibrated multi-camera rig, 8 lenses × 40 rotations) to COLMAP format — masked images on white background, poses from the rig calibration, structured-light point cloud as initialization — then trains 3DGS per species.

**Species:** 009-Neophasia_Menapia-001, 072-Colias_Eurytheme-002, 110-Nymphalis_l_album-001, 184-Speyeria_Hydaspe-001, 195-Lycaena_Arota-002
**Time:** ~1–2 h export + ~1 h training per species

---

## 1 — Clone the repo

```bash
git clone --recursive -b fineview https://github.com/rkalvitis/3DGS.git ~/gaussian-splatting
```

---

## 2 — Build the Singularity container

The same `3dgs.sif` serves the Table 1 reproduction and FineView — it includes COLMAP, h5py, and Pillow. Build **on the machine with the target GPU** (the CUDA extensions are compiled during the build; ~30 min):

```bash
mkdir -p ~/.singularity/tmp && export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
mkdir -p ~/containers
singularity build --fakeroot ~/containers/3dgs.sif ~/gaussian-splatting/3dgs.def
```

Verify:

```bash
singularity exec --nv ~/containers/3dgs.sif bash -c "
    colmap help | head -1
    python -c 'import h5py, PIL, tqdm; print(\"deps OK\")'
    python -c 'import diff_gaussian_rasterization; print(\"rasterizer OK\")'
"
```

Expected: a COLMAP version line, `deps OK`, `rasterizer OK`.

---

## 3 — Prepare the raw data

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

## 4 — Export to COLMAP format

Writes one COLMAP scene per species: masked white-background images, `cameras.txt`/`images.txt` from the h5 calibration, and the structured-light cloud installed as `points3D.ply` (the 3DGS init). COLMAP SIFT+triangulation is **off by default** — the structured-light cloud is used instead; set `RUN_COLMAP=1` only if you need a triangulated cloud for comparison.

```bash
screen -S fineview -U

export CODE_DIR=~/gaussian-splatting
export RAW_DATA=/path/to/raw_data
export DATA_DIR=/path/to/colmap_scenes
mkdir -p "$DATA_DIR"
```

<details>
<summary>Example — original setup on rhea (rhea.idsia.ch)</summary>

```bash
export CODE_DIR=/home/robertsk/gaussian-splatting
export RAW_DATA=/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data
export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output
export TORCH_CACHE=/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache
```

Container at `~/containers/3dgs.sif`; h5 calibration at `$RAW_DATA/camera_parameters.h5`; source data copied from the Mac at `~/Documents/repos/3DGS/imgs_filtered/` (`crop_undistort/`, `crop_mask_undistort/`, `correspondence_undistort/`) and `~/Documents/repos/fineview/camera_parameters.h5`.
</details>

```bash

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

## 5 — Train

Trains, renders, and computes metrics for all 5 species sequentially (~5 h total).

**Name every experiment with `EXP_NAME`.** All outputs and logs of a launch are grouped under `$OUTPUT_DIR/$EXP_NAME/` — use a fresh name per launch (e.g. `baseline_30k`, `white_noise_test`) so different experiments never mix in the output directory.

```bash
export OUTPUT_DIR=/path/to/fineview_output
export TORCH_CACHE=/path/to/torch_cache   # torchvision weights cache for LPIPS
mkdir -p "$OUTPUT_DIR" "$TORCH_CACHE"

singularity exec --nv --cleanenv --contain \
    --env DEVICE=${DEVICE:-0} \
    --env EXP_NAME=fineview_run1 \
    --bind "$CODE_DIR:/workspace" \
    --bind "$DATA_DIR:/data" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$TORCH_CACHE:/torch_cache" \
    ~/containers/3dgs.sif \
    bash /workspace/scripts/run_fineview_train.sh
```

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

## 6 — Monitor and collect results

```bash
tail -f "$OUTPUT_DIR"/fineview_run1/logs/009-Neophasia_Menapia-001.log   # live log
nvidia-smi                                                               # GPU utilisation
```

After all species finish (point `collect_results.py` at one experiment directory, not `$OUTPUT_DIR` itself):

```bash
singularity exec --nv --cleanenv --contain \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$CODE_DIR:/workspace" \
    ~/containers/3dgs.sif \
    python /workspace/collect_results.py /output/fineview_run1
```

Per-species metrics are in `$OUTPUT_DIR/<exp_name>/<species>/results.json`; each model directory also contains `hyperparams.txt`, `training_time.txt`, the trained point cloud, and rendered test views.
