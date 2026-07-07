# FineView → 3DGS on rhea

Converts 5 FineView insect species to COLMAP format (with background masking), then trains 3DGS on the rhea GPU server.

**Total estimated time:** ~1–2 h COLMAP + ~5 h training (5 species × ~60 min each at 60k iters)

---

## Paths at a glance

| What | rhea path |
|---|---|
| Code | `/home/robertsk/gaussian-splatting` |
| Raw images | `/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_undistort` |
| Masks | `/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_mask_undistort` |
| H5 calibration | `/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/camera_parameters.h5` |
| COLMAP output (scenes) | `/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data` |
| 3DGS training output | `/media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output` |
| Torch model cache | `/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache` |
| Singularity container | `~/containers/3dgs.sif` |

---

## Phase 1 — Push the fineview branch from your Mac

```bash
cd /Users/robertskalvitis/Documents/repos/3DGS

git checkout -b fineview   # or: git checkout fineview
git add fineview_pipeline/ \
        3dgs.def \
        scripts/run_fineview_colmap.sh \
        scripts/run_fineview_train.sh \
        RUN_FINEVIEW.md
git commit -m "Add FineView pipeline with masking, COLMAP, training scripts"
git push origin fineview
```

---

## Phase 2 — Set up rhea

SSH in (SUPSI VPN or campus ethernet required):

```bash
ssh robertsk@rhea.idsia.ch
```

### 2a — Fetch the fineview branch

If the repo is already cloned:

```bash
git -C ~/gaussian-splatting pull
git -C ~/gaussian-splatting checkout fineview
git -C ~/gaussian-splatting submodule update --init --recursive
```

If starting fresh:

```bash
git clone --recursive -b fineview \
    https://github.com/rkalvitis/3DGS.git \
    ~/gaussian-splatting
```

### 2b — Create directories

```bash
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_undistort
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_mask_undistort
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache
```

---

## Phase 3 — Transfer data to rhea

Run from your **Mac**:

```bash
# Images
rsync -avz --progress \
    /Users/robertskalvitis/Documents/repos/3DGS/imgs_filtered/crop_undistort/ \
    robertsk@rhea.idsia.ch:/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_undistort/

# Masks
rsync -avz --progress \
    /Users/robertskalvitis/Documents/repos/3DGS/imgs_filtered/crop_mask_undistort/ \
    robertsk@rhea.idsia.ch:/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_mask_undistort/

# Correspondence point clouds (structured-light reconstruction per species)
rsync -avz --progress \
    /Users/robertskalvitis/Documents/repos/3DGS/imgs_filtered/correspondence_undistort/ \
    robertsk@rhea.idsia.ch:/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/correspondence_undistort/

# H5 calibration file
rsync -avz --progress \
    /Users/robertskalvitis/Documents/repos/fineview/camera_parameters.h5 \
    robertsk@rhea.idsia.ch:/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/
```

Verify on rhea:

```bash
ls /media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/
# Expected:
#   crop_undistort/
#   crop_mask_undistort/
#   camera_parameters.h5

ls /media/white/nanodrones/roberts.kalvitis/3dgs/raw_data/crop_undistort/
# Expected: 009-Neophasia_Menapia-001/  072-Colias_Eurytheme-002/  ...
```

---

## Phase 4 — Rebuild the Singularity container

`3dgs.def` includes COLMAP (GPU SIFT), h5py, and Pillow. Rebuild on rhea (~30–40 min):

```bash
singularity build --fakeroot ~/containers/3dgs.sif \
    ~/gaussian-splatting/3dgs.def
```

Verify:

```bash
singularity exec --nv ~/containers/3dgs.sif bash -c "
    colmap help | head -1
    python -c 'import h5py, PIL, tqdm; print(\"deps OK\")'
    python -c 'import diff_gaussian_rasterization; print(\"rasterizer OK\")'
"
```

Expected:

```
COLMAP 3.x.x -- Structure-from-Motion and Multi-View Stereo
deps OK
rasterizer OK
```

---

## Phase 5 — Book a GPU

1. Check **MSTeams** (`rhea-users` channel) for a free GPU.
2. Confirm it is idle: `nvidia-smi -l 1` (look for 0 MiB usage, 0% util).
3. Post `"Using GPU 0"` on MSTeams.

---

## Phase 6 — Run COLMAP (export + triangulation)

This exports masked images (white background) and triangulates 3D points using known calibrated poses from the h5 file. ~1–2 hours.

```bash
screen -S fineview -U

export CODE_DIR=/home/robertsk/gaussian-splatting
export RAW_DATA=/media/white/nanodrones/roberts.kalvitis/3dgs/raw_data
export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data

CUDA_VISIBLE_DEVICES=0 singularity exec --nv --cleanenv --contain \
    --bind "$CODE_DIR:/workspace" \
    --bind "$RAW_DATA:/raw_data" \
    --bind "$DATA_DIR:/data" \
    ~/containers/3dgs.sif \
    bash /workspace/scripts/run_fineview_colmap.sh
```

Detach: **Ctrl+A, D** — reattach: `screen -r fineview -U`

### What this produces

```
3dgs_data/
  009-Neophasia_Menapia-001/
    sparse/0/
      cameras.txt       intrinsics for 8 physical lenses
      images.txt        calibrated poses (from h5, not SfM)
      points3D.bin      triangulated + outlier-filtered point cloud
    images/
      camera1/00.png … camera8/39.png   (insect on white background)
  072-Colias_Eurytheme-002/
  ...
```

### If you need to re-export images only (poses + point cloud already done)

```bash
CUDA_VISIBLE_DEVICES=0 singularity exec --nv --cleanenv --contain \
    --bind "$CODE_DIR:/workspace" \
    --bind "$RAW_DATA:/raw_data" \
    --bind "$DATA_DIR:/data" \
    ~/containers/3dgs.sif python -c "
import sys; sys.path.insert(0, '/workspace')
from fineview_pipeline.export_colmap import export_colmap

species = [
    (9,   '009-Neophasia_Menapia-001'),
    (72,  '072-Colias_Eurytheme-002'),
    (110, '110-Nymphalis_l_album-001'),
    (184, '184-Speyeria_Hydaspe-001'),
    (195, '195-Lycaena_Arota-002'),
]
for sid, sname in species:
    export_colmap(
        h5_path='/raw_data/camera_parameters.h5',
        base_path='/raw_data',
        species_id=sid, species_name=sname,
        out_dir=f'/data/{sname}',
        img_dir='/raw_data/crop_undistort',
        mask_dir='/raw_data/crop_mask_undistort',
    )
"
```

---

## Phase 7 — Train 3DGS

30 000 iterations, seed 1, evaluated only at 30k. ~1 hour per species.

First, create the output directory if it doesn't exist:

```bash
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output
```

Then run (single line):

```bash
export CODE_DIR=/home/robertsk/gaussian-splatting; export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data; export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output; CUDA_VISIBLE_DEVICES=0 singularity exec --nv --cleanenv --contain --bind "$CODE_DIR:/workspace" --bind "$DATA_DIR:/data" --bind "$OUTPUT_DIR:/output" --bind "/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache:/torch_cache" ~/containers/3dgs.sif bash /workspace/scripts/run_fineview_train.sh
```

Training logs: `$OUTPUT_DIR/logs/<species>.log`

### Training settings (scripts/run_fineview_train.sh)

| Setting | Value | Reason |
|---|---|---|
| `--iterations` | 30000 | Standard 3DGS convergence point |
| `--test_iterations` | 30000 | Evaluate only at end |
| `--save_iterations` | 30000 | Save only final model |
| `--seed` | 1 | Fixed for reproducibility |
| `--sh_degree` | 3 | Full SH for view-dependent wing appearance |
| `--densify_grad_threshold` | 0.0001 | Aggressive densification for wing detail |
| `--densify_until_iter` | 25000 | Densify for most of training |
| `--device` | 0 | GPU 0 (`CUDA_VISIBLE_DEVICES=0`) |

---

## Phase 8 — Monitor progress

```bash
# How many species have finished
ls /media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output/ | grep -v logs | wc -l

# Live log
tail -f /media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output/logs/009-Neophasia_Menapia-001.log

# GPU utilisation
nvidia-smi
```

---

## Phase 9 — Collect results

```bash
singularity exec --nv --cleanenv --contain \
    --bind /media/white/nanodrones/roberts.kalvitis/3dgs/fineview_output:/output \
    --bind /home/robertsk/gaussian-splatting:/workspace \
    ~/containers/3dgs.sif \
    python /workspace/collect_results.py
```

Free the GPU:

```bash
nvidia-smi   # confirm no processes running
# Post on MSTeams: "GPU 0 is free"
```

---

## Quick reference

| Item | Detail |
|---|---|
| Branch | `fineview` |
| Container | `~/containers/3dgs.sif` — Python 3.8, PyTorch 2.0.1, COLMAP 3.13, CUDA 11.8 |
| COLMAP script | `scripts/run_fineview_colmap.sh` — export + triangulate, ~1–2 h |
| Training script | `scripts/run_fineview_train.sh` — train + render + metrics, ~5 h |
| Species | 009, 072, 110, 184, 195 |
| If SSH drops | `screen -r fineview -U` |
| Results | `results.json` in each `fineview_output/<species>/` dir |
