# Reproducing 3DGS Table 1

Reproduces **Kerbl et al. 2023 (3D Gaussian Splatting) Table 1** — PSNR/SSIM/LPIPS across 13 scenes, each trained with 5 random seeds.

**Scale:** 13 scenes × 5 seeds = 65 sequential runs
**Hardware:** one NVIDIA GPU with ≥16 GB VRAM (tested on RTX 4080)
**Time:** ~25–30 min per run, ~28–32 h total
**Disk:** ~21 GB datasets + ~80–130 GB outputs

Changes vs the [official codebase](https://github.com/graphdeco-inria/gaussian-splatting):

| File | Change |
|---|---|
| `train.py`, `render.py`, `metrics.py` | Accept `--seed N` and `--device N`; training also writes `hyperparams.txt` and `training_time.txt` per run |
| `utils/general_utils.py` | `safe_state` keeps CPU seeds fixed at 0 (original behaviour); only the CUDA seed varies per run |
| `3dgs.def` | Singularity container definition (Python 3.8, PyTorch 2.0.1+cu118, CUDA SDK 11.8) |
| `scripts/run_experiments.sh` | Runs all 13 scenes × 5 seeds sequentially |
| `collect_results.py` | Aggregates per-run `results.json` into per-scene means |

---

## 1 — Clone the repo

Clone with `--recursive` — without it the CUDA extension submodules are empty and the container build fails.

```bash
git clone --recursive -b fineview https://github.com/rkalvitis/3DGS.git ~/gaussian-splatting
```

---

## 2 — Download the datasets

Pick a data directory (`$DATA_DIR`) on fast storage:

```bash
export DATA_DIR=/path/to/3dgs_data
mkdir -p "$DATA_DIR" && cd "$DATA_DIR"

# Mip-NeRF360 (~17 GB total)
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d mipnerf360
wget http://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
unzip 360_extra_scenes.zip -d mipnerf360   # flowers + treehill

# Tanks & Temples + Deep Blending (650 MB, one zip)
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
```

Required layout:

```
$DATA_DIR/
├── mipnerf360/   bicycle flowers garden stump treehill room counter kitchen bonsai
├── tandt/        truck train
└── db/           playroom drjohnson
```

---

## 3 — Build the Singularity container

Build **on the machine with the target GPU** — the CUDA extensions are compiled during the build (`TORCH_CUDA_ARCH_LIST` covers sm 7.0–8.9; extend it in `3dgs.def` for other architectures). Takes ~30 min.

```bash
# one-time: keep Singularity temp files out of /tmp
mkdir -p ~/.singularity/tmp
export SINGULARITY_TMPDIR=$HOME/.singularity/tmp

mkdir -p ~/containers
singularity build --fakeroot ~/containers/3dgs.sif ~/gaussian-splatting/3dgs.def
```

Verify:

```bash
singularity exec --nv ~/containers/3dgs.sif python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
import diff_gaussian_rasterization; print('diff-gaussian-rasterization: OK')
import simple_knn; print('simple-knn: OK')
from fused_ssim import fused_ssim; print('fused-ssim: OK')
"
```

Expected: `PyTorch: 2.0.1+cu118`, `CUDA available: True`, and all three extensions OK.

---

## 4 — Run the experiments

The job runs ~30 h, so launch it inside `screen` (reattach later with `screen -r 3dgs -U`). On a shared server, claim a free GPU first (`nvidia-smi`) and select it with `DEVICE`.

**Name every experiment with `EXP_NAME`.** All outputs and logs of a launch are grouped under `$OUTPUT_DIR/$EXP_NAME/` — without it, repeated launches dump 65 run directories into the same folder and become impossible to tell apart.

```bash
screen -S 3dgs -U

export CODE_DIR=~/gaussian-splatting
export DATA_DIR=/path/to/3dgs_data
export OUTPUT_DIR=/path/to/3dgs_output
export TORCH_CACHE=/path/to/torch_cache   # torchvision weights cache for LPIPS
mkdir -p "$OUTPUT_DIR" "$TORCH_CACHE"
```

<details>
<summary>Example — original setup on rhea (rhea.idsia.ch)</summary>

```bash
export CODE_DIR=/home/robertsk/gaussian-splatting
export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_output_2
export TORCH_CACHE=/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache
```

Container at `~/containers/3dgs.sif`. Check the MSTeams `rhea-users` channel for a free GPU, claim it there, and set `DEVICE` accordingly.
</details>

```bash

singularity exec --nv --cleanenv --contain \
    --env DEVICE=${DEVICE:-1} \
    --env EXP_NAME=table1_run1 \
    --bind "$CODE_DIR:/workspace" \
    --bind "$DATA_DIR:/data" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$TORCH_CACHE:/torch_cache" \
    ~/containers/3dgs.sif \
    bash /workspace/scripts/run_experiments.sh
```

Detach from screen with **Ctrl+A, D**.

| Flag | What it does |
|---|---|
| `--nv` | Exposes the NVIDIA driver to the container |
| `--cleanenv --contain` | No host env vars or auto-mounted home — the container only sees the binds |
| `DEVICE` | CUDA device index used by all runs (default 1) |
| `EXP_NAME` | Experiment name — groups this launch's 65 runs (and logs) under `$OUTPUT_DIR/$EXP_NAME/`. Use a fresh name per launch |

---

## 5 — Monitor

```bash
ls "$OUTPUT_DIR"/table1_run1 | grep -v logs | wc -l          # runs completed (of 65)
tail -f "$OUTPUT_DIR"/table1_run1/logs/bicycle_seed0.log     # live log of current run
nvidia-smi                                                   # GPU utilisation
```

---

## 6 — Collect results

After all runs finish:

```bash
singularity exec --nv --cleanenv --contain \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$CODE_DIR:/workspace" \
    ~/containers/3dgs.sif \
    python /workspace/collect_results.py /output/table1_run1
```

Prints mean PSNR/SSIM/LPIPS per scene across all seeds. Point it at one experiment directory — running it on `$OUTPUT_DIR` itself would mix the runs of every experiment stored there. Per-run metrics are in `$OUTPUT_DIR/<scene>_seed<N>/results.json` (7k and 30k iterations); each run directory also contains `cfg_args`, `hyperparams.txt`, `training_time.txt`, the Gaussian point clouds, and rendered test views.
