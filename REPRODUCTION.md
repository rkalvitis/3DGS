# Reproducing 3DGS Table 1 on rhea

Replicates **Kerbl et al. 2023 (3DGS) Table 1** — PSNR/SSIM/LPIPS across 13 scenes, each run 5 times with different random seeds. Runs sequentially on a single RTX 4080 16GB via Singularity on the rhea server.

**Scale:** 13 scenes × 5 seeds = 65 runs  
**Estimated time:** ~28–32 hours total (~25–30 min per run)

---

## Files in this directory

| File | Purpose |
|---|---|
| `train.py` | Modified to accept `--seed N` and `--device N` arguments |
| `render.py` | Modified to accept `--device N` argument |
| `metrics.py` | Modified to accept `--device N` argument |
| `utils/general_utils.py` | `safe_state` keeps `random`/`numpy`/`torch` CPU seeds fixed at 0 (original behaviour); only the CUDA seed varies per run via `--seed` |
| `3dgs.def` | Singularity definition file — original paper environment (PyTorch 1.12.1, Python 3.7.13) |
| `scripts/run_experiments.sh` | Runs all scenes × 5 seeds sequentially; set `DEVICE=` at the top to pick the GPU |

---

## Phase 1 — Connect to rhea

Connect via SUPSI VPN or campus ethernet, then SSH in:

```bash
ssh your_username@rhea.idsia.ch
```

**First time only** — redirect Singularity's temp files away from `/tmp`:

```bash
mkdir -p ~/.singularity/tmp
echo 'export SINGULARITY_TMPDIR=${HOME}/.singularity/tmp' >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 2 — Copy code to rhea

On rhea, clone your GitHub repo with `--recursive` to also pull all submodules (diff-gaussian-rasterization, simple-knn, fused-ssim). Without `--recursive` the submodule folders will be empty and the container build will fail.

```bash
git clone --recursive https://github.com/rkalvitis/3DGS.git ~/gaussian-splatting
```

For subsequent updates, pull on rhea after pushing from your local machine:

```bash
git -C ~/gaussian-splatting pull
git -C ~/gaussian-splatting submodule update --init --recursive
```

---

## Phase 3 — Download datasets on rhea

Store data on fast RAID storage, not in home:

```bash
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
mkdir -p /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_output_2
cd /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
```

**Mip-NeRF360** (~20 GB):

```bash
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d mipnerf360
```

**Tanks & Temples + Deep Blending** (both datasets in one zip, 650 MB):

```bash
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
```

The zip extracts directly into `tandt/` and `db/` — no reorganisation needed.

Required layout after downloading:

```
3dgs_data/
├── mipnerf360/
│   ├── bicycle/
│   ├── flowers/
│   ├── garden/
│   ├── stump/
│   ├── treehill/
│   ├── room/
│   ├── counter/
│   ├── kitchen/
│   └── bonsai/
├── tandt/
│   ├── truck/
│   └── train/
└── db/
    ├── playroom/
    └── drjohnson/
```

---

## Phase 4 — Build the Singularity container on rhea

The container **must be built on rhea** — the CUDA extensions are compiled during the build and must target the RTX 4080's architecture (sm_89). Takes 20–30 minutes.

```bash
mkdir -p ~/containers
singularity build --fakeroot ~/containers/3dgs.sif ~/gaussian-splatting/3dgs.def
```

Verify the build:

```bash
CUDA_VISIBLE_DEVICES=1 singularity exec --nv ~/containers/3dgs.sif python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
import diff_gaussian_rasterization; print('diff-gaussian-rasterization: OK')
import simple_knn; print('simple-knn: OK')
from fused_ssim import fused_ssim; print('fused-ssim: OK')
"
```

Expected output:

```
PyTorch: 2.0.1+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 4080
diff-gaussian-rasterization: OK
simple-knn: OK
fused-ssim: OK
```

---

## The definition file explained (`3dgs.def`)

The full file is in this directory. Here is what each block does and why each choice was made.

```singularity
Bootstrap: docker
From: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

Pulls an NVIDIA CUDA 11.8 image that includes the full developer toolkit (compiler, headers, libraries). This is the base for the entire container. Ubuntu 20.04 matches the era of the original paper and supports Python 3.7 cleanly.

**Why CUDA 11.8 despite the paper using cudatoolkit 11.6?**
The paper's README explicitly says *"known issues with 11.6"* for the SDK and confirms CUDA SDK 11.8 was used for compilation. The conda `cudatoolkit=11.6` package only provides runtime libraries — it does not install NVCC. The actual compiler (NVCC) comes from this base image and is version 11.8.

```singularity
%environment
    export PATH=/opt/conda/envs/gaussian_splatting/bin:/opt/conda/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-11.8
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
    export QT_QPA_PLATFORM=offscreen
```

Variables set every time the container runs. `TORCH_CUDA_ARCH_LIST` tells the CUDA extension build system which GPU architectures to compile for. `8.9` is sm_89 (RTX 4080 / Ada Lovelace). PyTorch 1.12.1 pre-built wheels do not include sm_89, so without this flag the extensions would not have native kernels for the RTX 4080. `QT_QPA_PLATFORM=offscreen` suppresses display errors on headless servers.

```singularity
%post
    # System build dependencies
    apt-get install -y build-essential cmake ninja-build ...
```

Installs system packages needed to compile the CUDA extensions: a C++ compiler, CMake, and ninja (a fast build backend used by PyTorch's extension builder).

```singularity
    # Install Miniconda with Python 3.7
    wget Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
    bash miniconda.sh -b -p /opt/conda
```

Installs conda with a Python 3.7 base, matching the paper's `python=3.7.13`. We use the py37 variant of Miniconda so the base interpreter is immediately 3.7.

```singularity
    conda install -n gaussian_splatting \
        pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 \
        cudatoolkit=11.6 pip=22.3.1 plyfile tqdm
```

Installs the exact versions from `environment.yml`. `cudatoolkit=11.6` provides the CUDA 11.6 runtime libraries that PyTorch 1.12.1 was linked against. `pip=22.3.1` matches the lockfile pin.

```singularity
    pip install opencv-python-headless joblib
```

Remaining pip dependencies from `environment.yml`. `opencv-python-headless` is used instead of `opencv-python` because the server has no display and the headless variant avoids pulling in GUI libraries.

```singularity
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting \
        /opt/gaussian-splatting-src

    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn
    pip install submodules/fused-ssim
```

Clones upstream 3DGS to get the submodule source code, then compiles and installs the three CUDA extensions into the container's Python. At build time, `TORCH_CUDA_ARCH_LIST` and `CUDA_HOME` are active, so NVCC 11.8 compiles native kernels for all listed architectures including sm_89. The upstream clone is deleted afterwards — at runtime your own code is mounted at `/workspace`.

```singularity
%labels / %help
```

Metadata and usage hints embedded in the image, readable with `singularity inspect ~/containers/3dgs.sif`.

---

## Phase 5 — Book the GPU and launch

**1. Check MSTeams** (`rhea-users` channel) for a free GPU.

**2. Confirm it is idle on rhea:**

```bash
nvidia-smi -l 1
# Free GPU = 0MiB memory usage, 0% GPU-Util. Press Ctrl+C to exit.
```

**3. Claim it on MSTeams:** post `"Using GPU X"`.

**4. Open a screen session** — the experiments run for ~30 hours, so this keeps them alive if your SSH connection drops:

```bash
screen -S 3dgs -U
```

Reattach at any time with `screen -r 3dgs -U`.

**5. Launch inside the screen session:**

```bash
export CODE_DIR=/home/$USER/gaussian-splatting
export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_data
export OUTPUT_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_output_2

mkdir -p "$OUTPUT_DIR"

singularity exec --nv --cleanenv --contain --bind "$CODE_DIR:/workspace" --bind "$DATA_DIR:/data" --bind "$OUTPUT_DIR:/output" --bind "/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache:/torch_cache" ~/containers/3dgs.sif bash /workspace/scripts/run_experiments.sh
```

| Flag | What it does |
|---|---|
| `--nv` | Exposes NVIDIA drivers to the container. Required for GPU access. |
| `--cleanenv` | Does not import host environment variables. Prevents version conflicts. |
| `--contain` | Does not auto-mount your home directory. You control what the container sees. |
| `--bind src:dest` | Mounts a host path at a path inside the container. |

**6. Detach from screen** so it keeps running after you close the terminal:

Press `Ctrl+A`, then `D`.

---

## Phase 6 — Monitor progress

```bash
# How many of the 65 runs have completed
ls /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_output_2/ | grep -v logs | wc -l

# Live log for whichever run is currently executing
tail -f /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_output_2/logs/bicycle_seed0.log

# GPU utilisation
nvidia-smi
```

---

## Phase 7 — Collect results

After all 65 runs finish, aggregate the `results.json` files using `collect_results.py` (included in this directory):

```bash
singularity exec --nv --cleanenv --contain \
    --bind /media/white/nanodrones/roberts.kalvitis/3dgs/3dgs_output_2:/output \
    --bind /home/$USER/gaussian-splatting:/workspace \
    ~/containers/3dgs.sif \
    python /workspace/collect_results.py
```

**Free the GPU when done:**

```bash
nvidia-smi  # confirm no processes running
# then post on MSTeams: "GPU X is free"
```

---

## Quick reference

| Item | Detail |
|---|---|
| GPU | 1× RTX 4080 16GB — set `DEVICE=1` at top of `scripts/run_experiments.sh` to change |
| Container | Python 3.7.13, PyTorch 1.12.1+cu116, CUDA SDK 11.8, built on rhea |
| Scenes | 9× Mip-NeRF360, 2× Tanks & Temples, 2× Deep Blending |
| Seeds | 5 (0, 1, 2, 3, 4) |
| Total runs | 65, sequential |
| Estimated wall time | ~28–32 hours |
| If SSH drops | `screen -r 3dgs -U` |
| Output per run | `results.json` — PSNR/SSIM/LPIPS at 7k and 30k iterations |
