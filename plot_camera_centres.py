#!/usr/bin/env python
"""
Plot camera centres in 3-D for every scene under an output root.

Mirrors fineview_pipeline.verify.plot_camera_centres, but reads camera
positions from each trained model's ``cameras.json`` (written by Scene at
training time) instead of the FineView H5 calibration. Works for any scene
(bicycle, garden, drjohnson, ...).

Seeds of the same scene share identical cameras, so only the first seed found
per scene is plotted. One PNG per scene is written to <output_dir>/camera_plots/.

Usage (inside the container):
    python /workspace/plot_camera_centres.py --output_dir /output
"""
import argparse
import json
import os
import re
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

SEED_RE = re.compile(r"_seed\d+$")


def scene_name(dirname: str) -> str:
    """Strip a trailing _seed<N> so all seeds of a scene group together."""
    return SEED_RE.sub("", dirname)


def find_scenes(output_dir: str) -> "OrderedDict[str, str]":
    """Map scene -> first model dir (with a cameras.json) found for it."""
    scenes: "OrderedDict[str, str]" = OrderedDict()
    for entry in sorted(os.listdir(output_dir)):
        if entry in ("logs", "camera_plots"):
            continue
        model_dir = os.path.join(output_dir, entry)
        cams_json = os.path.join(model_dir, "cameras.json")
        if not os.path.isfile(cams_json):
            continue
        scenes.setdefault(scene_name(entry), model_dir)
    return scenes


def plot_scene(scene: str, model_dir: str, save_path: str) -> int:
    with open(os.path.join(model_dir, "cameras.json")) as f:
        cams = json.load(f)
    centres = np.array([c["position"] for c in cams])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], s=5, alpha=0.5)
    ax.set_title(f"Camera centres  {scene}  ({len(centres)} views)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    return len(centres)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="/output",
                    help="Root holding <scene>_seed<N>/ model dirs")
    ap.add_argument("--plots_dir", default=None,
                    help="Where to write PNGs (default: <output_dir>/camera_plots)")
    args = ap.parse_args()

    plots_dir = args.plots_dir or os.path.join(args.output_dir, "camera_plots")
    os.makedirs(plots_dir, exist_ok=True)

    scenes = find_scenes(args.output_dir)
    if not scenes:
        print(f"No model dirs with cameras.json found under {args.output_dir}")
        return

    print(f"Found {len(scenes)} scenes under {args.output_dir}")
    for scene, model_dir in scenes.items():
        save_path = os.path.join(plots_dir, f"{scene}_cameras.png")
        n = plot_scene(scene, model_dir, save_path)
        print(f"  {scene:20s} {n:4d} views  -> {save_path}  (from {os.path.basename(model_dir)})")

    print(f"\nDone. Plots in {plots_dir}")


if __name__ == "__main__":
    main()
