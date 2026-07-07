#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, masks_dir=None):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        if masks_dir is not None and (masks_dir / fname).exists():
            mask = tf.to_tensor(Image.open(masks_dir / fname)).unsqueeze(0)[:, :1, :, :].cuda()
            masks.append((mask > 0.5).float())
        else:
            masks.append(None)
        image_names.append(fname)
    return renders, gts, masks, image_names

def masked_psnr(render, gt, mask):
    """Foreground-only PSNR: MSE averaged over masked pixels only."""
    sq_err = ((render - gt) ** 2) * mask
    n = mask.sum() * render.shape[1]            # foreground pixels × channels
    mse = sq_err.sum() / torch.clamp(n, min=1.0)
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "mask"
                renders, gts, masks, image_names = readImages(
                    renders_dir, gt_dir, masks_dir if masks_dir.exists() else None)
                have_masks = any(m is not None for m in masks)

                ssims = []
                psnrs = []
                lpipss = []
                ssims_fg = []
                psnrs_fg = []
                lpipss_fg = []

                lpips_criterion = LPIPS(net_type='vgg').cuda()

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_criterion(renders[idx], gts[idx]))

                    # Foreground-only metrics: mask both images so only the insect counts.
                    if masks[idx] is not None:
                        m = masks[idx]
                        r_fg, g_fg = renders[idx] * m, gts[idx] * m
                        psnrs_fg.append(masked_psnr(renders[idx], gts[idx], m))
                        ssims_fg.append(ssim(r_fg, g_fg))
                        lpipss_fg.append(lpips_criterion(r_fg, g_fg))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                if have_masks:
                    print("  -- foreground only --")
                    print("  SSIM_fg : {:>12.7f}".format(torch.tensor(ssims_fg).mean(), ".5"))
                    print("  PSNR_fg : {:>12.7f}".format(torch.tensor(psnrs_fg).mean(), ".5"))
                    print("  LPIPS_fg: {:>12.7f}".format(torch.tensor(lpipss_fg).mean(), ".5"))
                print("")

                metrics_summary = {"SSIM": torch.tensor(ssims).mean().item(),
                                   "PSNR": torch.tensor(psnrs).mean().item(),
                                   "LPIPS": torch.tensor(lpipss).mean().item()}
                if have_masks:
                    metrics_summary.update({"SSIM_fg": torch.tensor(ssims_fg).mean().item(),
                                            "PSNR_fg": torch.tensor(psnrs_fg).mean().item(),
                                            "LPIPS_fg": torch.tensor(lpipss_fg).mean().item()})
                full_dict[scene_dir][method].update(metrics_summary)
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("  Error:", repr(e))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    torch.cuda.set_device(torch.device(f"cuda:{args.device}"))
    evaluate(args.model_paths)
