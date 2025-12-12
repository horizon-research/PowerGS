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
from utils.display_power import display_power_metric_boost, display_power_metric
from utils.hvs_loss_calc import HVSLoss
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser



def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if "orig" in fname.split("_")[-1]:
            continue
        # print(fname)
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
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
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []
            display_powers = []
            display_powers_new = []
            hvs_losses = []
            # import ipdb; ipdb.set_trace()
            hvs_calc = HVSLoss(uniform_pooling_size=args.pooling_size)
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(-1)
                psnrs.append(-1)
                lpipss.append(-1)
                display_powers.append(-1)
                display_powers_new.append(-1)
                # print(args.pooling_size)
                hvs_losses.append(hvs_calc.calc_uniform_loss(renders[idx], gts[idx], pooling_size=args.pooling_size))



            full_dict[scene_dir][method].update({"SSIM":  -1,
                                                    "PSNR": -1,
                                                    "LPIPS": -1,
                                                    "Display Power": -1,
                                                    "HVS": torch.tensor(hvs_losses).mean().item()})
                                        
            per_view_dict[scene_dir][method].update({"HVS": {name: hvs.item() for hvs, name in zip(hvs_losses, image_names)}})

        with open(scene_dir + "/results_hvsq_p" + str(args.pooling_size) + ".json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view_hvsq_p" + str(args.pooling_size) + ".json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--pooling_size', '-p', required=True, type=int, default=1)
    args = parser.parse_args()
    evaluate(args.model_paths)
