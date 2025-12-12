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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False




import glob
def search_file(folder, pattern):
    search_path = os.path.join(folder, pattern) 
    files = glob.glob(search_path, recursive=False) 
    return files

def find_ckpt(folder):
# Perform the search
    pattern = "chkpnt*.pth"  # File name pattern
    files_found = search_file(folder, pattern)
    # import ipdb; ipdb.set_trace()
    # Check the results
    if files_found:
        if len(files_found) == 1:
            print("Found exactly one matching file:")
            print(files_found[0])
            return os.path.basename(files_found[0])
        else:
            raise(f"Found multiple matching files ({len(files_found)}):")

    else:
        print(f"No files matching the pattern '{pattern}' were found!")


def restore_gaussian_models(base_folder, num_models=4):
    """
    Restores multiple Gaussian models from their respective checkpoint files.

    Args:
        base_folder (str): The base folder containing the "L1" subfolder with checkpoints.
        num_models (int): The number of Gaussian models to restore.
    
    Returns:
        list: A list of restored Gaussian models.
    """
    gaussian_models = []
    
    for i in range(1, num_models + 1):
        gaussians = GaussianModel(3)
        folder = os.path.join(base_folder, f"L{i}")
        
        # Find the checkpoint file in the folder
        ckpt_name = find_ckpt(folder)
        if ckpt_name is None:
            raise FileNotFoundError(f"No checkpoint found in {folder}")
        
        # Restore the model from the checkpoint
        ckpt_path = os.path.join(folder, ckpt_name)
        ckpt, _= torch.load(ckpt_path)
        if i > 1:
            gaussians.init_index()
        gaussians.restore(ckpt, training_args=None)
        gaussian_models.append(gaussians)
    
    return gaussian_models

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--folder", type=str, default=None)

    args = parser.parse_args()

    models = restore_gaussian_models(args.folder, num_models=4)

    gaussian_finest = models[0]

    shs_dcs = torch.zeros((gaussian_finest.get_xyz.shape[0], 4, 3))
    highest_levels = torch.zeros((gaussian_finest.get_xyz.shape[0], 1))
    opacities = torch.ones((gaussian_finest.get_xyz.shape[0], 4))

    gaussian_finest.init_index()
    shs_dcs[:, 0, :] = gaussian_finest.get_features[:, 0, :].cpu()
    opacities[:, 0] = gaussian_finest.get_opacity[:, 0].cpu()


    for i in range(1, 4):
        gaussian = models[i]

        shs_dcs[:, i, :] = shs_dcs[:, i-1, :]
        features = gaussian.get_features[:, 0, :].cpu().unsqueeze(1)
        shs_dcs[gaussian.indexes.cpu().long(), i, :] = features

        opacities[:, i] = opacities[:, i-1]
        opacity = gaussian.get_opacity[:, 0].cpu().unsqueeze(1)
        opacities[gaussian.indexes.cpu().long(), i] = opacity

        highest_levels[gaussian.indexes.cpu().long()] = i


    # save higest levels and shs_dcs   
    torch.save(highest_levels, os.path.join(args.folder, "highest_levels.pt"))
    torch.save(shs_dcs, os.path.join(args.folder, "shs_dcs.pt"))
    torch.save(opacities, os.path.join(args.folder, "opacities.pt"))

    # All done
    print("\Composing complete.")
