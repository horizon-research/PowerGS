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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import glob
def search_file(folder, pattern):
    search_path = os.path.join(folder, pattern) 
    files = glob.glob(search_path, recursive=True) 
    return files

def find_ckpt(folder):
# Perform the search
    pattern = "chkpnt*.pth"  # File name pattern
    files_found = search_file(folder, pattern)

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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, base_folder, Ln):
    render_path = os.path.join(base_folder, Ln, "test", "phLn", "renders")
    gts_path = os.path.join(base_folder, Ln, "test", "phLn", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, base_folder, Ln):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.init_index()
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

        ckpt_name = find_ckpt(folder = os.path.join(base_folder, f"{Ln}"))
        if ckpt_name is None:
            raise FileNotFoundError(f"No checkpoint found in {base_folder}")
        # Restore the model from the checkpoint
        ckpt_path = os.path.join(os.path.join(base_folder, f"{Ln}"), ckpt_name)
        ckpt, _= torch.load(ckpt_path)
        gaussians.restore(ckpt, training_args=None)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, base_folder, Ln)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, base_folder, Ln)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--FR_folder", type=str, default="output")
    parser.add_argument("--Ln", type=str, default="L1")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.FR_folder, args.Ln)