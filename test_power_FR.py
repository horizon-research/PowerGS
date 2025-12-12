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
from gaussian_power_analysis import render
from gaussian_power_analysis.fov_power_model import fov_hardware_power_modeling
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json




import glob
def search_file(folder, pattern):
    search_path = os.path.join(folder, pattern) 
    files = glob.glob(search_path, recursive=False) 
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



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, highest_levels, shs_dcs, opacities, base_folder):

    gaze_samples = [(0.25 * i, 0.25 * j) for i in range(1, 4) for j in range(1, 4)]
    print("Gaze samples:", gaze_samples)

    render_path = os.path.join(base_folder, name, "ph", "renders_fov")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    rendering_powers_path = os.path.join(base_folder, "FR_rendering_powers")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    rendering_powers = []


    shs_dcs = shs_dcs.cuda()
    highest_levels = highest_levels.cuda()
    opacities = opacities.cuda()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gaze = gaze_samples[idx % len(gaze_samples)]
        gazeArray = torch.tensor([gaze[0], gaze[1]]).float().cuda()
        renderings = render(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipeline,  # Add the 'pipe' argument here
            bg_color=background,
            alpha=0.05,
            gazeArray=gazeArray,
            highest_levels=highest_levels,
            shs_dcs=shs_dcs,
            opacities=opacities,
            fr=True
        )

        rendering = renderings["render"]

        result = fov_hardware_power_modeling(renderings["point_statistics"], renderings["pixel_statistics"])
        # import ipdb; ipdb.set_trace()
        # renderings["point_statistics"]
        # renderings["pixel_statistics"]
        # print(result)
        # gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + f"_{gaze[0]}_{gaze[1]}.png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        rendering_powers.append(result)
    
    # compute mean and list of per view power
    # Compute means for each metric
    metrics = ["total_power", "total_dram_power", "total_sram_power", "total_flops_power", 
            "preprocessing_power", "sorting_power", 
            "rendering_power"]


    mean_metrics = {
        f"mean_{metric}": sum(r[metric] for r in rendering_powers) / len(rendering_powers)
        for metric in metrics
    }

    # Create a JSON structure
    output_data = {
        "mean_metrics": mean_metrics,
        "per_view_metrics": rendering_powers
    }

    # Save per-view and mean metrics in a JSON file with good formatting
    output_path = rendering_powers_path + ".json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved rendering power data to {output_path}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, base_folder = "output"):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
        ckpt_name = find_ckpt(folder = os.path.join(base_folder, f"L1"))
        if ckpt_name is None:
            raise FileNotFoundError(f"No checkpoint found in {base_folder}")
        # Restore the model from the checkpoint
        ckpt_path = os.path.join(os.path.join(base_folder, f"L1"), ckpt_name)
        ckpt, _= torch.load(ckpt_path)
        gaussians.restore(ckpt, training_args=None)
        # gaussians.append(gaussians)


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # load mask info
        mask_path = os.path.join(base_folder, "highest_levels.pt")
        highest_levels = torch.load(mask_path)
        shs_dcs_path = os.path.join(base_folder, "shs_dcs.pt")
        shs_dcs = torch.load(shs_dcs_path)
        opacities_path = os.path.join(base_folder, "opacities.pt")
        opacities = torch.load(opacities_path)
        
        render_set(dataset.model_path, "test", -1, scene.getTestCameras(), gaussians, pipeline, background, highest_levels, shs_dcs, opacities, base_folder)

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.FR_folder)