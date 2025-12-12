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
from gaussian_power_analysis.power_model import hardware_power_modeling
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    rendering_powers_path = os.path.join(model_path, "ours_{}".format(iteration)+"_rendering_powers")

    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    rendering_powers = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        renderings = render(view, gaussians, pipeline, background)
        rendering = renderings["render"]

        result = hardware_power_modeling(renderings["point_statistics"], renderings["pixel_statistics"])
        # import ipdb; ipdb.set_trace()
        # renderings["point_statistics"]
        # renderings["pixel_statistics"]
        # print(result)
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
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


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)