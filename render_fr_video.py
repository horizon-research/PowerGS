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

import cv2
import numpy as np

# Make sure you have this import or something equivalent for generating interpolated views
from subjective_exp.inter_pose_same_res import generate_interpolated_views


def search_file(folder, pattern):
    """
    Utility to search for files in `folder` matching `pattern`.
    """
    search_path = os.path.join(folder, pattern)
    files = glob.glob(search_path, recursive=False)
    return files


def find_ckpt(folder):
    """
    Looks in `folder` for a single checkpoint file named `chkpnt*.pth`.
    Raises an exception if multiple or none are found.
    Returns the base name of the found checkpoint.
    """
    pattern = "chkpnt*.pth"
    files_found = search_file(folder, pattern)

    if files_found:
        if len(files_found) == 1:
            print("Found exactly one matching file:", files_found[0])
            return os.path.basename(files_found[0])
        else:
            raise RuntimeError(f"Found multiple matching files ({len(files_found)}): {files_found}")
    else:
        raise FileNotFoundError(f"No files matching the pattern '{pattern}' were found in '{folder}'!")

def add_circle(rendering, radius=50, thickness=1):
    """
    rendering: [H,W,3], torch.uint8 on CUDA
    Overlays a circular ring in the center. 
    Adjust radius/thickness as needed.
    """
    _, H, W = rendering.shape
    device = rendering.device
    center_x = W // 2
    center_y = H // 2

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    outer_sq = (radius + thickness)**2
    inner_sq = (radius - thickness)**2

    mask = (((xx - center_x)**2 + (yy - center_y)**2) <= outer_sq) & \
           (((xx - center_x)**2 + (yy - center_y)**2) >= inner_sq)
           
    # import ipdb; ipdb.set_trace()

    red_color = torch.tensor([1,0,0], dtype=torch.float, device=device)
    
    # import ipdb; ipdb.set_trace()
    rendering[:, mask] = red_color.unsqueeze(1)
    return rendering


def render_set(
    name,
    views,
    gaussians,
    pipeline,
    background,
    highest_levels,
    shs_dcs,
    opacities,
    base_folder
):
    """
    Renders the given `views` with foveated rendering, saves the frames as a video,
    and records power metrics in a JSON file.
    """

    # You can customize gaze samples. For demonstration, we fix them as (0.5, 0.5).
    # If you want a 1:1 mapping (one gaze sample per view), define gaze_samples = [ (0.5, 0.5) ].
    gaze_samples = [(0.5, 0.5) for _ in views]

    # # Prepare output directories
    # render_dir = os.path.join(base_folder, name, "ph", "renders_fov_0.5")
    # makedirs(render_dir, exist_ok=True)

    # # Path for JSON with power stats
    # rendering_powers_path = os.path.join(base_folder, name, "ph", "FR_rendering_powers_0.5.json")
    # makedirs(os.path.dirname(rendering_powers_path), exist_ok=True)

    # We will also write out an MP4 video of the rendered frames
    video_path = os.path.join("./videos/", f"{name}_render.mp4")

    # Set up a list to track per-view power metrics
    rendering_powers = []

    shs_dcs = shs_dcs.cuda()
    highest_levels = highest_levels.cuda()
    opacities = opacities.cuda()

    # --- Prepare for video writing ---
    # Render a first frame to figure out size
    # (If you have at least one view)
    if len(views) == 0:
        print("No views to render!")
        return

    first_view = views[0]
    first_gaze = gaze_samples[0]
    with torch.no_grad():
        first_renderings = render(
            viewpoint_camera=first_view,
            pc=gaussians,
            pipe=pipeline,
            bg_color=background,
            alpha=0.05,
            gazeArray=torch.tensor([first_gaze[0], first_gaze[1]]).float().cuda(),
            highest_levels=highest_levels,
            shs_dcs=shs_dcs,
            opacities=opacities,
            fr=True
        )
    first_render = first_renderings["render"]
    frame_height, frame_width = first_render.shape[1], first_render.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 90, (frame_width, frame_height))  # 30 FPS (adjust as needed)

    # --- Main rendering loop ---
    for idx, (view, gaze) in enumerate(tqdm(zip(views, gaze_samples), total=len(views), desc="Foveated Rendering Progress")):
        with torch.no_grad():
            renderings = render(
                viewpoint_camera=view,
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                alpha=0.05,
                gazeArray=torch.tensor([gaze[0], gaze[1]]).float().cuda(),
                highest_levels=highest_levels,
                shs_dcs=shs_dcs,
                opacities=opacities,
                fr=True
            )

        # Retrieve the color image
        rendering = renderings["render"]
        
        
        # Optionally add a circle overlay
        rendering = add_circle(rendering)

        # # Save as PNG as well (optional)
        # out_path = os.path.join(render_dir, f'{idx:05d}_{gaze[0]}_{gaze[1]}.png')
        # torchvision.utils.save_image(rendering, out_path)

        # Convert tensor to BGR for video
        rendering_np = rendering.permute(1, 2, 0).cpu().numpy()
        rendering_np = np.clip(rendering_np, 0, 1)
        rendering_np_bgr = (rendering_np * 255).astype(np.uint8)[..., ::-1]
        

        video_writer.write(rendering_np_bgr)

        # Power modeling
        # result = fov_hardware_power_modeling(renderings["point_statistics"], renderings["pixel_statistics"])
        # rendering_powers.append(result)

    # Release video writer
    video_writer.release()
    print(f"Video saved to {video_path}")

    # # Compute mean metrics over all views
    # metrics = [
    #     "total_power", "total_dram_power", "total_sram_power", "total_flops_power",
    #     "preprocessing_power", "sorting_power", "rendering_power"
    # ]
    # mean_metrics = {
    #     f"mean_{metric}": sum(r[metric] for r in rendering_powers) / len(rendering_powers)
    #     for metric in metrics
    # }

    # # Create a JSON structure
    # output_data = {
    #     "mean_metrics": mean_metrics,
    #     "per_view_metrics": rendering_powers
    # }

    # # Save JSON
    # with open(rendering_powers_path, "w") as f:
    #     json.dump(output_data, f, indent=4)

    # print(f"Saved rendering power data to {rendering_powers_path}")


def render_sets(
    dataset : ModelParams,
    iteration : int,
    pipeline : PipelineParams,
    skip_train : bool,
    skip_test : bool,
    start_f : int,
    end_f : int,
    inter_num : int,
    base_folder = "output",
    scene_name = "scene",
    method_name = "method"
):
    """
    Loads the model, selects test cameras, interpolates them, then renders
    foveated frames (and a video) for that subset of cameras.
    """
    with torch.no_grad():
        # Load the model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

        # Find the checkpoint
        ckpt_name = find_ckpt(folder=os.path.join(base_folder, "levels/L1"))
        ckpt_path = os.path.join(base_folder, "levels/L1", ckpt_name)
        ckpt, _ = torch.load(ckpt_path, weights_only=False)
        gaussians.restore(ckpt, training_args=None)

        # Prepare background color
        bg_color = [1,1,1] if dataset.white_background else [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load precomputed mask info
        mask_path = os.path.join(base_folder, "levels/highest_levels.pt")
        highest_levels = torch.load(mask_path, weights_only=False)

        shs_dcs_path = os.path.join(base_folder, "levels/shs_dcs.pt")
        shs_dcs = torch.load(shs_dcs_path, weights_only=False)

        opacities_path = os.path.join(base_folder, "levels/opacities.pt")
        opacities = torch.load(opacities_path, weights_only=False)

        # Get test cameras
        test_cams = scene.getTestCameras()

        # Slice test cameras
        subset_cams = test_cams[start_f:end_f]

        # Generate interpolated views
        views = generate_interpolated_views(subset_cams, inter_num)
        if not views:
            print("No views generated after interpolation!")
            return
        views = views[:1350]  # Limit to 1350 views if needed
        
        render_name = f"{scene_name}_{start_f}_{end_f}_{inter_num}_{method_name}"

        # Actually render them (foveated)
        render_set(
            name=render_name,
            views=views,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
            highest_levels=highest_levels,
            shs_dcs=shs_dcs,
            opacities=opacities,
            base_folder=base_folder
        )


if __name__ == "__main__":
    # Setup command line argument parser
    parser = ArgumentParser(description="Foveated Rendering script with video export")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # Additional arguments for selecting camera range and interpolation
    parser.add_argument("--start_f", default=10, type=int,
                        help="Starting frame index for subset of test cameras")
    parser.add_argument("--end_f", default=13, type=int,
                        help="Ending frame index for subset of test cameras")
    parser.add_argument("--inter_num", default=900, type=int,
                        help="Number of interpolated views to generate")
    parser.add_argument("--scene_name", default="scene", type=str,
                        help="Scene name for output file naming")
    parser.add_argument("--method_name", default="method", type=str,
                        help="Method name for output file naming")
    parser.add_argument("--FR_folder", type=str, default="output",
                        help="Folder for saving FR outputs")

    args = get_combined_args(parser)
    print("Foveated Rendering from model:", args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Execute rendering
    render_sets(
        dataset=model.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        start_f=args.start_f,
        end_f=args.end_f,
        inter_num=args.inter_num,
        base_folder=args.FR_folder,
        scene_name=args.scene_name,
        method_name=args.method_name
    )
