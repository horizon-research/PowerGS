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
import sys
curr_file_path = os.path.abspath(__file__)
curr_dir_path = os.path.dirname(curr_file_path)
sys.path.append(os.path.join(curr_dir_path, "../"))
from utils.hvs_loss_calc import HVSLoss
from utils.display_power import display_power
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import math
import json
import shutil


from utils.controllable_pruning import controllable_prune, final_test, test_ssim_loss, test_psnr_loss, test_display_power, test_render_power, test_hvs_loss

import numpy as np

def log_message(message, file_path='training_log.log'):
    with open(file_path, 'a') as log_file:
        log_file.write(message + '\n')
        log_file.flush()

def finetune_a_search(recipe):
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    update_scale_max = 2.0
    update_scale_min = 1.0
    adjust_interval = 50

    first_iter = 30000 + 1
    history_pass = [0.0]
    
    iterations = recipe['iterations']
    gaussians = recipe['gaussians']
    scene = recipe['scene']
    pipe = recipe['pipe']
    display_lambda = recipe['display_lambda']
    background = recipe['background']
    debug_from = recipe['debug_from']
    testing_iterations = recipe['testing_iterations']
    tb_writer = recipe['tb_writer']
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    log_path = recipe['log_path']   
    dataset = recipe['dataset']
    hvs_calc = recipe['hvs_calc']

    adjust_iterations = iterations - 500
    iterations = iterations + 30000
    adjusting_counter = 0

    has_pass = False
    

    progress_bar = tqdm(range(first_iter, iterations + 1), desc="Training progress")
    for iteration in range(first_iter, iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, masking=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        loss = 100 * hvs_calc.calc_uniform_loss(image.unsqueeze(0), gt_image.unsqueeze(0))
        Ll1 = loss.detach()
        
        loss = loss.mean() + display_lambda * display_power(image)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Optimizer step
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            # Adjust display lambda
            if adjusting_counter % adjust_interval == 0 and adjusting_counter < adjust_iterations:
                update_scale = update_scale_min + (update_scale_max - update_scale_min) * (math.cos(math.pi * (adjusting_counter) / adjust_iterations) + 1) / 2
                tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset, pooling_size=args.pooling_size)
                hvs_pass = tested_hvs  <= args.target_hvs
                pass_test = hvs_pass
                if pass_test:
                    history_pass.append(display_lambda)
                    display_lambda *= update_scale
                    has_pass = True
                else:
                    display_lambda /= update_scale
                log_message(f'[{iteration}] Meet: {pass_test}, Display Lambda: {display_lambda}, HVS: {tested_hvs}, PS: {args.pooling_size}', log_path)
            if adjusting_counter == adjust_iterations:
                # log current display_lambda and every lambda in history_pass
                log_message(f"\n[ITER {iteration}] Final Test, Display Lambda: {display_lambda}", log_path)
                for idx, pass_lambda in enumerate(history_pass):
                    log_message(f"[ITER {iteration}] Final Test, History Pass {idx}: {pass_lambda}", log_path)
                # max_pass_display_lambda = max(history_pass)
                last_pass_display_lambda = history_pass[-1]
                log_message(f"\n[ITER {iteration}] Final Test, last Maximum Pass Test: {last_pass_display_lambda}", log_path)
                display_lambda = last_pass_display_lambda
            adjusting_counter += 1
    
    return gaussians, display_lambda, has_pass

def smooth_filter(total_powers, passes):
    # Apply 1D filter [0.25, 0.5, 0.25] to smooth total powers, respecting pass/fail status
    filtered_powers = total_powers.copy()
    for i in range(1, len(total_powers) - 1):
        if not passes[i]:
            continue
            
        weights = [0.0, 0.0, 0.0]  # Left, Center, Right weights
        total_weight = 0.0
        
        # Check left neighbor
        if passes[i-1]:
            weights[0] = 0.25
            total_weight += 0.25
            
        # Current point always contributes
        weights[1] = 0.5
        total_weight += 0.5
        
        # Check right neighbor
        if passes[i+1]:
            weights[2] = 0.25
            total_weight += 0.25
            
        # Normalize weights to sum to 1
        weights = [w/total_weight for w in weights]
        
        # Calculate weighted average
        filtered_powers[i] = (weights[0] * total_powers[i-1] + 
                            weights[1] * total_powers[i] + 
                            weights[2] * total_powers[i+1])
    
    # Handle endpoints
    if passes[0] and passes[1]:
        filtered_powers[0] = 0.67 * total_powers[0] + 0.33 * total_powers[1]
    
    if passes[-1] and passes[-2]:
        filtered_powers[-1] = 0.67 * total_powers[-1] + 0.33 * total_powers[-2]

    return filtered_powers



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        if args.level > 2:
            gaussians.init_index()
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    ouput_folder = scene.model_path
    log_path = os.path.join(ouput_folder, 'log.txt')

    # rm log file

    if os.path.exists(log_path):
        os.remove(log_path)


    # Prepare for training
    
    coarse_prune_ratio = 0.15
    coarse_search_num = 7
    fine_search_num = 5

    per_finetune_budget = 5000 # 5000 x relative_prune_ratio
    min_iter = 1500

    first_iter = 5000

    original_num = gaussians.get_xyz.shape[0] 
    coarse_prune_num = int(gaussians.get_xyz.shape[0] * coarse_prune_ratio)

    if args.level == 2:
        targeted_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=1)
        targeted_hvs = targeted_hvs.cpu().numpy()
        # save as target_hvs.npy in the output folder
        np.save(os.path.join(ouput_folder, 'target_hvs.npy'), targeted_hvs)
        gaussians.init_index()
    else:
        # read from target_hvs.npy from L2 directory
        l2_folder = os.path.join(os.path.dirname(ouput_folder), 'L2')
        targeted_hvs = np.load(os.path.join(l2_folder, 'target_hvs.npy'))

    # 3 - Test the initial quality
    tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=args.pooling_size)
    ## 4- initial display_lambda
    if args.level > 2:
        last_folder = os.path.join(os.path.dirname(ouput_folder), 'L' + str(args.level - 1))
        last_display_lambda = np.load(os.path.join(last_folder, 'final_display_lambda.npy'))
        display_lambda = float(last_display_lambda)
    else:
        display_lambda = 1e-3


    args.target_hvs = targeted_hvs.item()

    # Log initial and target quality metrics
    log_message(f'Initial HVS: {tested_hvs}', log_path)
    log_message(f'Target HVS: {targeted_hvs}', log_path)
    # coarse search
    # mkdir
    os.makedirs(scene.model_path + "/coarse", exist_ok=True)
    coarse_total_powers = []
    coarse_display_powers = []
    coarse_render_powers = []
    coarse_display_lambdas = []
    coarse_prune_ratios = []
    coarse_hvss = []
    coarse_passes = []

    hvs_calc = HVSLoss(uniform_pooling_size=args.pooling_size)
    for idx in range(coarse_search_num):
        recipe = {
            "iterations": None,
            "gaussians": gaussians,
            "scene": scene,
            "pipe": pipe,
            "background": background,
            "debug_from": debug_from,
            "testing_iterations": testing_iterations,
            "tb_writer": tb_writer,
            "log_path": log_path,
            "dataset": dataset,
            "display_lambda": display_lambda,
            "hvs_calc": hvs_calc
        }

        tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=args.pooling_size)
        # Log initial quality metrics
        log_message(f"======================== [Coarse {idx}] ========================", log_path)
        log_message(f'Initial HVS: {tested_hvs}', log_path)
        log_message(f'Target HVS: {targeted_hvs}', log_path)
        if idx == 0:
            recipe["iterations"] = first_iter
            pre_ratio = 1
        else:
            pre_ratio = 1 - coarse_prune_ratio * (idx-1)
            recipe["iterations"] = max( int(per_finetune_budget * (coarse_prune_ratio / pre_ratio)), min_iter)
        log_message(f"Fine search idx: {idx}, Prune Ratio: {coarse_prune_ratio}, Pre Ratio: {pre_ratio}, Iteration: {recipe['iterations']}", log_path)
        gaussians, display_lambda, has_pass = finetune_a_search(recipe)

        tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=args.pooling_size)
        display_power = test_display_power(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, scale=3) * args.display_scale
        render_power = test_render_power(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset)
        total_power = display_power + render_power

        log_message(f"HVS: {tested_hvs}", log_path)
        log_message(f"Display Power: {display_power:.4f}", log_path)
        log_message(f"Render Power: {render_power:.4f}", log_path)
        log_message(f"Total Power: {total_power:.4f}", log_path)
        torch.save((gaussians.capture(), -1), scene.model_path + "/coarse/" + str(idx) + ".pth")

        # import ipdb; ipdb.set_trace()
        coarse_total_powers.append(total_power.item())
        coarse_display_powers.append(display_power.item())
        coarse_render_powers.append(render_power)
        coarse_display_lambdas.append(display_lambda)
        coarse_prune_ratios.append(coarse_prune_ratio * idx)
        coarse_hvss.append(tested_hvs.item())
        coarse_passes.append(has_pass)
        if idx < coarse_search_num - 1:
            gaussians, pass_test, quality = controllable_prune(gaussians, scene, pipe, background, dataset, args, \
                                                            coarse_prune_num, ouput_folder, hvs=False, pooling_size=1, no_save=True, skip_test=True)
    
    # save all coarse search results to a json file

    output_data = {
        "total_powers": coarse_total_powers,
        "display_powers": coarse_display_powers,
        "render_powers": coarse_render_powers,
        "display_lambdas": coarse_display_lambdas,
        "prune_ratios": coarse_prune_ratios,
        "hvss": coarse_hvss
    }
    output_path = scene.model_path + "/coarse_search_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)


    coarse_total_powers_smooth = smooth_filter(coarse_total_powers, coarse_passes)
    # print total powers
    log_message(f"Total Powers: {coarse_total_powers}", log_path)
    # print the passes
    log_message(f"Passes: {coarse_passes}", log_path)
    # print the filtered total powers
    log_message(f"Filtered Total Powers: {coarse_total_powers_smooth}", log_path)

    # Find best coarse search result with lowest power among passing results
    passing_powers = [p if pass_test else float('inf') for p, pass_test in zip(coarse_total_powers_smooth, coarse_passes)]
    min_power_idx = passing_powers.index(min(passing_powers))
    start_idx = 0 if min_power_idx == 0 else min_power_idx - 1

    left_ratio = 1 - coarse_prune_ratios[start_idx]
    if left_ratio < 2 * coarse_prune_ratio:
        fine_prune_ratio = left_ratio / (fine_search_num + 1)
    else:
        fine_prune_ratio = 2 * coarse_prune_ratio / (fine_search_num + 1)
    
    

    # Create fine search output directory
    os.makedirs(scene.model_path + "/fine", exist_ok=True)
    
    # Initialize tracking variables for fine search
    fine_total_powers = []
    fine_display_powers = []
    fine_render_powers = []
    fine_display_lambdas = []
    fine_prune_ratios = []
    fine_hvss = []
    fine_passes = []
    display_lambda = coarse_display_lambdas[start_idx]
    
    # log about finesearch is starting and the best idx is xxx
    log_message(f"======================== [Fine Search] ========================", log_path)
    log_message(f"Starting from coarse search idx: {start_idx}", log_path)
    
    # Perform fine search iterations
    for idx in range(fine_search_num):
        # Load best coarse checkpoint
        shift_model = False
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.init_index()
        if fine_prune_ratio * (idx+1) <= coarse_prune_ratio:
            checkpoint_path = scene.model_path + "/coarse/" + str(start_idx) + ".pth"
        else:
            checkpoint_path = scene.model_path + "/coarse/" + str(start_idx+1) + ".pth"
            shift_model = True
            # log that the model is shifted
            log_message(f"Fine search idx: {idx}, Shift model from {start_idx} to {start_idx+1}", log_path)
        model_params, _ = torch.load(checkpoint_path)
        gaussians.restore(model_params, opt)
        recipe = {
            "iterations": None,
            "gaussians": gaussians,
            "scene": scene,
            "pipe": pipe,
            "background": background,
            "debug_from": debug_from,
            "testing_iterations": testing_iterations,
            "tb_writer": tb_writer,
            "log_path": log_path,
            "dataset": dataset,
            "display_lambda": display_lambda,
            "hvs_calc": hvs_calc
        }
        prune_num = int(original_num  * fine_prune_ratio * (idx+1))
        if prune_num > coarse_prune_num:
            assert shift_model
            prune_num = prune_num - coarse_prune_num
            # print the shift
            log_message(f"Fine search idx: {idx}, Shift model from {start_idx} to {start_idx+1}", log_path)
        # if gaussians.get_xyz.shape[0] > fine_prune_num:
            
        gaussians, pass_test, quality = controllable_prune(gaussians, scene, pipe, background, dataset, args,
                                                        prune_num, ouput_folder, hvs=False, pooling_size=1, 
                                                        no_save=True, skip_test=True)
        # else:
        #     log_message(f"Fine search idx: {idx}, point number {gaussians.get_xyz.shape[0]} is less than {fine_prune_num}, skip pruning", log_path)
        #     break
        tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=args.pooling_size)
        
        # Log initial metrics for this fine iteration
        log_message(f"======================== [Fine {idx}] ========================", log_path)
        # Log initial and target quality metrics
        log_message(f'Initial HVS: {tested_hvs}', log_path)
        log_message(f'Target HVS: {targeted_hvs}', log_path)
        # compute pre ratio
        real_fine_prune_ratio = fine_prune_ratio * (idx+1)
        if shift_model:
            pre_ratio = 1 - (start_idx + 1) * coarse_prune_ratio
            real_fine_prune_ratio = real_fine_prune_ratio - coarse_prune_ratio
            print(f"Fine search idx: {idx}, Shift model from {start_idx} to {start_idx+1}")
        else:
            pre_ratio = 1 - start_idx * coarse_prune_ratio
        recipe["iterations"] = max( int(per_finetune_budget * (real_fine_prune_ratio / pre_ratio)), min_iter)

        # log the current ratio, prune ratio, per_ratio, and iteration number
        log_message(f"Fine search idx: {idx}, Prune Ratio: {fine_prune_ratio * (idx+1)}, Pre Ratio: {pre_ratio}, Iteration: {recipe['iterations']}", log_path)
        # log the real prune ratio gs.xyz / original count
        log_message(f"Fine search idx: {idx}, Real Prune Ratio: {1 - gaussians.get_xyz.shape[0] / original_num}", log_path)
        gaussians, display_lambda, has_pass = finetune_a_search(recipe)

        # Test and log final metrics
        tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=args.pooling_size)
        display_power = test_display_power(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, scale=3) * args.display_scale
        render_power = test_render_power(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset)
        total_power = display_power + render_power

        log_message(f"HVS: {tested_hvs}", log_path)
        log_message(f"Display Power: {display_power:.4f}", log_path)
        log_message(f"Render Power: {render_power:.4f}", log_path)
        log_message(f"Total Power: {total_power:.4f}", log_path)

        
        # Save checkpoint
        torch.save((gaussians.capture(), -1), scene.model_path + "/fine/" + str(idx) + ".pth")

        fine_total_powers.append(total_power.item())
        fine_display_powers.append(display_power.item())
        fine_render_powers.append(render_power)
        fine_display_lambdas.append(display_lambda)
        fine_prune_ratios.append(fine_prune_ratio * (idx+1) + coarse_prune_ratio * start_idx)   
        fine_hvss.append(tested_hvs.item())
        fine_passes.append(has_pass)


    # Save fine search results
    fine_output_data = {
        "total_powers": fine_total_powers,
        "display_powers": fine_display_powers,
        "render_powers": fine_render_powers,
        "display_lambdas": fine_display_lambdas,
        "prune_ratios": fine_prune_ratios,
        "hvss": fine_hvss
    }
    fine_output_path = scene.model_path + "/fine_search_results.json"
    with open(fine_output_path, "w") as f:
        json.dump(fine_output_data, f, indent=4)



    # Find best model across all coarse and fine search results
    all_powers = []
    all_paths = []
    all_metrics = []
    
    # Gather coarse search results
    for idx, (power, passes) in enumerate(zip(coarse_total_powers, coarse_passes)):
        if passes:
            all_powers.append(power)
            all_paths.append(("coarse", idx))
            all_metrics.append({
                "total_power": coarse_total_powers[idx],
                "display_power": coarse_display_powers[idx],
                "render_power": coarse_render_powers[idx],
                "display_lambda": coarse_display_lambdas[idx],
                "prune_ratio": coarse_prune_ratios[idx],
                "hvs": coarse_hvss[idx]
            })
    
    # Gather fine search results
    for idx, (power, passes) in enumerate(zip(fine_total_powers, fine_passes)):
        if passes:
            all_powers.append(power)
            all_paths.append(("fine", idx))
            all_metrics.append({
                "total_power": fine_total_powers[idx],
                "display_power": fine_display_powers[idx],
                "render_power": fine_render_powers[idx],
                "display_lambda": fine_display_lambdas[idx],
                "prune_ratio": fine_prune_ratios[idx],
                "hvs": fine_hvss[idx]
            })

    if not all_powers:
        log_message(f"No passing results found for scene", log_path)
        return

    # Find best result (lowest power among passing results)
    min_power_idx = all_powers.index(min(all_powers))
    search_type, model_idx = all_paths[min_power_idx]
    best_metrics = all_metrics[min_power_idx]
    
    best_checkpoint_path = scene.model_path + f"/{search_type}/" + str(model_idx) + ".pth"
    log_message(f"Best Model Checkpoint ({search_type}): {best_checkpoint_path}", log_path)

    # Print best metrics
    log_message(f"======================== [Best Model] ========================", log_path)
    log_message(f"Total Power: {best_metrics['total_power']:.4f}", log_path)
    log_message(f"Display Power: {best_metrics['display_power']:.4f}", log_path)
    log_message(f"Render Power: {best_metrics['render_power']:.4f}", log_path)
    log_message(f"Display Lambda: {best_metrics['display_lambda']:.4f}", log_path)
    log_message(f"Prune Ratio: {best_metrics['prune_ratio']:.4f}", log_path)
    log_message(f"hvs: {best_metrics['hvs']:.4f}", log_path)

    # Move best checkpoint to root folder
    best_model_path = os.path.join(scene.model_path, "chkpnt_best.pth")
    shutil.copy2(best_checkpoint_path, best_model_path)
    log_message(f"Best model saved to: {best_model_path}", log_path)

    # import ipdb; ipdb.set_trace()
    np.save(os.path.join(scene.model_path, 'final_display_lambda.npy'), best_metrics['display_lambda'])


    # remove coarse and fine search folder
    shutil.rmtree(scene.model_path + "/coarse")
    shutil.rmtree(scene.model_path + "/fine")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--pooling_size', type=int, default = 1)
    parser.add_argument('--level', type=int, default = 1)
    parser.add_argument("--display_scale", type=int, default = 1)


    # New Parameters for Pruning
    parser.add_argument('--q_required', type=float, default = 0.99)



    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    opt = op.extract(args)
    # assert args.checkpoint_iterations is not None and len(args.checkpoint_iterations) > 0, "Please provide checkpoint iterations"
    # opt.iterations = args.checkpoint_iterations[-1]
    # import ipdb; ipdb.set_trace()
    opt.position_lr_init = opt.position_lr_init * 0.1

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
