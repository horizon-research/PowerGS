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
import shutil
curr_file_path = os.path.abspath(__file__)
curr_dir_path = os.path.dirname(curr_file_path)
sys.path.append(os.path.join(curr_dir_path, "../"))

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
import numpy as np

from utils.controllable_pruning import controllable_prune, final_test, test_ssim_loss, test_psnr_loss
from utils.display_power import display_power
def log_message(message, file_path='training_log.log'):
    with open(file_path, 'a') as log_file:
        log_file.write(message + '\n')
        log_file.flush()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    ouput_folder = scene.model_path
    log_path = os.path.join(ouput_folder, 'log_final.txt')
    # remove existing log file
    if os.path.exists(log_path):
        os.remove(log_path)

    # find if there is optimal_prune_ratio.txt in the output folder
    # Check for optimal prune ratio file
    optimal_ratio_file = os.path.join(ouput_folder, "optimal_prune_ratio.txt")


    if os.path.exists(optimal_ratio_file):
        # Read the optimal ratio
        with open(optimal_ratio_file, 'r') as f:
            optimal_ratio = float(f.read().strip())
        
            
        log_message(f"Using optimal prune ratio from fitting: {optimal_ratio}", log_path)
    else:
        # load the best checkpoint, record its point number, compare with the current point number to get the optimal prune ratio
        best_checkpoint = os.path.join(ouput_folder, "chkpnt_best.pth")
        # load it using another gs
        best_gaussians = GaussianModel(dataset.sh_degree)
        best_gaussians.training_setup(opt)
        if os.path.exists(best_checkpoint):
            (model_params, _) = torch.load(best_checkpoint)
            best_gaussians.restore(model_params, opt)
        else:
            # check if there is search_best directory
            search_best_dir = os.path.join(ouput_folder, "search_best")
            best_checkpoint = os.path.join(search_best_dir, "chkpnt_best.pth")
            if os.path.exists(best_checkpoint):
                (model_params, _) = torch.load(best_checkpoint)
                best_gaussians.restore(model_params, opt)
        
        optimal_ratio = 1 - best_gaussians.get_xyz.shape[0] / gaussians.get_xyz.shape[0]

        # remove the best_gaussians
        del best_gaussians
        log_message("No optimal prune ratio found, INFER FROM SEARCH BEST", log_path)

    # Create search_best directory
    search_best_dir = os.path.join(ouput_folder, "search_best")
    os.makedirs(search_best_dir, exist_ok=True)
    # cp chkpnt_best.pth to search_best directory
    best_checkpoint = os.path.join(ouput_folder, "chkpnt_best.pth")
    # move into search_best directory
    if os.path.exists(best_checkpoint):
        shutil.move(best_checkpoint, os.path.join(search_best_dir, "chkpnt_best.pth"))


    # log the optimal prune ratio
    log_message(f'Optimal prune ratio: {optimal_ratio}', log_path)
    # compute the total prune times
    prune_times = args.prune_iterations // args.prune_interval
    # compute how much to prune each time
    prune_ratio = (optimal_ratio) / prune_times
    # compute how many points to prune each time
    prune_num = int(prune_ratio * gaussians.get_xyz.shape[0])

    tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset)
    tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset)
    target_ssim = tested_ssim * args.q_required
    target_psnr = tested_psnr * args.q_required

    args.target_psnr = target_psnr
    args.target_ssim = target_ssim
    
    # Log initial quality metrics
    log_message(f"Initial SSIM: {tested_ssim:.4f}", log_path)
    log_message(f"Initial PSNR: {tested_psnr:.4f}", log_path) 
    log_message(f"Target SSIM: {target_ssim:.4f}", log_path)
    log_message(f"Target PSNR: {target_psnr:.4f}", log_path)

    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    pruning_counter = 0
    adjusting_counter = 0
    display_lambda = 1e-3
    update_scale_max = 2.0
    update_scale_min = 1.0
    history_pass = [0.0]

    adjusting_stage = False
    start_pnum = gaussians.get_xyz.shape[0]
    for iteration in range(first_iter, opt.iterations + 1):        
        
        if pruning_counter >= args.prune_iterations:
            adjusting_stage = True
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

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if adjusting_stage:
            # import ipdb; ipdb.set_trace()
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
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # Pruning
            if pruning_counter % args.prune_interval == 0 and pruning_counter < args.prune_iterations:
                print(f"\n[ITER {iteration}] Pruning {prune_num} points")
                gaussians, pass_test, quality = controllable_prune(gaussians, scene, pipe, bg, dataset, args, prune_num, ouput_folder, hvs=False, skip_test=True, pooling_size=1)
                log_message(f'[{iteration}] Pruning: {pass_test}, Pnum: {gaussians.get_xyz.shape[0]}', log_path)
            if pruning_counter == args.prune_iterations:
                print(f"\n[ITER {iteration}] Final Test")
                # print final P ratio
                log_message(f'Final P ratio: {gaussians.get_xyz.shape[0] / start_pnum}, optimal ratio {optimal_ratio}', log_path)
            

            # Adjusting
            if adjusting_stage:
                # Adjust display lambda
                if adjusting_counter % args.adjust_interval == 0 and adjusting_counter < args.adjust_iterations:
                    update_scale = update_scale_min + (update_scale_max - update_scale_min) * (math.cos(math.pi * (adjusting_counter) / args.adjust_iterations) + 1) / 2
                    tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
                    tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
                    psnr_pass = tested_psnr >= args.target_psnr
                    ssim_pass = tested_ssim >= args.target_ssim
                    pass_test = psnr_pass & ssim_pass
                    if pass_test:
                        history_pass.append(display_lambda)
                        display_lambda *= update_scale
                    else:
                        display_lambda /= update_scale
                    log_message(f'[{iteration}] Meet: {pass_test}, Display Lambda: {display_lambda}, SSIM: {tested_ssim}, PSNR: {tested_psnr}', log_path)
                if adjusting_counter == args.adjust_iterations:
                    # log current display_lambda and every lambda in history_pass
                    log_message(f"\n[ITER {iteration}] Final Test, Display Lambda: {display_lambda}", log_path)
                    for idx, pass_lambda in enumerate(history_pass):
                        log_message(f"[ITER {iteration}] Final Test, History Pass {idx}: {pass_lambda}", log_path)
                    # max_pass_display_lambda = max(history_pass)
                    last_pass_display_lambda = history_pass[-1]
                    log_message(f"\n[ITER {iteration}] Final Test, last Pass Test: {last_pass_display_lambda}", log_path)
                    # save to output folder as final_display_lambda.npy
                    np.save(os.path.join(ouput_folder, 'final_display_lambda_final.npy'), last_pass_display_lambda)
                adjusting_counter += 1
            pruning_counter += 1

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


    # New Parameters for Pruning
    parser.add_argument('--prune_interval', type=int, default = 1000)
    parser.add_argument('--prune_iterations', type=int, default = 25000)
    parser.add_argument('--adjust_interval', type=int, default = 1000)
    parser.add_argument('--adjust_iterations', type=int, default = 25000)
    parser.add_argument('--q_required', type=float, default = 0.99)



    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    opt = op.extract(args)
    assert args.checkpoint_iterations is not None and len(args.checkpoint_iterations) > 0, "Please provide checkpoint iterations"
    opt.iterations = args.checkpoint_iterations[-1]
    # import ipdb; ipdb.set_trace()
    opt.position_lr_init = opt.position_lr_init * 0.1

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
