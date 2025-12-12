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

import numpy as np
import shutil

from utils.controllable_pruning import controllable_prune, final_test, test_ssim_loss, test_psnr_loss, test_hvs_loss
from utils.display_power import display_power


import math
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
        if args.level > 2:
            gaussians.init_index()
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)



    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ouput_folder = scene.model_path
    log_path = os.path.join(ouput_folder, 'log_final.txt')

    # remove log file if exists
    if os.path.exists(log_path):
        os.remove(log_path)

    # find if there is optimal_prune_ratio.txt in the output folder
    # Check for optimal prune ratio file
    optimal_ratio_file = os.path.join(ouput_folder, "optimal_prune_ratio.txt")

    if os.path.exists(optimal_ratio_file):
        # Read the optimal ratio
        with open(optimal_ratio_file, 'r') as f:
            optimal_ratio = float(f.read().strip())
        
        log_message(f'Optimal prune ratio: {optimal_ratio}', log_path)

    else:
        # load the best checkpoint, record its point number, compare with the current point number to get the optimal prune ratio
        best_checkpoint = os.path.join(ouput_folder, "chkpnt_best.pth")
        # load it using another gs
        best_gaussians = GaussianModel(dataset.sh_degree)
        best_gaussians.training_setup(opt)
        if args.level > 2:
                    best_gaussians.init_index()
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
        if optimal_ratio < 0:
            log_message("Optimal prune ratio is less than 0, set to 0.0", log_path)
            optimal_ratio = 0.0

    # Create search_best directory
    search_best_dir = os.path.join(ouput_folder, "search_best")
    os.makedirs(search_best_dir, exist_ok=True)
    # cp chkpnt_best.pth to search_best directory
    best_checkpoint = os.path.join(ouput_folder, "chkpnt_best.pth")
    # move into search_best directory
    if os.path.exists(best_checkpoint):
        shutil.move(best_checkpoint, os.path.join(search_best_dir, "chkpnt_best.pth"))

    # Prepare for training
    # log the optimal prune ratio
    log_message(f'Optimal prune ratio: {optimal_ratio}', log_path)
    # compute the total prune times
    prune_times = args.prune_iterations // args.prune_interval
    # compute how much to prune each time
    prune_ratio = (optimal_ratio) / prune_times
    # compute how many points to prune each time
    prune_num = int(prune_ratio * gaussians.get_xyz.shape[0])

    ## 2 - Decide Required Q
    if args.level == 2:
        targeted_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=1)
        targeted_hvs = targeted_hvs.cpu().numpy()
        # save as target_hvs.npy in the output folder
        np.save(os.path.join(ouput_folder, 'target_hvs_final.npy'), targeted_hvs)
        gaussians.init_index()
    else:
        # read from target_hvs.npy from L2 directory
        l2_folder = os.path.join(os.path.dirname(ouput_folder), 'L2')
        targeted_hvs = np.load(os.path.join(l2_folder, 'target_hvs_final.npy'))

    args.target_hvs = torch.from_numpy(targeted_hvs).to('cuda').float()

    # 3 - Test the initial quality
    tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=background, dataset=dataset, pooling_size=args.pooling_size)

    # Log initial and target quality metrics
    log_message(f'Initial HVS: {tested_hvs}', log_path)
    log_message(f'Target HVS: {targeted_hvs}', log_path)



    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    pruning_counter = 0
    adjusting_counter = 0
    ## 3- initial display_lambda
    if args.level > 2:
        last_folder = os.path.join(os.path.dirname(ouput_folder), 'L' + str(args.level - 1))
        last_display_lambda = np.load(os.path.join(last_folder, 'final_display_lambda_final.npy'))
        display_lambda = float(last_display_lambda)
    else:
        display_lambda = 1e-3
    update_scale_max = 2.0
    update_scale_min = 1.0
    history_pass = [0.0]
    adjusting_stage = False
    start_pnum = gaussians.get_xyz.shape[0]


    hvs_calc = HVSLoss(uniform_pooling_size=args.pooling_size)

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if pruning_counter >= args.prune_iterations:
            adjusting_stage = True

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
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = 100 * hvs_calc.calc_uniform_loss(image.unsqueeze(0), gt_image.unsqueeze(0))

        if adjusting_stage:
            loss = loss + display_lambda * display_power(image)

        Ll1 = loss.detach()
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
                gaussians, pass_test, quality = controllable_prune(gaussians, scene, pipe, bg, dataset, args, prune_num, ouput_folder, hvs=True, pooling_size=args.pooling_size, skip_test=True)
                log_message(f'[{iteration}] Pruning: {pass_test}, Pnum: {gaussians.get_xyz.shape[0]}, PS: {args.pooling_size}', log_path)
            if pruning_counter == args.prune_iterations:
                print(f"\n[ITER {iteration}] Final Test")
                # print final P ratio
                log_message(f'Final P ratio: {gaussians.get_xyz.shape[0] / start_pnum}, optimal ratio {optimal_ratio}', log_path)
            
            pruning_counter += 1

            # Adjusting
            if adjusting_stage:
                # Adjust display lambda
                if adjusting_counter % args.adjust_interval == 0 and adjusting_counter < args.adjust_iterations:
                    update_scale = update_scale_min + (update_scale_max - update_scale_min) * (math.cos(math.pi * (adjusting_counter) / args.adjust_iterations) + 1) / 2
                    tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset, pooling_size=args.pooling_size)
                    hvs_pass = tested_hvs <= args.target_hvs
                    pass_test = hvs_pass
                    if pass_test:
                        history_pass.append(display_lambda)
                        display_lambda *= update_scale
                    else:
                        display_lambda /= update_scale
                    log_message(f'[{iteration}] Meet: {pass_test}, Display Lambda: {display_lambda}, HVS: {tested_hvs}, PS: {args.pooling_size}', log_path)
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


    # New Parameters for Masking
    parser.add_argument('--adjust_interval', type=int, default = 1000)
    parser.add_argument('--adjust_iterations', type=int, default = 25000)
    parser.add_argument('--prune_interval', type=int, default = 1000)
    parser.add_argument('--prune_iterations', type=int, default = 25000)
    parser.add_argument('--pooling_size', type=int, default = 1)
    parser.add_argument('--level', type=int, default = 1)


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
