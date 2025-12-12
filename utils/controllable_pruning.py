import torch
from gaussian_renderer import render, network_gui
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
import os
from utils.hvs_loss_calc import HVSLoss
from utils.display_power import display_power_metric_boost

from gaussian_power_analysis import render as render_power
from gaussian_power_analysis.power_model import hardware_power_modeling

def metric_pruning(gaussians, prune_num=None, scene=None ,pipe=None, bg=None, dataset=None):
    view_stacks = scene.getTrainCameras().copy()
    pnum = gaussians.get_xyz.shape[0]
    metrics = torch.zeros((pnum, 1), device="cuda")
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            contibs = render_pkg["dominance_count"].unsqueeze(1).float()
            gs_count = render_pkg["intersect_count"].unsqueeze(1).float()
            cur_metric = contibs / (gs_count + 1e-7)
            cur_metric[gs_count < 1] = 0
            # max over all views
            metrics[metrics < cur_metric] = cur_metric[metrics < cur_metric]

    _, indices = torch.sort(metrics, descending=False, dim=0)
    # make sure we are not pruning all the points
    assert prune_num < pnum
    mask_index = indices[:prune_num]
    # make a mask vector that is 1 for the points to prune
    mask = torch.zeros((pnum, 1), device="cuda")
    mask[mask_index] = 1
    mask = mask.bool()
    gaussians.prune_points(mask.squeeze())
    return gaussians


def test_ssim_loss(gaussians, scene=None, pipe=None, bg=None, dataset=None):
    view_stacks = scene.getTestCameras().copy()
    ssim_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            gt_image = gt_image.unsqueeze(0)
            image = image.unsqueeze(0)
            ssim_sum += ssim(image, gt_image)
    return ssim_sum / len(view_stacks)


def test_display_power(gaussians, scene=None, pipe=None, bg=None, dataset=None, scale=1):
    view_stacks = scene.getTestCameras().copy()
    display_power_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            display_power_sum += display_power_metric_boost(image, scale=scale)
    return display_power_sum / len(view_stacks)

def test_render_power(gaussians, scene=None, pipe=None, bg=None, dataset=None):
    view_stacks = scene.getTestCameras().copy()
    rendering_powers = []
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render_power(viewpoint_cam, gaussians, pipe, bg)
            result = hardware_power_modeling(render_pkg["point_statistics"], render_pkg["pixel_statistics"])
            rendering_powers.append(result["total_power"])  
    return sum(rendering_powers) / len(view_stacks)


def test_intesect_num(gaussians, scene=None, pipe=None, bg=None, dataset=None):
    view_stacks = scene.getTestCameras().copy()
    intersect_count_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            intersect_count_sum += torch.sum(render_pkg["intersect_count"])
    return intersect_count_sum / len(view_stacks)


def test_psnr_loss(gaussians, scene=None, pipe=None, bg=None, dataset=None):
    view_stacks = scene.getTestCameras().copy()
    psnr_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            gt_image = gt_image.unsqueeze(0)
            image = image.unsqueeze(0)
            # import ipdb; ipdb.set_trace()
            psnr_sum += psnr(image.contiguous(), gt_image.contiguous()).mean()

    return psnr_sum / len(view_stacks)


def test_hvs_loss(gaussians, scene=None, pipe=None, bg=None, dataset=None, pooling_size=1):
    view_stacks = scene.getTestCameras().copy()
    hvs_calc = HVSLoss(uniform_pooling_size=pooling_size)
    hvs_loss_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            # import ipdb; ipdb.set_trace()
            hvs_loss_sum += hvs_calc.calc_uniform_loss(image.unsqueeze(0), gt_image.unsqueeze(0), pooling_size=pooling_size)

    return hvs_loss_sum / len(view_stacks)

def controllable_prune(gaussians, scene, pipe, bg, dataset, args, prune_num, working_dir, hvs, pooling_size,
                       skip_test=False, skip_opacity_reset=False, no_save=False):
    """
    Prunes Gaussians based on opacity and metric testing, then adjusts scale weight based on the results.
    """
    # Perform initial opacity-based pruning
    # gaussians.opacity_prune(threshold=0.005)


    if skip_test:
        pass_test = True
    elif not hvs:
        # Perform metric testing
        tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
        tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
        
        # Check if the tested metrics pass the thresholds
        psnr_pass = tested_psnr >= args.target_psnr
        ssim_pass = tested_ssim >= args.target_ssim
        pass_test = psnr_pass & ssim_pass
    else:
        tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset, pooling_size=pooling_size)
        pass_test = tested_hvs <= args.target_hvs
    
    if pass_test:
        # Save the current best model
        if not no_save:
            torch.save((gaussians.capture(), -1), f"{working_dir}/current-best.pth")
        gaussians = metric_pruning(gaussians, prune_num=prune_num, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
        # Reset opacity maximum
        if not skip_opacity_reset:
            gaussians.reset_opacity_max(max=0.5)
        # Clear CUDA cache
        torch.cuda.empty_cache()

    quality = dict()
    if skip_test:
        pass
    elif not hvs:
        quality["ssim"] = tested_ssim
        quality["psnr"] = tested_psnr
    else:
        quality["hvs"] = tested_hvs

    return gaussians, pass_test, quality


def final_test(gaussians, scene, pipe, bg, dataset, args, opt, working_dir, hvs, pooling_size):
    """
    Prunes Gaussians based on opacity and metric testing, then adjusts scale weight based on the results.
    """
    if not hvs:
        # Perform metric testing
        tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
        tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset)
        
        # Check if the tested metrics pass the thresholds
        psnr_pass = tested_psnr >= args.target_psnr
        ssim_pass = tested_ssim >= args.target_ssim
        pass_test = psnr_pass & ssim_pass
    else:
        tested_hvs = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, dataset=dataset, pooling_size=pooling_size)
        pass_test = tested_hvs <= args.target_hvs
        
    restore_failed = False
    
    if pass_test:
        pass
    else:
        best_checkpoint = working_dir + "/current-best.pth"
        if not os.path.exists(best_checkpoint):
            restore_failed = True
        else:
            (model_params, _) = torch.load(best_checkpoint)
            gaussians.best_restore(model_params, opt)


    quality = dict()
    if not hvs:
        quality["ssim"] = tested_ssim
        quality["psnr"] = tested_psnr
    else:
        quality["hvs"] = tested_hvs

    return gaussians, pass_test, quality, restore_failed