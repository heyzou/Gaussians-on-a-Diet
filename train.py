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
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spherical_sample_path, generate_spiral_path, generate_spherify_path,  gaussian_poses, circular_poses
import json
import copy
from pathlib import Path
import json
import torchvision
import numpy as np
from os import makedirs
import os
import wandb
import time
import torchvision.transforms.functional as tf
from PIL import Image
import lpips
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui,render_imp,render_depth,render_pic
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.taming_utils import compute_gaussian_score, get_edges
import cv2
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
#except ImportError:
TENSORBOARD_FOUND = False
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,wandb=wandb):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_loader = scene.getTrainCameras()
    #viewpoint_stack = scene.getTrainCameras()
    viewpoint_num = len(scene.scene_info.train_cameras)
    viewpoint_stack = list(range(0,viewpoint_num))
    all_edges = []
    idx_video=0
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    grads_xyz = torch.zeros_like(gaussians._xyz, device="cuda")
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    out_pts_list1=[]
    gt_list1=[]
    counts_array = None
    densify_iter_num=0
    cam_pose_pkg=generate_ellipse_path(scene.getTrainCameras().dataset , n_frames=600)
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        #     viewpoint_indices = list(range(len(viewpoint_stack)))
        # rand_idx = randint(0, len(viewpoint_indices) - 1)
        # viewpoint_cam = viewpoint_stack.pop(rand_idx)
        # vind = viewpoint_indices.pop(rand_idx)
        if not viewpoint_stack:
            viewpoint_stack = list(range(0,len(scene.scene_info.train_cameras)))
        rend_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_idx = viewpoint_stack.pop(rend_idx)
        viewpoint_cam = train_loader.dataset[viewpoint_idx]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        #gaussians._scaling = gaussians._scaling * 0.001
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii, colors_precomp_copy,colors_pixel = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["colors_precomp_copy"], render_pkg["colors_pixel"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()
        grads_xyz = gaussians._xyz.grad
        grads_colors_pixel = torch.norm(colors_pixel.grad, dim=0)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # if iteration % 50 == 0:
            #     model_path=args.model_path
            #     render_path = os.path.join(model_path, 'circular')
            #     makedirs(render_path, exist_ok=True)
            #     n_frames = 600
            #     radius=0.5
            #     view = copy.deepcopy(scene.getTrainCameras().dataset[13])
            #     angle = 4 * np.pi * idx_video / n_frames
                
            #     cam = circular_poses(view, radius, angle)
            #     rendering = render(cam, gaussians, pipe, background)["render"]
            #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx_video) + ".png"))
            #     render_path = os.path.join(model_path, 'video_cycle')
            #     makedirs(render_path, exist_ok=True)
            #     pose = cam_pose_pkg[idx_video]
            #     view = scene.getTrainCameras().dataset[0]
            #     view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            #     view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            #     view.camera_center = view.world_view_transform.inverse()[3, :3]
            #     rendering = render(view, gaussians, pipe, background)["render"]
            #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx_video) + ".png"))
            #     idx_video += 1
            # if iteration % 200 == 0:
            #     render_pkg_temp = render(scene.getTestCameras().dataset[2], gaussians, pipe, bg)
            #     image_temp = render_pkg_temp["render"].detach().cpu()
            #     idx=int(iteration/200)
            #     filepath = os.path.join(args.model_path, "vedio")
            #     os.makedirs(filepath, exist_ok=True)
            #     filename = f"image_{idx:03d}.png"
            #     full_path = os.path.join(filepath, filename)
            #     torchvision.utils.save_image(image_temp, full_path)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp,wandb=wandb)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            #gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor,colors_precomp_copy, grads_xyz,visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold, radii, args.target_num,iteration, wandb=wandb)
                    #densify_iter_num += 1
                if iteration==5000 and args.our_door:
                    out_pts_list=[]
                    gt_list=[]
                    views=scene.getTrainCameras().dataset
                    for view in views:
                        gt = view.original_image[0:3, :, :]
                        render_depth_pkg = render_depth(view, gaussians, pipe, background)
                        out_pts = render_depth_pkg["out_pts"]
                        accum_alpha = render_depth_pkg["accum_alpha"]


                        prob=1-accum_alpha

                        prob = prob/prob.sum()
                        prob = prob.reshape(-1).cpu().numpy()


                        factor=1/(image.shape[1]*image.shape[2]*len(views)/(args.target_num* 0.8))


                        N_xyz=prob.shape[0]
                        num_sampled=int(N_xyz*factor)

                        indices = np.random.choice(N_xyz, size=num_sampled, 
                                                   p=prob,replace=False)
                        
                        out_pts = out_pts.permute(1,2,0).reshape(-1,3)
                        gt = gt.permute(1,2,0).reshape(-1,3)

                        out_pts_list.append(out_pts[indices])
                        gt_list.append(gt[indices])       

    

                    out_pts_merged=torch.cat(out_pts_list)
                    gt_merged=torch.cat(gt_list)

                    gaussians.reinitial_pts(out_pts_merged, gt_merged)
                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                    #viewpoint_stack = scene.getTrainCameras().copy()
                # # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                # #     gaussians.reset_opacity()
                

            # if iteration == opt.densify_until_iter:
            #     imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
            if iteration > opt.densify_from_iter  and iteration < 15000:#opt.densify_from_iter 
                non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99)
 
                render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background)
                accum_weights = render_pkg["accum_weights"]


                imp_score = torch.max(imp_score,accum_weights)
                if iteration % 50==0:
                    prune_mask = torch.zeros(gaussians._xyz.shape[0], device='cuda',dtype=torch.bool)
                    if gaussians._xyz.shape[0] > args.target_num:
                        imp_score = update_imp_score( scene, gaussians, pipe, background)
                        #imp_score = taming_imp_score(scene, gaussians, pipe, bg, opt)
                        k_value=gaussians._xyz.shape[0] - args.target_num
                        _,imp_indices = torch.topk(imp_score, k = k_value,dim=0, largest=False)
                        prune_mask[imp_indices]=True 
                         
                        #prune_mask = torch.logical_or(prune_mask,imp_score==0)
                    #else:
                        
                        #imp_score = update_imp_score( scene, gaussians, pipe, background)
                        #imp_score = taming_imp_score(scene, gaussians, pipe, bg, opt)
                        #k_value = int(0.001 * gaussians._xyz.shape[0])
                        #_,imp_indices = torch.topk(imp_score, k = k_value,dim=0, largest=False)
                        #prune_mask[imp_score==0]= True
                        
                        #big_points_vs = gaussians.max_radii2D > 200
                        #big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.3 * scene.cameras_extent
                        #prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                        #gaussians.training_setup(opt)
                    gaussians.prune_points(prune_mask)
                    #print("grads_xyz",grads_xyz)
                    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
                    grads_xyz = torch.zeros_like(gaussians._xyz, device="cuda")

                    torch.cuda.empty_cache()   
                if iteration > opt.densify_until_iter:
                    
                    render_depth_pkg = render_depth(viewpoint_cam, gaussians, pipe, background)
                    topk= int(gaussians._xyz.shape[0]/3000)
                    if topk!=0:
                        _, topk_indices = torch.topk(grads_colors_pixel.view(-1), topk)
                        
                        out_pts = render_depth_pkg["out_pts"].permute(1,2,0).reshape(-1,3)
                        gt_rgb = gt_image[0:3, :, :].permute(1,2,0).reshape(-1,3)
                        
                        out_pts_list1.append(out_pts[topk_indices])
                        gt_list1.append(gt_rgb[topk_indices])
                        if not viewpoint_stack:
                            out_pts_merged=torch.cat(out_pts_list1,dim=0)
                            gt_merged=torch.cat(gt_list1,dim=0)
                            out_pts_list1=[]
                            gt_list1=[]
                    if iteration % 500 == 0:
                        gradient_map_color= torch.norm(colors_pixel.grad, dim=0)
                        grads_np_color = gradient_map_color.detach().cpu().numpy()
                        # Normalize gradient map to [0, 255] and convert to color
                        grad_map_norm_color = cv2.normalize(grads_np_color, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        grad_map_norm_color[grad_map_norm_color < 50]=0
                        grad_color = cv2.applyColorMap(grad_map_norm_color, cv2.COLORMAP_JET)
                        image_temp = image.permute(1, 2, 0).detach().cpu().numpy()
                        grad_color_rgb = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)
                        image_temp = (image_temp * 255).astype(np.uint8)
                        alpha = 0.5  # Transparency factor for blending
                        grad_color_rgb[grad_map_norm_color == 0] = (0, 0, 0)
                        overlay = cv2.addWeighted(image_temp, 1, grad_color_rgb, 0.5, 0)
                        if wandb:
                            wandb.log({"render_color_grad": [wandb.Image(overlay)]}, step=iteration)
                        gaussians.immigrate_pts(out_pts_merged,gt_merged)
                        mask =torch.zeros(gaussians._xyz.shape[0], device='cuda',dtype=torch.bool)
                        mask[-out_pts_merged.shape[0]:]=True
                        render_pkc_pic=render_pic(viewpoint_cam, gaussians, pipe, background,mask=mask)
                        render_img=render_pkc_pic["render"]
                        if wandb:
                            wandb.log({"render_img": [wandb.Image(render_img.detach().cpu())]}, step=iteration)
                        
                        gaussians.immigrate_pts(out_pts_merged,gt_merged)
                        mask =torch.zeros(gaussians._xyz.shape[0], device='cuda',dtype=torch.bool)
                        mask[-out_pts_merged.shape[0]:]=True
                        render_pkc_pic=render_pic(viewpoint_cam, gaussians, pipe, background,mask=mask)
                        render_img=render_pkc_pic["render"]
                        if wandb:
                            wandb.log({"render_img": [wandb.Image(render_img.detach().cpu())]}, step=iteration)
                        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
            torch.cuda.cudart().cudaProfilerStop()
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
def update_imp_score( scene, gaussians, pipe, background):
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
    views = scene.getTrainCameras().dataset
    random_indices = torch.randperm(len(scene.getTrainCameras().dataset))[:100]
    #random_views = random.sample(views, 10) 
    for index in random_indices:
        # print(idx)
        view = scene.getTrainCameras().dataset[index]
        render_pkg = render_imp(view, gaussians, pipe, background)
        accum_weights = render_pkg["accum_weights"]
        area_max = render_pkg["area_max"]

        accum_area_max = accum_area_max+area_max


        imp_score = torch.max(imp_score,accum_weights)
        #imp_score = imp_score + accum_weights
    return imp_score
def taming_imp_score(scene, gaussians, pipe, bg, opt):
    random_indices = torch.randperm(len(scene.getTrainCameras().dataset))[:100]
    all_edges=[]
    camlist=[]
    edge_losses=[]
    for index in random_indices:
        view = scene.getTrainCameras().dataset[index]
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())
        camlist.append(view)
        edge_losses.append(edges_loss_norm)
    score_coefficients = {'view_importance': 50, 'edge_importance': 50, 'mse_importance': 50, 'grad_importance': 25, 'dist_importance': 50, 'opac_importance': 100, 'dept_importance': 5, 'loss_importance': 10, 'radii_importance': 10, 'scale_importance': 20, 'count_importance': 0.1, 'blend_importance': 50}
    gaussian_importance = compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg, score_coefficients, opt)
    return gaussian_importance
    
    
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, wandb=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, },step=iteration)
        wandb.log({f"total_points":scene.gaussians._xyz.shape[0],  },step=iteration)
        memory_allocated = torch.cuda.memory_allocated() / 10_24**2
        wandb.log({"memory_allocated":memory_allocated, },step=iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras().dataset}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras().dataset[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips_fn(image, gt_image))
                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()  

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_psnr":psnr_test, },step=iteration)
                    wandb.log({f"{config['name']}_ssims":ssims_test, },step=iteration)
                    wandb.log({f"{config['name']}_lpipss":lpipss_test, },step=iteration)
                    #wandb.log({"opacity_all": wandb.Histogram(scene.gaussians.get_opacity.cpu().numpy()),},step=iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views.dataset, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        args.resulotion = None
        render_pkg = render(view, gaussians, pipeline,background)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

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

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger
def render_sets(dataset : ModelParams,opt, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree=3)
        args.resolution = -1
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        print("picture resolution is:",scene.getTrainCameras().dataset[0].original_image.shape)
        #gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), },step=iteration)

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, },step=iteration)
    
    return visible_count

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
    #parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=range(0000,op.iterations+2_000,2_000))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[op.iterations])
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--target_num", type=int, default=800_000)
    parser.add_argument("--our_door", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-1]
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"3DGS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork",code_dir="."),
            config=vars(args)
        )
        wandb.run.log_code(".")
        #wandb.run.log_code("~/home/zhangym/Workspace/gaussian-splatting-wandb/gaussian_model.py")
        #wandb.run.log_code("~/home/zhangym/Workspace/gaussian-splatting-wandb/arguments/__init__.py")
    else:
        wandb = None
    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,wandb=wandb)
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)
    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args),op.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")

    # All done
    print("\nTraining complete.")
