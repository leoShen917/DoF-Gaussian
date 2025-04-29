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
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_focus
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import depth_double_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from bokeh_renderer.utils import render_bokeh
from bokeh_renderer import depth_priors
from utils.io_utils import *
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, gap = pipe.interval)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras)
    
    all_view = None
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # finetune the depth model/ can change model
    depth_priors.train(args)
    # load the depth priors
    image_list_train = load_img_list(args.source_path, args.llffhold)
    d_priors = load_depths(image_list_train,
                        os.path.join(dataset.model_path, 'depth_priors', 'results'), 
                        trainCameras[0].image_height, trainCameras[0].image_width)
    colmap_depths, colmap_masks = load_colmap(image_list_train, args.source_path, trainCameras[0].image_height, trainCameras[0].image_width)
    d_priors = align_scales(d_priors, colmap_depths, colmap_masks)
    d_priors = torch.tensor(d_priors).to('cuda')
    os.makedirs(os.path.join(dataset.model_path,"error"),exist_ok=True)
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not all_view:
            viewpoint_stack = scene.getTrainCameras().copy()
            all_view = [i for i in range(0, len(viewpoint_stack))]
        random_view = all_view.pop(randint(0, len(all_view)-1))
        viewpoint_cam = viewpoint_stack[random_view]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                    render_pkg["render"], 
                                                                    render_pkg["viewspace_points"], 
                                                                    render_pkg["visibility_filter"], 
                                                                    render_pkg["radii"])
        
        rendered_mask: torch.Tensor = render_pkg["mask"]
        rendered_depth: torch.Tensor = render_pkg["depth"]
        rendered_middepth: torch.Tensor = render_pkg["middepth"]
        rendered_normal: torch.Tensor = render_pkg["normal"]
        depth_distortion: torch.Tensor = render_pkg["depth_distortion"]

        gt_image = viewpoint_cam.original_image

        edge = viewpoint_cam.edge
        rendered_depth = rendered_depth / rendered_mask
        rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)
        if iteration >= opt.regularization_from_iter:
            # depth distortion loss
            lambda_distortion = opt.lambda_distortion
            depth_distortion = torch.where(rendered_mask>0,depth_distortion/(rendered_mask * rendered_mask).detach(),0)
            distortion_map = depth_distortion[0] * edge
            distortion_loss = distortion_map.mean()
            # normal consistency loss
            
            depth_middepth_normal, _ = depth_double_to_normal(viewpoint_cam, rendered_depth, rendered_middepth)
            depth_ratio = 0.6
            rendered_normal = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
            rendered_normal = rendered_normal.permute(1,2,0)
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=-1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            lambda_depth_normal = opt.lambda_depth_normal
        else:
            lambda_distortion = 0
            lambda_depth_normal = 0
            distortion_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            
        if iteration > opt.scatter_iter:#:
            if iteration % 1000 == 0:
                print("gaussian numbers",gaussians._xyz.shape[0])
                torchvision.utils.save_image(rendered_image,args.model_path+"/error/rgb_clear_%d.png" %iteration)  
            K_bokeh = gaussians.get_K_bokeh
            disp_focus = gaussians.get_disp_focus
            rendered_image = render_bokeh(rendered_image,
                                rendered_depth,
                                K_bokeh=10 * K_bokeh[random_view],
                                gamma=2,
                                disp_focus=disp_focus[random_view],
                                defocus_scale=1, iteration=iteration) 
        # loss
        lambda_depth = opt.lambda_depth * (1-iteration/opt.iterations*0.9)
        if iteration < 10000: 
            Ll1_render = l1_loss(rendered_image, gt_image)
            depth_loss =  l1_loss(d_priors[random_view:(random_view+1)],rendered_depth)
        else: 
            Ll1_render =  l1_loss_focus(rendered_image, gt_image, rendered_depth, disp_focus[random_view],True)
            depth_loss =  l1_loss_focus(d_priors[random_view:(random_view+1)], rendered_depth, rendered_depth, disp_focus[random_view],False)
        if iteration % 1000 == 0 and iteration>900:
            torchvision.utils.save_image(torch.mean(torch.abs(rendered_image-gt_image),dim=0),args.model_path+"/error/%d.png" %iteration)  
            torchvision.utils.save_image(rendered_image,args.model_path+"/error/rgb_%d.png" %iteration)  
            torchvision.utils.save_image((rendered_depth/rendered_depth.max()).unsqueeze(0),args.model_path+"/error/depth_%d.png" %iteration)  

        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))
        loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion + lambda_depth * depth_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0.4 * distortion_loss.item() + 0.6 * ema_depth_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_dep": f"{ema_depth_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            psnr = training_report(tb_writer, iteration, Ll1_render, loss, distortion_loss, depth_normal_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),gaussians.get_K_bokeh,gaussians.get_disp_focus)
                
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print(gaussians.get_K_bokeh,gaussians.get_disp_focus)
                torch.save(gaussians.get_K_bokeh, os.path.join(scene.model_path,"K_list.pt"))
                torch.save(gaussians.get_disp_focus, os.path.join(scene.model_path,"focus_list.pt"))
            # Densification
            if iteration<opt.densify_until_iter and iteration>opt.opacity_reset_interval:
            #     # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, iteration)
                    gaussians.prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, iteration)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
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

def training_report(tb_writer, iteration, Ll1, loss, depth_loss, normal_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, K_bokeh, disp_focus):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for i in range(len(K_bokeh)):
            tb_writer.add_scalar('K_bokeh/K_%d' %i, K_bokeh[i], iteration)
            tb_writer.add_scalar('disp_focus/focus_%d' %i, disp_focus[i], iteration)
    # Report test and samples of training set
    if iteration % 1000 == 0:
    # if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        psnr_ = 0
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                l1_depth = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    depth = render_result["depth"]
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # l1_depth /= len(config['cameras'])  
                l1_depth = 0        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} depth {}".format(iteration, config['name'], l1_test, psnr_test, l1_depth))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), file=file_object)
                        psnr_ = psnr_test
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_depth', l1_depth, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        return psnr_
    else:
        return 0
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=lp.extract(args), 
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
