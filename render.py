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
import cv2
import os
import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from bokeh_renderer.utils import get_video_rendering_path
from utils import graphics_utils
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np 
from scene.cameras import Camera
import time
import imageio
from utils.image_utils import psnr
from bokeh_renderer.utils import render_bokeh, render_bokeh_ex
from utils.graphics_utils import focal2fov, fov2focal
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_video=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    if save_video: video_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    image_array = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, rendered_depth, rendered_middepth, rendered_mask = render_pkg["render"], render_pkg["depth"], render_pkg["middepth"], render_pkg["mask"]
    
        rendered_depth1 = rendered_depth / rendered_mask
        rendered_depth1 = torch.nan_to_num(rendered_depth1, 0, 0)

        if save_video:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            rendering = rendering.permute(1,2,0).cpu().numpy()
            rendering = (rendering * 255).astype(np.uint8)
            image_array.append(rendering)
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendered_depth1/rendered_depth1.max(), os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
    if save_video:
        imageio.mimwrite(os.path.join(video_path, 'output.mp4'), np.stack(image_array),fps=30, quality=10)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, save_video : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        
        if save_video:
            pose_bounds = np.load(os.path.join(dataset.source_path, 'poses_bounds.npy'))
            depth_ranges = pose_bounds[:, -2:]
            near_far = np.array([depth_ranges[:, 0].min().item(), depth_ranges[:, 1].max().item()]).astype(np.float32)
            rendering_video_meta = render_video(scene.getTrainCameras(),near_far)
            render_set(dataset.model_path, "video", scene.loaded_iter, rendering_video_meta, gaussians, pipeline, background, save_video)
                    
def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def render_video(Cameras, near_far):
    # W = Cameras[0].image_width
    # H = Cameras[0].image_height
    FovX = Cameras[0].FoVx
    FovY = Cameras[0].FoVy
    tar_ixt = Cameras[0].intrinsic
    view = []
    for i in range(len(Cameras)):
        view.append(Cameras[i].extrinsic.unsqueeze(0))
    ref_poses = torch.cat(view).cpu().numpy()
    train_c2w_all = np.linalg.inv(ref_poses)
    poses_paths = get_video_rendering_path(ref_poses,'spiral', near_far=near_far, train_c2w_all=train_c2w_all, n_frames=180)
    rendering_video_meta = []
    id = 0
    for pose in poses_paths[0]:
        R = np.array(pose[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(pose[:3, 3], np.float32)
        extrinsic = np.zeros([4,4],dtype=np.float32)
        extrinsic[:3,:3] = R.T
        extrinsic[:3,3] = T
        extrinsic[3,3] = 1
        rendering_meta = Camera(colmap_id=2, R=R, T=T, 
                  FoVx=FovX, FoVy=FovY, 
                  image=Cameras[0].original_image, gt_alpha_mask=None,
                  gt_depth=None,
                  image_name=str(id).zfill(3), uid=id, data_device='cuda',
                  intrinsic=torch.tensor(tar_ixt,dtype=torch.float32).cuda(),
                  extrinsic=torch.tensor(pose,dtype=torch.float32).cuda())
        rendering_video_meta.append(rendering_meta)
    return rendering_video_meta
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_video", default=True, type=bool)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.save_video)