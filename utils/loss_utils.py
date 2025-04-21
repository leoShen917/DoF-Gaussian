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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_maskloss(network_output, gt, mask):
    return (torch.abs((network_output - gt))*mask).mean()

def l1_relative_loss(network_output, gt):
    return torch.abs((normalize(network_output) - gt)).mean()

def bokeh_weight(x):
    weight = torch.zeros_like(x)
    x = (x-x.min())/(x.max()-x.min())    
    weight = 1/(1+torch.exp(-15*(x-0.3)))
    return weight

def l1_loss_focus(network_output, gt, depth, disp_focus, flag):
    per_loss = torch.abs((network_output - gt))
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth), depth)
    disps = disp_map[0].detach()
    disps = torch.clamp(disps / torch.quantile(disps,0.98),min=1e-5,max=1.0)
    focus_weight = 1-bokeh_weight(torch.abs(disps-disp_focus.detach()))
    return (focus_weight * per_loss).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ncc(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    sigma1 = torch.sqrt(sigma1_sq + 1e-4)
    sigma2 = torch.sqrt(sigma2_sq + 1e-4)

    image1_norm = (img1 - mu1) / (sigma1 + 1e-8)
    image2_norm = (img2 - mu2) / (sigma2 + 1e-8)

    ncc = F.conv2d((image1_norm * image2_norm), window, padding=0, groups=channel)

    return torch.mean(ncc, dim=2)

def compute_depth_loss(depth_pred, depth_gt, mask_gt):
    loss_list = []
    for pred, gt, mask in zip(depth_pred, depth_gt, mask_gt):
        log_pred = torch.log(pred[mask])
        log_target = torch.log(gt[mask])
        alpha = (log_target - log_pred).sum()/mask.sum()
        log_diff = torch.abs((log_pred - log_target + alpha))
        d = 0.05*0.2*(log_diff.sum()/mask.sum())
        loss_list.append(d)

    return torch.stack(loss_list, 0).mean()

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

