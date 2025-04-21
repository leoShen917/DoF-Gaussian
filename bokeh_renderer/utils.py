import torch
import torch.nn as nn
from bokeh_renderer.scatter import ModuleRenderScatter
from bokeh_renderer.scatter_ex import ModuleRenderScatterEX
from utils.loss_utils import bokeh_weight
import numpy as np
from scipy.spatial.transform import Rotation
import torchvision
def render_bokeh(rgbs, 
                 rendered_depth, 
                 K_bokeh=20, 
                 gamma=4, 
                 disp_focus=90/255, 
                 defocus_scale=1,
                 iteration=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classical_renderer = ModuleRenderScatter().to(device)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(rendered_depth), rendered_depth)
    disps = disp_map[0].detach()
    # disps = torch.clamp(disps / torch.quantile(disps,0.99),min=1e-5,max=1.0)
    disps = torch.clamp((disps - disps.min()) / (torch.quantile(disps,0.95)- disps.min()),min=1e-5,max=1.0)
    signed_disp = disps - disp_focus
    defocus = (K_bokeh) * signed_disp / defocus_scale
    # if iteration > 10000:
    #     weight = bokeh_weight(torch.abs(signed_disp)).detach()
    #     defocus = defocus * weight
    stable = torch.ones_like(rgbs)*1e-5
    rgbs = torch.where(rgbs!=0, rgbs, stable)
    defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous().to(device)
    rgbs = rgbs.unsqueeze(0).contiguous()

    bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    bokeh_classical = bokeh_classical[0]
    return bokeh_classical

def render_bokeh_ex(rgbs, 
                 rendered_depth, 
                 K_bokeh=20, 
                 gamma=4, 
                 disp_focus=90/255, 
                 defocus_scale=1,
                 iteration=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classical_renderer = ModuleRenderScatterEX().to(device)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(rendered_depth), rendered_depth)
    disps = disp_map[0].detach()
    # disps = torch.clamp(disps / torch.quantile(disps,0.98),min=1e-5,max=1.0)
    disps = torch.clamp((disps - disps.min()) / (torch.quantile(disps,0.98)- disps.min()),min=1e-5,max=1.0)
    signed_disp = disps - disp_focus
    defocus = (K_bokeh) * signed_disp / defocus_scale
    if iteration > 10000:
        weight = bokeh_weight(torch.abs(signed_disp)).detach()
        defocus = defocus * weight
    stable = torch.ones_like(rgbs)*1e-5
    rgbs = torch.where(rgbs!=0, rgbs, stable)
    defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous()
    rgbs = rgbs.unsqueeze(0).contiguous()

    bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    bokeh_classical = bokeh_classical[0]
    return bokeh_classical

def get_interpolate_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = Rotation.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0]) > 180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N+1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral_render_path(c2ws_all, near_far, rads_scale=0.5, N_views=120):
    # center pose
    c2w = poses_avg(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    # print(near_far)
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    # print(focal)
    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:, :3, 3] - c2w[:3, 3][None]
    rads = np.percentile(np.abs(tt), 70, 0)*rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)

def get_video_rendering_path(ref_poses, mode, near_far, train_c2w_all, n_frames=60, rads_scale=0.3):
    # loop over batch
    poses_paths = []
    # ref_poses = ref_poses[4:5]
    ref_poses = ref_poses[None]
    # ref_poses = np.concatenate([ref_poses[:,0:1],ref_poses[:,18:19]],axis=1)
    for batch_idx, cur_src_poses in enumerate(ref_poses):
        if mode == 'interpolate':
            # convert to c2ws
            pose_square = torch.eye(4).unsqueeze(0).repeat(cur_src_poses.shape[0], 1, 1)
            cur_src_poses = torch.from_numpy(cur_src_poses)
            pose_square[:, :3, :] = cur_src_poses[:,:3]
            cur_c2ws = pose_square.double().inverse()[:, :3, :].to(torch.float32).cpu().detach().numpy()
            cur_path = get_interpolate_render_path(cur_c2ws, n_frames)
        elif mode == 'spiral':
            cur_c2ws_all = train_c2w_all
            cur_near_far = near_far.tolist()
            cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
        else:
            raise Exception(f'Unknown video rendering path mode {mode}')

        # convert back to extrinsics tensor
        cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32)
        poses_paths.append(cur_w2cs)

    poses_paths = torch.stack(poses_paths, dim=0)
    return poses_paths