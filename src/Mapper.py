import os
import shutil
import traceback
import subprocess
import time
import cv2
import numpy as np
import open3d as o3d
import torch
import math

from ast import literal_eval
from colorama import Fore, Style
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.common import (get_camera_from_tensor, get_samples, get_samples_with_pixel_grad,
                        get_tensor_from_camera, random_select, setup_seed,
                        as_intrinsics_matrix, get_npc_input_pcl,
                        preprocess_point_cloud, execute_global_registration, refine_registration,
                        compute_rel_trans, compute_cos_rel_rot)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.utils.Logger import Logger

from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim
from typing import Literal, Optional

import wandb


class Mapper(object):
    """
    Mapper thread.

    """

    def __init__(self, cfg, args, slam):

        self.cfg = cfg
        self.args = args
        self.idx = slam.idx
        self.output = slam.output
        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.renderer = slam.renderer_map
        self.renderer.sigmoid_coefficient = cfg['rendering']['sigmoid_coef_mapper']
        self.npc = slam.npc
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list
        self.exposure_feat_shared = slam.exposure_feat
        self.exposure_feat = self.exposure_feat_shared[0].clone(
        ).requires_grad_()

        self.camera_before_correction = slam.camera_before_correction
        self.use_cbc = slam.use_cbc

        self.wandb = cfg['wandb']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.project_name = cfg['project_name']
        self.use_view_direction = cfg['use_view_direction']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.dynamic_r_add, self.dynamic_r_query = None, None
        self.encode_exposure = cfg['model']['encode_exposure']

        self.radius_add_max = cfg['pointcloud']['radius_add_max']
        self.radius_add_min = cfg['pointcloud']['radius_add_min']
        self.radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = cfg['pointcloud']['color_grad_threshold']
        self.eval_img = cfg['rendering']['eval_img']

        self.device = cfg['mapping']['device']
        self.fix_geo_decoder = cfg['mapping']['fix_geo_decoder']
        self.fix_color_decoder = cfg['mapping']['fix_color_decoder']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = False
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.pixels_adding = cfg['mapping']['pixels_adding']
        self.pixels_based_on_color_grad = cfg['mapping']['pixels_based_on_color_grad']
        self.num_joint_iters = cfg['mapping']['iters']
        self.geo_iter_first = cfg['mapping']['geo_iter_first']
        self.iters_first = cfg['mapping']['iters_first']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.keyframe_global_every = cfg['mapping']['keyframe_global_every']
        self.geo_iter_ratio = cfg['mapping']['geo_iter_ratio']
        self.vis_inside = cfg['mapping']['vis_inside']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        self.frustum_edge = cfg['mapping']['frustum_edge']
        self.filter_before_add_points = cfg['mapping']['filter_before_add_points']
        self.save_ckpts = cfg['mapping']['save_ckpts']
        self.crop_edge = 0 if cfg['cam']['crop_edge'] is None else cfg['cam']['crop_edge']
        self.save_rendered_image = cfg['mapping']['save_rendered_image']
        self.min_iter_ratio = cfg['mapping']['min_iter_ratio']
        self.fixed_segment_size = cfg["mapping"]["fixed_segment_size"]
        self.segment_strategy = cfg["mapping"]["segment_strategy"]
        self.segment_rel_trans = cfg["mapping"]["segment_rel_trans"]
        self.segment_rot_cos = cfg["mapping"]["segment_rot_cos"]

        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        self.keyframe_dict = []
        self.keyframe_list = []
        self.keyframe_list_global = []
        self.keyframe_dict_global = []
        self.frame_reader = get_dataset(
            cfg, args, device=self.device)
        self.n_img = len(self.frame_reader)
        self.logger = Logger(cfg, args, self)
        self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                     verbose=self.verbose, device=self.device, wandb=self.wandb,
                                     vis_inside=self.vis_inside, total_iters=self.num_joint_iters,
                                     img_dir=os.path.join(self.output, 'rendered_image'))
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.npc_geo_feats = None
        self.npc_col_feats = None
        self.end = False

    def set_pipe(self, pipe):
        self.pipe = pipe

    def filter_point_before_add(self, rays_o, rays_d, gt_depth, prev_c2w):
        with torch.no_grad():
            points = rays_o[..., None, :] + \
                rays_d[..., None, :] * gt_depth[..., None, None]
            points = points.reshape(-1, 3).cpu().numpy()
            H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy

            if torch.is_tensor(prev_c2w):
                prev_c2w = prev_c2w.cpu().numpy()
            w2c = np.linalg.inv(prev_c2w)
            ones = np.ones_like(points[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [points, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)

            edge = 0
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
        return torch.from_numpy(~mask).to(self.device).reshape(-1)

    def get_mask_from_c2w(self, c2w, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.
        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame. for each (x,y)<->(width,height)

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        points = np.array(self.npc.get_cloud_pos()).reshape(-1, 3)

        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        # make the axis consistent and let the frustum feature selection get the correct features.
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = self.frustum_edge  # crop here on width and height
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        points = points[mask]

        return np.where(mask)[0].tolist()

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=8, pixels=200):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels,
            H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device, depth_filter=True)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            #cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    # def check_new_fragment(self, method: Literal["fixed", "rot_trans"], idx:Optional[torch.Tensor]=None, cur_c2w:Optional[torch.Tensor]=None) -> bool:
    #     if method == "fixed":
    #         new = idx.item() % self.fixed_segment_size
    #         index_segment = idx.item() // self.fixed_segment_size

    #         #move old segments using the error between the predicted keyframe and the gt keyframe
    #         fragments = self.npc.get_fragments()

    #         if fragments is None:
    #             return new == 0
    #         elif new == 0:
    #             print("updating old geometry")
    #             with torch.no_grad():
    #                 last_segment = list(fragments.keys())[-1]
    #                 print(last_segment)
    #                 fragment = fragments[last_segment]
    #                 pred_c2w = fragment["keyframe"].cuda()
    #                 gt_c2w = fragment["gt_camera"].cuda()

    #                 delta = gt_c2w@pred_c2w.inverse()

    #                 points = fragment["npc"]
    #                 ones = torch.ones(len(points), 1).cuda()
    #                 homo_points = torch.cat([torch.tensor(points).cuda(), ones], dim=1).cuda().float()
    #                 homo_points = homo_points@torch.t(delta)
    #                 pts = homo_points[:, :3]
    #                 z = homo_points[:, -1:]
    #                 pts = pts / z
    #                 self.npc.set_fragment_npc(last_segment, pts.tolist())

    #                 cameras = self.estimate_c2w_list[self.fixed_segment_size*(index_segment-1):self.fixed_segment_size*(index_segment)]

    #                 self.camera_before_correction[0] = cameras[-1]
    #                 self.use_cbc[0] = True

    #                 # print(cameras, cameras.shape)
    #                 # print(self.camera_before_correction)

    #                 cameras_tensor = cameras.cuda()

    #                 new_cameras_tensor = delta@cameras_tensor

    #                 new_cameras_tensor = new_cameras_tensor.cpu()

    #                 # print(new_cameras_tensor[0], cameras[0])

    #                 self.estimate_c2w_list[self.fixed_segment_size*(index_segment-1):self.fixed_segment_size*index_segment] = new_cameras_tensor

    #         return new == 0
    #     elif method == "rot_trans":
    #         return self.npc.check_rot_trans(cur_c2w)
    #     else:
    #         raise NotImplementedError

    def check_new_fragment(self, method: Literal["fixed", "rot_trans"], idx: Optional[torch.Tensor] = None, cur_c2w: Optional[torch.Tensor] = None) -> bool:
        if method == "fixed":
            new = idx.item() % self.fixed_segment_size
            return new == 0
        elif method == "rot_trans":
            return self.npc.check_rot_trans(cur_c2w)
        else:
            raise NotImplementedError

    def optimize_map(self, num_joint_iters, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w,
                     keyframe_dict, keyframe_list, cur_c2w, color_refine=False, new_fragment=None):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 
            prev_c2w (tensor): est_c2w of last mapping frame.

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        npc = self.npc
        cfg = self.cfg
        device = self.device
        init = True if idx == 0 else False
        bottom = torch.tensor([0, 0, 0, 1.0], device=self.device).reshape(1, 4)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                print("global")
                num = self.mapping_window_size-2
                optimize_frame = list(
                    range(max(0, len(keyframe_dict)-1-num), len(keyframe_dict)-1))
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)
            elif self.keyframe_selection_method == "segments":
                print("segments")
                keyframe_dict, keyframe_list = self.npc.get_segments_keyframe_dict()
                for keyframe in keyframe_dict.values():
                    if self.use_dynamic_radius:
                        keyframe["dynamic_r_query"] = torch.load(f'{self.output}/dynamic_r_frame/r_query_{keyframe["frame"]:05d}.pt', map_location=self.device)

                    if self.encode_exposure:
                        keyframe["exposure_feat"] = self.exposure_feat_all[keyframe["frame"] // self.cfg["mapping"]["every_frame"]]
                optimize_frame = list(range(0, len(keyframe_dict)))

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0 and self.keyframe_selection_method != "segments":
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        if self.save_selected_keyframes_info and self.keyframe_selection_method != "segments":
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels // 10 if self.keyframe_selection_method == "segments" else self.mapping_pixels//len(
            optimize_frame)

        decoders_para_list = []
        color_pcl_para = []
        geo_pcl_para = []
        gt_depth_np = cur_gt_depth.cpu().numpy()
        gt_depth = cur_gt_depth.to(device)
        gt_color = cur_gt_color.to(device)

        if idx == 0:
            add_pts_num = torch.clamp(self.pixels_adding * ((gt_depth.median()/2.5)**2),
                                      min=self.pixels_adding, max=self.pixels_adding*3).int().item()
        else:
            add_pts_num = self.pixels_adding
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
            0, H, 0, W, add_pts_num,
            H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True)

        if not color_refine:
            frame_pts_add = 0
            if self.filter_before_add_points:
                if idx != 0:
                    # make sure add enough points to the non-overlapping area
                    mask_add = self.filter_point_before_add(
                        batch_rays_o, batch_rays_d, batch_gt_depth, self.prev_c2w)
                    _ = self.npc.add_neural_points(batch_rays_o[mask_add], batch_rays_d[mask_add],
                                                   batch_gt_depth[mask_add], batch_gt_color[mask_add],
                                                   dynamic_radius=self.dynamic_r_add[j, i][mask_add] if self.use_dynamic_radius else None, idx=idx, gt_color=gt_color, gt_depth=gt_depth, cur_c2w=cur_c2w, gt_camera=gt_cur_c2w)
                    print(f'{_} locations to add points in non-overlapping area.')
                    frame_pts_add += _

                    # try add points to overlapped area too
                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                        0, H, 0, W, int(1000),
                        H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True)
                    mask_add = self.filter_point_before_add(
                        batch_rays_o, batch_rays_d, batch_gt_depth, self.prev_c2w)
                    _ = self.npc.add_neural_points(batch_rays_o[~mask_add], batch_rays_d[~mask_add],
                                                   batch_gt_depth[~mask_add], batch_gt_color[~mask_add],
                                                   dynamic_radius=self.dynamic_r_add[j, i][~mask_add] if self.use_dynamic_radius else None, idx=idx, gt_color=gt_color, gt_depth=gt_depth, cur_c2w=cur_c2w, gt_camera=gt_cur_c2w)
                    print(f'{_} locations to add points in overlapping area.')
                    frame_pts_add += _
                else:
                    _ = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                                   dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None, idx=idx, gt_color=gt_color, gt_depth=gt_depth, cur_c2w=cur_c2w, gt_camera=gt_cur_c2w)
                    print(f'{_} locations to add points.')
                    frame_pts_add += _
            else:
                _ = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                               dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None, idx=idx, gt_color=gt_color, gt_depth=gt_depth, cur_c2w=cur_c2w, gt_camera=gt_cur_c2w)
                print(f'{_} locations to add points.')
                frame_pts_add += _

            if self.pixels_based_on_color_grad > 0:

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples_with_pixel_grad(
                    0, H, 0, W, self.pixels_based_on_color_grad,
                    H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device,
                    depth_filter=True, return_index=True)
                _ = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                               is_pts_grad=True, dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None, idx=idx, gt_color=gt_color, gt_depth=gt_depth, cur_c2w=cur_c2w, gt_camera=gt_cur_c2w)
                print(f'{_} locations to add points based on pixel gradients.')
                frame_pts_add += _

        # apply transformation
        transformed_camera = self.npc.apply_transformation()
        if transformed_camera is not None:
            cur_c2w = transformed_camera.to(cur_c2w.device)

        # clone all point feature from shared npc, (N_points, c_dim)
        npc_geo_feats = self.npc.get_geo_feats(self.end)
        npc_col_feats = self.npc.get_col_feats(self.end)
        self.cloud_pos_tensor = torch.tensor(
            self.npc.get_cloud_pos(self.end), device=self.device)
        if self.encode_exposure:
            self.exposure_feat = self.exposure_feat_shared[0].clone(
            ).requires_grad_()

        if self.frustum_feature_selection:  # required if not color_refine
            masked_c_grad = {}
            mask_c2w = cur_c2w
            indices = self.get_mask_from_c2w(mask_c2w, gt_depth_np)
            geo_pcl_grad = npc_geo_feats[indices].clone(
            ).detach().requires_grad_(True)
            color_pcl_grad = npc_col_feats[indices].clone(
            ).detach().requires_grad_(True)

            geo_pcl_para = [geo_pcl_grad]
            color_pcl_para = [color_pcl_grad]

            masked_c_grad['geo_pcl_grad'] = geo_pcl_grad
            masked_c_grad['color_pcl_grad'] = color_pcl_grad
            masked_c_grad['indices'] = indices
        else:
            masked_c_grad = {}
            geo_pcl_grad = npc_geo_feats.clone().detach().requires_grad_(True)
            color_pcl_grad = npc_col_feats.clone().detach().requires_grad_(True)

            geo_pcl_para = [geo_pcl_grad]
            color_pcl_para = [color_pcl_grad]

            masked_c_grad['geo_pcl_grad'] = geo_pcl_grad
            masked_c_grad['color_pcl_grad'] = color_pcl_grad

        if not self.fix_geo_decoder:
            decoders_para_list += list(
                self.decoders.geo_decoder.parameters())
        if not self.fix_color_decoder:
            decoders_para_list += list(
                self.decoders.color_decoder.parameters())

        if self.fix_color_decoder:
            decoders_para_list += list(
                self.decoders.color_decoder.embedder.parameters())
            decoders_para_list += list(
                self.decoders.color_decoder.embedder_rel_pos.parameters())

        if self.fix_geo_decoder:
            decoders_para_list += list(
                self.decoders.geo_decoder.embedder.parameters())
            decoders_para_list += list(
                self.decoders.geo_decoder.embedder_rel_pos.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        optim_para_list = [{'params': decoders_para_list, 'lr': 0},
                           {'params': geo_pcl_para, 'lr': 0},
                           {'params': color_pcl_para, 'lr': 0}]
        if self.BA:
            optim_para_list.append({'params': camera_tensor_list, 'lr': 0})
        if self.encode_exposure:
            optim_para_list.append(
                {'params': self.exposure_feat, 'lr': 0.001})
        optimizer = torch.optim.Adam(optim_para_list)

        if idx > 0 and not color_refine:
            num_joint_iters = np.clip(int(num_joint_iters*frame_pts_add/300), int(
                self.min_iter_ratio*num_joint_iters), 2*num_joint_iters)

        for joint_iter in range(num_joint_iters):
            tic = time.perf_counter()
            if self.frustum_feature_selection:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                indices = masked_c_grad['indices']
                npc_geo_feats[indices] = geo_feats
                npc_col_feats[indices] = col_feats
            else:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                npc_geo_feats = geo_feats  # all feats
                npc_col_feats = col_feats

            if joint_iter <= (self.geo_iter_first if init else int(num_joint_iters*self.geo_iter_ratio)):
                self.stage = 'geometry'
            else:
                self.stage = 'color'
            cur_stage = 'init' if init else 'stage'
            optimizer.param_groups[0]['lr'] = cfg['mapping'][cur_stage][self.stage]['decoders_lr']
            optimizer.param_groups[1]['lr'] = cfg['mapping'][cur_stage][self.stage]['geometry_lr']
            # if idx == self.n_img-1 and self.color_refine:
            #     optimizer.param_groups[0]['lr'] = cfg['mapping'][cur_stage]['color']['decoders_lr']
            #     optimizer.param_groups[1]['lr'] = 0.0
            #     optimizer.param_groups[2]['lr'] = cfg['mapping'][cur_stage]['color']['color_lr']/10.0
            # else:
            optimizer.param_groups[2]['lr'] = cfg['mapping'][cur_stage][self.stage]['color_lr']

            if self.BA:
                # when to conduct BA
                if joint_iter >= num_joint_iters*(self.geo_iter_ratio+0.2) and (joint_iter <= num_joint_iters*(self.geo_iter_ratio+0.3)):
                    optimizer.param_groups[3]['lr'] = self.BA_cam_lr
                else:
                    optimizer.param_groups[3]['lr'] = 0.0

            if self.vis_inside:
                self.visualizer.vis(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.npc, self.decoders,
                                    npc_geo_feats, npc_col_feats, freq_override=False,
                                    dynamic_r_query=self.dynamic_r_query, cloud_pos=self.cloud_pos_tensor,
                                    exposure_feat=self.exposure_feat)

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []
            batch_r_query_list = []
            exposure_feat_list = []
            indices_tensor = []

            camera_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']

                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = cur_c2w

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                    0, H, 0, W, pixs_per_image,
                    H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())
                if self.use_dynamic_radius:
                    if frame == -1:
                        batch_r_query_list.append(self.dynamic_r_query[j, i])
                    else:
                        batch_r_query_list.append(
                            keyframe_dict[frame]['dynamic_r_query'][j, i])

                if self.encode_exposure:  # needs to render frame by frame
                    exposure_feat_list.append(
                        self.exposure_feat if frame == -1 else keyframe_dict[frame]['exposure_feat'].to(device))
                    # log frame idx of pixels
                    frame_indices = torch.full(
                        (i.shape[0],), frame, dtype=torch.long, device=self.device)
                    indices_tensor.append(frame_indices)

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)
            r_query_list = torch.cat(
                batch_r_query_list) if self.use_dynamic_radius else None

            with torch.no_grad():
                inside_mask = batch_gt_depth <= torch.minimum(
                    10*batch_gt_depth.median(), 1.2*torch.max(batch_gt_depth))

            batch_rays_d, batch_rays_o = batch_rays_d[inside_mask], batch_rays_o[inside_mask]
            batch_gt_depth, batch_gt_color = batch_gt_depth[inside_mask], batch_gt_color[inside_mask]
            if self.use_dynamic_radius:
                r_query_list = r_query_list[inside_mask]
            ret = self.renderer.render_batch_ray(npc, self.decoders, batch_rays_d, batch_rays_o, device, self.stage,
                                                 gt_depth=batch_gt_depth, npc_geo_feats=npc_geo_feats,
                                                 npc_col_feats=npc_col_feats,
                                                 is_tracker=True if self.BA else False,
                                                 cloud_pos=self.cloud_pos_tensor,
                                                 dynamic_r_query=r_query_list,
                                                 exposure_feat=None)
            depth, uncertainty, color, valid_ray_mask = ret

            depth_mask = (batch_gt_depth > 0) & valid_ray_mask
            depth_mask = depth_mask & (~torch.isnan(depth))
            geo_loss = torch.abs(
                batch_gt_depth[depth_mask]-depth[depth_mask]).sum()
            loss = geo_loss.clone()
            if self.stage == 'color':
                if self.encode_exposure:
                    indices_tensor = torch.cat(indices_tensor, dim=0)[
                        inside_mask]
                    start_end = []
                    for i in torch.unique_consecutive(indices_tensor, return_counts=False):
                        match_indices = torch.where(indices_tensor == i)[0]
                        start_idx = match_indices[0]
                        end_idx = match_indices[-1] + 1
                        start_end.append((start_idx.item(), end_idx.item()))
                    for i, exposure_feat in enumerate(exposure_feat_list):
                        start, end = start_end[i]
                        affine_tensor = self.decoders.color_decoder.mlp_exposure(
                            exposure_feat)
                        rot, trans = affine_tensor[:9].reshape(
                            3, 3), affine_tensor[-3:]
                        color_slice = color[start:end].clone()
                        color_slice = torch.matmul(color_slice, rot) + trans
                        color[start:end] = color_slice
                    color = torch.sigmoid(color)
                color_loss = torch.abs(
                    batch_gt_color[depth_mask] - color[depth_mask]).sum()

                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()

            # put selected and updated params back to npc
            if self.frustum_feature_selection:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                indices = masked_c_grad['indices']
                npc_geo_feats, npc_col_feats = npc_geo_feats.detach(), npc_col_feats.detach()
                npc_geo_feats[indices], npc_col_feats[indices] = geo_feats.clone(
                ).detach(), col_feats.clone().detach()
            else:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                npc_geo_feats, npc_col_feats = geo_feats.detach(), col_feats.detach()

            toc = time.perf_counter()
            if not self.wandb:
                if joint_iter % 100 == 0:
                    if self.stage == 'geometry':
                        print('iter: ', joint_iter, ', time',
                              f'{toc - tic:0.6f}', ', geo_loss: ', f'{geo_loss.item():0.6f}')
                    else:
                        print('iter: ', joint_iter, ', time', f'{toc - tic:0.6f}',
                              ', geo_loss: ', f'{geo_loss.item():0.6f}', ', color_loss: ', f'{color_loss.item():0.6f}')

            if joint_iter == num_joint_iters-1:
                print('idx: ', idx.item(), ', time', f'{toc - tic:0.6f}', ', geo_loss_pixel: ', f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}',
                      ', color_loss_pixel: ', f'{(color_loss.item()/depth_mask.sum().item()):0.4f}')
                if self.wandb:
                    if not self.gt_camera:
                        wandb.log({'idx_map': int(idx.item()), 'time': float(f'{toc - tic:0.6f}'),
                                   'geo_loss_pixel': float(f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'color_loss_pixel': float(f'{(color_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'pts_total': self.npc.index_ntotal()})
                    else:
                        wandb.log({'idx': int(idx.item()), 'time': float(f'{toc - tic:0.6f}'),
                                   'geo_loss_pixel': float(f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'color_loss_pixel': float(f'{(color_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'pts_total': self.npc.index_ntotal()})

                    wandb.log({'idx_map': int(idx.item()),
                               'num_joint_iters': num_joint_iters})

        if (not self.vis_inside) or idx == 0:
            self.visualizer.vis(idx, self.num_joint_iters-1, cur_gt_depth, cur_gt_color, cur_c2w, self.npc, self.decoders,
                                npc_geo_feats, npc_col_feats, freq_override=True if idx == 0 else False,
                                dynamic_r_query=self.dynamic_r_query,
                                cloud_pos=self.cloud_pos_tensor, exposure_feat=self.exposure_feat,
                                cur_total_iters=num_joint_iters, save_rendered_image=True if self.save_rendered_image else False)

        if self.frustum_feature_selection:
            self.npc.update_geo_feats(geo_feats, indices=indices, end=self.end)
            self.npc.update_col_feats(col_feats, indices=indices, end=self.end)
        else:
            self.npc.update_geo_feats(npc_geo_feats, end=self.end)
            self.npc.update_col_feats(npc_col_feats, end=self.end)
        self.npc_geo_feats = npc_geo_feats
        self.npc_col_feats = npc_col_feats
        print('Mapper has updated point features.')

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.encode_exposure:
            self.exposure_feat_shared[0] = self.exposure_feat.clone().detach()
            self.exposure_feat_all.append(self.exposure_feat.detach().cpu())
            torch.save(self.decoders.color_decoder.state_dict(),
                       f'{self.output}/ckpts/color_decoder/{idx:05}.pt')

        if self.BA:
            return cur_c2w
        else:
            return None

    def run(self, time_string):
        cfg = self.cfg
        setup_seed(cfg["setup_seed"])
        scene_name = cfg['data']['input_folder'].split('/')[-1]

        if self.use_dynamic_radius:
            os.makedirs(f'{self.output}/dynamic_r_frame', exist_ok=True)
        if self.encode_exposure:
            os.makedirs(f"{self.output}/ckpts/color_decoder", exist_ok=True)
        if self.wandb:
            wandb.init(config=cfg, project=self.project_name, group=f'slam_{scene_name}',
                       name='mapper_'+time_string,
                       settings=wandb.Settings(code_dir="."), dir=self.cfg["wandb_folder"],
                       tags=[scene_name])
            wandb.run.log_code(".")
            wandb.watch((self.decoders.geo_decoder,
                        self.decoders.color_decoder), criterion=None, log="all")

        self.exposure_feat_all = ([] if self.encode_exposure else None)
        _, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])

        self.estimate_c2w_list[0] = gt_c2w.cpu()
        init = True
        prev_idx = -1
        self.prev_c2w = self.estimate_c2w_list[0]
        while (1):
            if not init:
                while True:
                    idx = self.pipe.recv()
                    if idx == self.n_img - 1:
                        break
                    if idx > 0 and idx % self.every_frame == 0:
                        break
            else:
                idx = torch.tensor(0)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx.item()]

            if self.use_dynamic_radius:
                ratio = self.radius_query_ratio
                intensity = rgb2gray(gt_color.cpu().numpy())
                grad_y = filters.sobel_h(intensity)
                grad_x = filters.sobel_v(intensity)
                color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
                color_grad_mag = np.clip(
                    color_grad_mag, 0.0, self.color_grad_threshold)  # range 0~1
                fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                                        self.radius_add_max, self.radius_add_max, self.radius_add_min])
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                          ratio*self.radius_add_max, ratio*self.radius_add_max, ratio*self.radius_add_min])
                dynamic_r_add = fn_map_r_add(color_grad_mag)
                dynamic_r_query = fn_map_r_query(color_grad_mag)
                self.dynamic_r_add, self.dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
                    self.device), torch.from_numpy(dynamic_r_query).to(self.device)
                if init:
                    torch.save(
                        self.dynamic_r_query, f'{self.output}/dynamic_r_frame/r_query_{idx:05d}.pt')

            color_refine = True if (
                idx == self.n_img-1 and self.color_refine) else False

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            new_fragment = self.check_new_fragment(
                method=self.segment_strategy, idx=idx, cur_c2w=cur_c2w)
            if not init:
                num_joint_iters = cfg['mapping']['iters']
                self.mapping_window_size = cfg['mapping']['mapping_window_size']*(
                    2 if self.n_img > 4000 else 1)

                if idx == self.n_img-1 and self.color_refine:  # end of SLAM
                    print("color_refine")
                    self.npc.train_index_global()
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.geo_iter_ratio = 0.4
                    num_joint_iters *= 10
                    self.fix_color_decoder = True
                    self.frustum_feature_selection = False
                    self.keyframe_selection_method = 'segments'
                    self.end = True
                    # self.npc.set_nn_num(16)
                else:
                    outer_joint_iters = 1

            else:
                outer_joint_iters = 1
                num_joint_iters = self.iters_first  # more iters on first run

            num_joint_iters = num_joint_iters//outer_joint_iters

            if new_fragment:
                self.keyframe_dict = []
                self.keyframe_list = []

            #fix dynamic radius and exposure_feat for tum and scannet
            # if self.end:
            #     # self.npc.set_nn_num(16)
            #     for i in range(0, self.n_img, 5):

            #         _, gt_c, gt_d, gt_c2w_refine = self.frame_reader[i]
            #         cur_c2w_refine = self.estimate_c2w_list[i].to(
            #             self.device)
            #         cur_frame_depth, cur_frame_color = self.visualizer.vis_value_only(torch.tensor(i), 0, gt_d, gt_c, cur_c2w_refine, self.npc, self.decoders,
            #                                                                           self.npc.get_geo_feats(self.end), self.npc.get_col_feats(self.end), freq_override=True,
            #                                                                           dynamic_r_query=self.dynamic_r_query, cloud_pos=torch.tensor(self.npc.get_cloud_pos(self.end), device=self.device), exposure_feat=None)

            #         img = cv2.cvtColor(
            #             cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
            #         cv2.imwrite(os.path.join(
            #             f'{self.output}/refinement', f'before_color_{i:05d}.png'), img)

            #         depth = cur_frame_depth.cpu().numpy()
            #         depth = ((depth - 0) / (np.max(depth) - 0))*255
            #         img2 = cv2.applyColorMap(
            #             depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
            #         cv2.imwrite(os.path.join(
            #             f'{self.output}/refinement', f'before_depth_{i:05d}.png'), img2)

            #     _, gt_c, gt_d, gt_c2w_refine = self.frame_reader[self.n_img - 1]
            #     cur_c2w_refine = self.estimate_c2w_list[self.n_img - 1].to(
            #         self.device)
            #     cur_frame_depth, cur_frame_color = self.visualizer.vis_value_only(torch.tensor(self.n_img - 1), 0, gt_d, gt_c, cur_c2w_refine, self.npc, self.decoders,
            #                                                                       self.npc.get_geo_feats(self.end), self.npc.get_col_feats(self.end), freq_override=True,
            #                                                                       dynamic_r_query=self.dynamic_r_query, cloud_pos=torch.tensor(self.npc.get_cloud_pos(self.end), device=self.device), exposure_feat=None)

            #     img = cv2.cvtColor(
            #         cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
            #     cv2.imwrite(os.path.join(
            #         f'{self.output}/refinement', f'before_color_{(self.n_img - 1):05d}.png'), img)

            #     depth = cur_frame_depth.cpu().numpy()
            #     depth = ((depth - 0) / (np.max(depth) - 0))*255
            #     img2 = cv2.applyColorMap(
            #         depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
            #     cv2.imwrite(os.path.join(
            #         f'{self.output}/refinement', f'before_depth_{(self.n_img - 1):05d}.png'), img2)

            #     # self.npc.set_nn_num(8)

            for outer_joint_iter in range(outer_joint_iters):
                # start BA when having enough keyframes
                self.BA = (len(self.keyframe_list) >
                           4) and cfg['mapping']['BA']

                _ = self.optimize_map(num_joint_iters, idx, gt_color, gt_depth, gt_c2w,
                                      self.keyframe_dict_global if self.keyframe_selection_method == "global" else self.keyframe_dict, self.keyframe_list_global if self.keyframe_selection_method == "global" else self.keyframe_list, cur_c2w, color_refine=color_refine, new_fragment=new_fragment)
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

            if not self.end:  # visualization for debugging
                cur_frame_depth, cur_frame_color = self.visualizer.vis_value_only(idx, 0, gt_depth, gt_color, cur_c2w, self.npc, self.decoders,
                                                                                  self.npc_geo_feats, self.npc_col_feats, freq_override=True,
                                                                                  dynamic_r_query=self.dynamic_r_query, cloud_pos=self.cloud_pos_tensor, exposure_feat=self.exposure_feat)
                img = cv2.cvtColor(
                    cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(
                    f'{self.output}/segments', f'color_{idx:05d}.png'), img)

                depth = cur_frame_depth.cpu().numpy()
                depth = ((depth - 0) / (np.max(depth) - 0))*255
                img2 = cv2.applyColorMap(
                    depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
                cv2.imwrite(os.path.join(
                    f'{self.output}/segments', f'depth_{idx:05d}.png'), img2)

            if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) and (idx not in self.keyframe_list) and (not torch.isinf(gt_c2w).any()) and (not torch.isnan(gt_c2w).any()):
                self.keyframe_list.append(idx)
                dic_of_cur_frame = {'gt_c2w': gt_c2w.detach().cpu(), 'idx': idx, 'color': gt_color.detach().cpu(),
                                    'depth': gt_depth.detach().cpu(), 'est_c2w': cur_c2w.clone().detach()}
                if self.use_dynamic_radius:
                    dic_of_cur_frame.update(
                        {'dynamic_r_query': self.dynamic_r_query.detach()})
                if self.encode_exposure:
                    dic_of_cur_frame.update(
                        {'exposure_feat': self.exposure_feat.detach().cpu()})
                self.keyframe_dict.append(dic_of_cur_frame)

            if (idx % self.keyframe_global_every == 0 or (idx == self.n_img-2)) and (idx not in self.keyframe_list_global) and (not torch.isinf(gt_c2w).any()) and (not torch.isnan(gt_c2w).any()):
                self.keyframe_list_global.append(idx)
                dic_of_cur_frame = {'gt_c2w': gt_c2w.detach().cpu(), 'idx': idx, 'color': gt_color.detach().cpu(),
                                    'depth': gt_depth.detach().cpu(), 'est_c2w': cur_c2w.clone().detach()}
                if self.use_dynamic_radius:
                    dic_of_cur_frame.update(
                        {'dynamic_r_query': self.dynamic_r_query.detach()})
                if self.encode_exposure:
                    dic_of_cur_frame.update(
                        {'exposure_feat': self.exposure_feat.detach().cpu()})
                self.keyframe_dict_global.append(dic_of_cur_frame)

            init = False
            self.prev_c2w = self.estimate_c2w_list[idx]

            if (idx % 300 == 0 or idx == self.n_img-1):
                cloud_pos = np.array(self.npc.input_pos())
                cloud_rgb = np.array(self.npc.input_rgb())
                point_cloud = np.hstack((cloud_pos, cloud_rgb))
                #npc_cloud = np.array(self.npc.cloud_pos())
                if idx == self.n_img-1:
                    np.save(f'{self.output}/final_point_cloud', point_cloud)
                    #np.save(f'{self.output}/npc_cloud', npc_cloud)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud_pos)
                    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb/255.0)
                    o3d.io.write_point_cloud(
                        f'{self.output}/final_point_cloud.ply', pcd)
                    print('Saved point cloud and point normals.')
                if self.wandb:
                    wandb.log(
                        {f'Cloud/point_cloud_{idx:05d}': wandb.Object3D(point_cloud)})

            if (idx > 0 and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
                self.logger.log(idx, self.keyframe_dict_global, self.keyframe_list_global,
                                selected_keyframes=self.selected_keyframes
                                if self.save_selected_keyframes_info else None, npc=self.npc, 
                                exposure_feat=self.exposure_feat_all
                                if self.encode_exposure else None,
                                last_log=(idx == self.n_img-1))

            # mapping of first frame is done, can begin tracking
            self.mapping_idx[0] = idx
            self.pipe.send(idx)

            if idx == self.n_img-1:
                print('Color refinement done.')
                print('Mapper finished.')
                break

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            # for test-deterministic.py script
            if cfg["stop"] and idx != 0 and (idx.item() % cfg["stop"] == 0):
                return

        #print registration times info
        registration_times = np.array(self.npc.get_registration_times())
        pgo_times = np.array(self.npc.get_pgo_times())
        print(f"number of pgos : {len(pgo_times)}, average time per pgo : {np.mean(pgo_times)}")
        print(f"number of registrations : {len(registration_times)}, average time per registration {np.mean(registration_times)}")
        try:
            print('✨ Point-SLAM finished, evaluating...')
            ate_rmse = subprocess.check_output(['python', '-u', 'src/tools/eval_ate.py', str(cfg['config_path']), '--output', str(cfg['data']['output'])],
                                               text=True, stderr=subprocess.STDOUT)
            print('ate_rmse: ', ate_rmse)
            ate_rmse = literal_eval(str(ate_rmse))

            ate_rmse_no_align = subprocess.check_output(['python', '-u', 'src/tools/eval_ate.py', str(cfg['config_path']), '--output', str(cfg['data']['output']), '--no_align'],
                                                        text=True, stderr=subprocess.STDOUT)
            print('ate_rmse_wo_align: ', ate_rmse_no_align)
            ate_rmse_no_align = literal_eval(str(ate_rmse_no_align))

            if self.wandb:
                wandb.log(
                    {'ate-rmse': ate_rmse["absolute_translational_error.rmse"]})
                wandb.log(
                    {'ate-rmse-no-align': ate_rmse_no_align["absolute_translational_error.rmse"]})
            print('Successfully evaluated trajectory.')
        except Exception as e:
            traceback.print_exception(e)
            self.save_ckpts = True  # in case needed
            print('Failed to evaluate trajectory.')

        # re-render frames at the end for meshing

        print('Starting re-rendering frames...')
        # self.npc.set_nn_num(16)
        render_idx, frame_cnt, psnr_sum, ssim_sum, lpips_sum, depth_l1_render = 0, 0, 0, 0, 0, 0
        os.makedirs(f'{self.output}/rendered_every_frame', exist_ok=True)
        os.makedirs(f'{self.output}/rendered_image', exist_ok=True)
        if self.eval_img:
            cal_lpips = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=True).to(self.device)
        try:
            while render_idx < self.n_img:
                _, gt_color, gt_depth, gt_c2w = self.frame_reader[render_idx]
                cur_c2w = self.estimate_c2w_list[render_idx].to(
                    self.device)

                if self.encode_exposure:
                    try:
                        state_dict = torch.load(f'{self.output}/ckpts/color_decoder/{render_idx:05}.pt',
                                                map_location=self.device)
                        self.decoders.color_decoder.load_state_dict(
                            state_dict)
                    except:
                        print(
                            'Color decoder not loaded, will use saved weights in checkpoint.')

                r_query_frame = torch.load(f'{self.output}/dynamic_r_frame/r_query_{render_idx:05d}.pt', map_location=self.device) \
                    if self.use_dynamic_radius else None

                cur_frame_depth, cur_frame_color = self.visualizer.vis_value_only(idx, 0, gt_depth, gt_color, cur_c2w, self.npc, self.decoders,
                                                                                    self.npc_geo_feats, self.npc_col_feats, freq_override=True,
                                                                                    dynamic_r_query=r_query_frame, cloud_pos=self.cloud_pos_tensor,
                                                                                    exposure_feat=self.exposure_feat_all[
                                                                                        render_idx // cfg["mapping"]["every_frame"]
                                                                                    ].to(self.device)
                                                                                    if self.encode_exposure else None)
                cur_frame_depth[gt_depth == 0] = 0  
                np.save(f'{self.output}/rendered_every_frame/depth_{render_idx:05d}',
                        cur_frame_depth.cpu().numpy())
                np.save(f'{self.output}/rendered_every_frame/color_{render_idx:05d}',
                        cur_frame_color.cpu().numpy())
                if render_idx % 5 == 0:
                    img = cv2.cvtColor(
                        cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(
                        f'{self.output}/rendered_image', f'frame_{render_idx:05d}.png'), img)

                    depth = cur_frame_depth.cpu().numpy()
                    depth = ((depth - 0) / (np.max(depth) - 0))*255
                    img2 = cv2.applyColorMap(
                        depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
                    cv2.imwrite(os.path.join(
                        f'{self.output}/rendered_image', f'frame_depth_{render_idx:05d}.png'), img2)

                if self.wandb and self.eval_img:
                    mse_loss = torch.nn.functional.mse_loss(
                        gt_color[gt_depth > 0], cur_frame_color[gt_depth > 0])
                    psnr_frame = -10. * torch.log10(mse_loss)
                    ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                            data_range=1.0, size_average=True)
                    lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                            torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
                    psnr_sum += psnr_frame
                    ssim_sum += ssim_value
                    lpips_sum += lpips_value
                    wandb.log({'idx_frame': render_idx,
                                'psnr_frame': psnr_frame})
                depth_l1_render += torch.abs(
                    gt_depth[gt_depth > 0] - cur_frame_depth[gt_depth > 0]).mean().item()
                render_idx += cfg['mapping']['every_frame']
                frame_cnt += 1
                if render_idx % 400 == 0:
                    print(f'frame {render_idx}')
            else:
                # _, gt_c, gt_d, gt_c2w_refine = self.frame_reader[self.n_img - 1]
                # cur_c2w_refine = self.estimate_c2w_list[self.n_img - 1].to(
                #     self.device)

                # r_query_frame = torch.load(f'{self.output}/dynamic_r_frame/r_query_{(self.n_img - 1):05d}.pt', map_location=self.device) \
                #     if self.use_dynamic_radius else None

                # cur_frame_depth, cur_frame_color = self.visualizer.vis_value_only(torch.tensor(self.n_img - 1), 0, gt_d, gt_c, cur_c2w_refine, self.npc, self.decoders,
                #                                                                     self.npc_geo_feats, self.npc_col_feats, freq_override=True,
                #                                                                     dynamic_r_query=r_query_frame, cloud_pos=self.cloud_pos_tensor, exposure_feat=None)

                # img = cv2.cvtColor(cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(os.path.join(f'{self.output}/rendered_image', f'frame_{(self.n_img-1):05d}.png'), img)

                # depth = cur_frame_depth.cpu().numpy()
                # depth = ((depth - 0) / (np.max(depth) - 0))*255
                # img2 = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
                # cv2.imwrite(os.path.join(f'{self.output}/rendered_image', f'frame_depth_{(self.n_img - 1):05d}.png'), img2)

                if self.wandb and self.eval_img:
                    avg_psnr = psnr_sum / frame_cnt
                    avg_ssim = ssim_sum / frame_cnt
                    avg_lpips = lpips_sum / frame_cnt
                    wandb.log({'avg_ms_ssim': avg_ssim})
                    wandb.log({'avg_psnr': avg_psnr})
                    wandb.log({'avg_lpips': avg_lpips})
                print(f'depth_l1_render: {depth_l1_render/frame_cnt}')
                if self.wandb:
                    wandb.log(
                        {'depth_l1_render': depth_l1_render/frame_cnt})
        except Exception as e:
            traceback.print_exception(e)
            print('Re-rendering frames failed.')
        print(f'Finished rendering {frame_cnt} frames.')

        if cfg['meshing']['eval_rec']:
            try:
                print('Evaluating reconstruction...')
                params_list = ['python', '-u', 'src/tools/get_mesh_tsdf_fusion.py',
                                str(cfg['config_path']
                                    ), '--input_folder', cfg['data']['input_folder'],
                                '--output', cfg['data']['output'], '--no_render']
                if cfg['dataset'] != 'replica':
                    params_list.append('--no_eval')

                try:
                    result_recon_obj = subprocess.run(
                        params_list, text=True, check=True, capture_output=True)
                    result_recon = str(result_recon_obj.stdout)
                except subprocess.CalledProcessError as e:
                    print(e.stderr)

                if cfg['dataset'] == 'replica':
                    # requires only one pair {} inside the printed result
                    print(result_recon)
                    start_index = result_recon.find('{')
                    end_index = result_recon.find('}')
                    result_dict = result_recon[start_index:end_index+1]
                    result_dict = literal_eval(result_dict)
                    if self.wandb:
                        wandb.log(result_dict)
                torch.cuda.empty_cache()

            except Exception as e:
                traceback.print_exception(e)
                print('Failed to evaluate 3D reconstruction.')

        if os.path.exists(f'{self.output}/dynamic_r_frame'):
            shutil.rmtree(f'{self.output}/dynamic_r_frame')
        # if os.path.exists(f'{self.output}/rendered_every_frame'):
        #     shutil.rmtree(f'{self.output}/rendered_every_frame')
        if not self.save_ckpts:
            if os.path.exists(f'{self.output}/ckpts'):
                shutil.rmtree(f'{self.output}/ckpts')
        if self.wandb:
            print('wandb finished.')
            wandb.finish()
