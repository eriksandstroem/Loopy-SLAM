import torch
import warnings
from src.common import get_rays, raw2outputs_nerf_color


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=3000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.N_surface = cfg['rendering']['N_surface']
        self.near_end_surface = cfg['rendering']['near_end_surface']
        self.far_end_surface = cfg['rendering']['far_end_surface']
        self.sample_near_pcl = cfg['rendering']['sample_near_pcl']
        self.skip_zero_depth_pixel = cfg['rendering']['skip_zero_depth_pixel']

        self.near_end = cfg['rendering']['near_end']

        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.crop_edge = 0 if cfg['cam']['crop_edge'] is None else cfg['cam']['crop_edge']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, npc, stage='color', device=None,
                    npc_geo_feats=None, npc_col_feats=None,
                    is_tracker=False, cloud_pos=None,
                    pts_views_d=None, ray_pts_num=None,
                    dynamic_r_query=None, exposure_feat=None):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            npc (): Neural point cloud.
            stage (str, optional): 'geometry'|'color', defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.
            is_tracker(bool): used only in Tracker process
            cloud_pos(tensor): used only in Tracker process
        Returns:
            ret (tensor): occupancy (and color) value of input points, (N,)
            valid_ray_mask (tensor): 
        """
        assert torch.is_tensor(p)
        if device == None:
            device = npc.device()
        p_split = torch.split(p, self.points_batch_size)
        dynamic_r_query_split = torch.split(dynamic_r_query, self.points_batch_size) if dynamic_r_query is not None else [None]*len(p_split)
        rets = []
        ray_masks = []
        point_masks = []
        for pi, r_query in zip(p_split, dynamic_r_query_split):
            pi = pi.unsqueeze(0)
            ret, valid_ray_mask, point_mask = decoders(pi, npc, stage, npc_geo_feats, npc_col_feats,
                                                       ray_pts_num, is_tracker, cloud_pos, pts_views_d,
                                                       r_query, exposure_feat)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            rets.append(ret)
            ray_masks.append(valid_ray_mask)
            point_masks.append(point_mask)

        ret = torch.cat(rets, dim=0)
        ray_mask = torch.cat(ray_masks, dim=0)
        point_mask = torch.cat(point_masks, dim=0)

        return ret, ray_mask, point_mask

    def render_batch_ray(self, npc, decoders, rays_d, rays_o, device, stage, gt_depth=None,
                         npc_geo_feats=None, npc_col_feats=None, is_tracker=False, cloud_pos=None,
                         dynamic_r_query=None, exposure_feat=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            npc (): Neural point cloud.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None. input (N, )
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc.
            npc_col_feats (tensor): point cloud color features.
            is_tracker (bool, optional): tracker has different gradient flow in eval_points.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
            valid_ray_mask (tensor): filter coner cases.
        """

        N_surface = self.N_surface
        N_rays = rays_o.shape[0]
        near_end = self.near_end

        with torch.no_grad():
            
            if gt_depth is not None:
                far_bb = torch.minimum(
                    5*gt_depth.mean(), torch.max(gt_depth*1.2)).repeat(rays_o.shape[0], 1).float()
            else:
                far_bb = 10 * \
                    torch.ones((rays_o.shape[0], 1), device=device).float()

        if gt_depth is None:
            gt_depth = torch.zeros(N_rays, 1, device=device)
            near = 0.3
            far = far_bb
        elif torch.numel(gt_depth) != 0:
            gt_depth = gt_depth.reshape(-1, 1)

            if torch.max(gt_depth) > 0.:
                far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
            else:  # current batch, gt_depth all 0.0
                far = far_bb
        else:
            # handle error, gt_depth is empty
            warnings.warn('tensor gt_depth is empty, info:')
            print('rays_o', rays_o.shape, 'rays_d', rays_d.shape,
                  'gt_depth', gt_depth, 'is_tracker', is_tracker)
            gt_depth = torch.zeros(N_rays, 1, device=device)
            far = far_bb

        if N_surface > 0:
            gt_none_zero_mask = gt_depth > 0
            gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
            mask_rays_near_pcl = torch.ones(
                N_rays, device=device).type(torch.bool)

            gt_none_zero = gt_depth[gt_none_zero_mask]
            gt_depth_surface = gt_none_zero.repeat(
                1, N_surface)

            t_vals_surface = torch.linspace(
                0.0, 1.0, steps=N_surface, device=device)
            
            z_vals_surface_depth_none_zero = self.near_end_surface * gt_depth_surface * \
                (1.-t_vals_surface) + self.far_end_surface * gt_depth_surface * \
                (t_vals_surface)

            z_vals_surface = torch.zeros(
                gt_depth.shape[0], N_surface, device=device)
            z_vals_surface[gt_none_zero_mask,
                           :] = z_vals_surface_depth_none_zero
            if gt_none_zero_mask.sum() < N_rays:
                if self.sample_near_pcl:
                    z_vals_depth_zero, mask_not_near_pcl = npc.sample_near_pcl(rays_o[~gt_none_zero_mask].clone().detach(),
                                                                               rays_d[~gt_none_zero_mask].clone().detach(),
                        near_end, torch.max(far), N_surface)
                    if torch.sum(mask_not_near_pcl.ravel()):
                        rays_not_near = torch.nonzero(~gt_none_zero_mask, as_tuple=True)[
                            0][mask_not_near_pcl]
                        mask_rays_near_pcl[rays_not_near] = False
                    z_vals_surface[~gt_none_zero_mask, :] = z_vals_depth_zero
                else:
                    z_vals_surface[~gt_none_zero_mask, :] = torch.linspace(near_end, torch.max(
                        far), steps=N_surface, device=device).repeat((~gt_none_zero_mask).sum(), 1)

        z_vals = z_vals_surface

        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pointsf = pts.reshape(-1, 3)

        ray_pts_num = N_surface
        rays_d_pts = rays_d.repeat_interleave(
            ray_pts_num, dim=0).reshape(-1, 3)
        if self.use_dynamic_radius:
            dynamic_r_query = dynamic_r_query.reshape(
                -1, 1).repeat_interleave(ray_pts_num, dim=0)

        raw, valid_ray_mask, point_mask = self.eval_points(
            pointsf, decoders, npc, stage, device, npc_geo_feats,
            npc_col_feats, is_tracker, cloud_pos, rays_d_pts,
            ray_pts_num=ray_pts_num, dynamic_r_query=dynamic_r_query,
            exposure_feat=exposure_feat)

        with torch.no_grad():
            raw[torch.nonzero(~point_mask).flatten(), -1] = - \
                100.0
        raw = raw.reshape(N_rays, ray_pts_num, -1)
        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, device=device, coef=self.sigmoid_coefficient)

        # filter two corner cases:
        # 1. rays has no gt_depth and it's not close to the current npc
        # 2. rays has gt_depth, but all its sampling locations have no neighbors in current npc
        mask_not_near_pcl = torch.nonzero(valid_ray_mask, as_tuple=True)[0]
        valid_ray_mask = valid_ray_mask & mask_rays_near_pcl

        if not self.sample_near_pcl:
            depth[~gt_none_zero_mask] = 0
        if self.skip_zero_depth_pixel:
            color[~gt_none_zero_mask] = torch.zeros((1, 3), device=device)
        return depth, uncertainty, color, valid_ray_mask

    def render_img(self, npc, decoders, c2w, device, stage, gt_depth=None,
                   npc_geo_feats=None, npc_col_feats=None,
                   dynamic_r_query=None, cloud_pos=None, exposure_feat=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            npc (): Neural point cloud.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            # for all pixels, considering cropped edges
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)  # (H, W, 3)->(H*W, 3)
            rays_d = rays_d.reshape(-1, 3)
            if self.use_dynamic_radius:
                dynamic_r_query = dynamic_r_query.reshape(-1, 1)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            # run batch by batch
            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:       
                    ret = self.render_batch_ray(
                        npc, decoders, rays_d_batch, rays_o_batch, device, stage,
                        gt_depth=None, npc_geo_feats=npc_geo_feats, npc_col_feats=npc_col_feats,
                        cloud_pos=cloud_pos,
                        dynamic_r_query=dynamic_r_query[i:i +
                                                        ray_batch_size] if self.use_dynamic_radius else None,
                        exposure_feat=exposure_feat)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(
                        npc, decoders, rays_d_batch, rays_o_batch, device, stage,
                        gt_depth=gt_depth_batch, npc_geo_feats=npc_geo_feats, npc_col_feats=npc_col_feats,
                        cloud_pos=cloud_pos,
                        dynamic_r_query=dynamic_r_query[i:i +
                                                        ray_batch_size] if self.use_dynamic_radius else None,
                        exposure_feat=exposure_feat)

                depth, uncertainty, color, valid_ray_mask = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                # list of tensors here
                color_list.append(color)

            # cat to one big tensor
            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color