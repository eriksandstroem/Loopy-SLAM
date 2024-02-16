import numpy as np
import random
import torch
import torch.nn.functional as F
import cv2
import warnings
import open3d as o3d
import copy

from skimage.color import rgb2gray
from skimage import filters

from time import perf_counter
from contextlib import ContextDecorator
import time


class mytimer(ContextDecorator):
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        elapsed = perf_counter() - self.time
        print(
            f"{self.msg if self.msg is not None else 'Finished'} in {elapsed:.6f} seconds")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics (fx, fy, cx, cy).

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        print('please install torchsearchsorted.')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0 to (l-1)

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.
    i,j are flattened.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)

    rays_d = torch.sum(dirs * c2w[:3, :3], -1)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n pixels (u,v) from dense (u,v).

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]
    j = j[indices]
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth[indices]
    color = color[indices]
    return i, j, depth, color


def sobel_grad(image, ksize=3):
    """
    Args:
        image (torch.tensor): color or depth image
        ksize (int, optional): filter size. Defaults to 3.
    Returns:
        grad_mag: sobel gradient magnitude
    """
    if image.dim() > 2:
        intensity = rgb2gray(image.cpu().numpy())
    else:
        intensity = image.cpu().numpy()
    grad_x = cv2.Sobel(intensity, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(intensity, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))

    return torch.from_numpy(grad_mag).to(image.device)


def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..(H1-1), W0..(W1-1)

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device), indexing='ij')
    i = i.t()
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
    return i, j, depth, color


def get_sample_uv_with_grad(H0, H1, W0, W1, n, image):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1
    image (numpy.ndarray): color image or estimated normal image

    """
    intensity = rgb2gray(image.cpu().numpy())
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    img_size = (image.shape[0], image.shape[1])
    selected_index = np.argpartition(grad_mag, -5*n, axis=None)[-5*n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    mask = (indices_h >= H0) & (indices_h < H1) & (
        indices_w >= W0) & (indices_w < W1)
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(
        np.array((indices_h, indices_w)), img_size)
    samples = np.random.choice(
        range(0, indices_h.shape[0]), size=n, replace=False)

    return selected_index[samples]


def get_selected_index_with_grad(H0, H1, W0, W1, n, image, ratio=15, gt_depth=None, depth_limit=False):
    """
    return uv coordinates with top color gradient from an image region H0..H1, W0..W1

    Args:
        ratio (int): sample from top ratio * n pixels within the region
        gt_depth (torch.Tensor): depth input, will be passed if using self.depth_limit
    Returns:
        selected_index (ndarray): index of top color gradient uv coordinates
    """
    intensity = rgb2gray(image.cpu().numpy())
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # random sample from top 20*n elements within the region
    img_size = (image.shape[0], image.shape[1])
    # try skip the top color grad. pixels
    selected_index = np.argpartition(grad_mag, -ratio*n, axis=None)[-ratio*n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    mask = (indices_h >= H0) & (indices_h < H1) & (
        indices_w >= W0) & (indices_w < W1)
    if gt_depth is not None:
        if depth_limit:
            mask_depth = torch.logical_and((gt_depth[torch.from_numpy(indices_h).to(image.device), torch.from_numpy(indices_w).to(image.device)] <= 5.0),
                                           (gt_depth[torch.from_numpy(indices_h).to(image.device), torch.from_numpy(indices_w).to(image.device)] > 0.0))
        else:
            mask_depth = gt_depth[torch.from_numpy(indices_h).to(
                image.device), torch.from_numpy(indices_w).to(image.device)] > 0.0
        mask = mask & mask_depth.cpu().numpy()
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(
        np.array((indices_h, indices_w)), img_size)

    return selected_index, grad_mag


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device,
                depth_filter=False, return_index=False, depth_limit=None):
    """
    Get n rays from the image region H0..H1, W0..W1.
    H, W: image size.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, depth, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[
            mask], rays_d[mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color


def get_samples_with_pixel_grad(H0, H1, W0, W1, n_color, H, W, fx, fy, cx, cy, c2w, depth, color, device,
                                depth_filter=True, return_index=True, depth_limit=None):
    """
    Get n rays from the image region H0..H1, W0..W1 based on color gradients, normal map gradients and random selection
    H, W: height, width.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """

    assert (n_color > 0), 'invalid number of rays to sample.'

    index_color_grad, index_normal_grad = [], []
    if n_color > 0:
        index_color_grad = get_sample_uv_with_grad(
            H0, H1, W0, W1, n_color, color)

    merged_indices = np.union1d(index_color_grad, index_normal_grad)

    i, j = np.unravel_index(merged_indices.astype(int), (H, W))
    i, j = torch.from_numpy(j).to(device).float(), torch.from_numpy(
        i).to(device).float()  # (i-cx), on column axis
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    i, j = i.long(), j.long()
    sample_depth = depth[j, i]
    sample_color = color[j, i]
    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[
            mask], rays_d[mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    Returns:
        torch.Tensor(N*3*4 if batch input or 3*4): Transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_rotation_from_tensor(quad):
    N = len(quad.shape)
    if N == 1:
        quad = quad.unsqueeze(0)
    R = quad2rotation(quad).squeeze(0)
    return R


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]

    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    quad = rot.as_quat()
    quad = np.roll(quad, 1)

    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, device='cuda:0', coef=0.1):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, (N_rays,N_samples,4) ): prediction from model. i.e. (R,G,B) and density σ
        z_vals (tensor, (N_rays,N_samples) ): integration time. i.e. the sampled locations on this ray
        rays_d (tensor, (N_rays,3) ): direction of each ray.
        device (str, optional): device. Defaults to 'cuda:0'.
        coef (float, optional): to multipy the input of sigmoid function when calculating occupancy. Defaults to 0.1.    

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty along the ray, see eq(7) in paper.
        rgb_map (tensor, (N_rays,3)): estimated RGB color of a ray.
        weights (tensor, (N_rays,N_samples) ): weights assigned to each sampled color.
    """

    # def raw2alpha(raw, dists, act_fn=F.relu):
    #     return 1. - torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]

    raw[..., -1] = torch.sigmoid(coef*raw[..., -1])
    alpha = raw[..., -1]

    weights = alpha.float() * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(device).float(), (1.-alpha + 1e-10).float()], -1).float(), dim=-1)[:, :-1]
    weights_sum = torch.sum(weights, dim=-1).unsqueeze(-1)+1e-10
    rgb_map = torch.sum(weights[..., None] * rgb, -2)/weights_sum
    depth_map = torch.sum(weights * z_vals, -1)/weights_sum.squeeze(-1)

    tmp = (z_vals-depth_map.unsqueeze(-1))
    depth_var = torch.sum(weights*tmp*tmp, dim=1)
    return depth_map, depth_var, rgb_map, weights


def get_rays(H, W, fx, fy, cx, cy, c2w, device, crop_edge=0):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    i, j = torch.meshgrid(torch.linspace(crop_edge, W-1-crop_edge, W-2*crop_edge),
                          torch.linspace(crop_edge, H-1-crop_edge, H-2*crop_edge), indexing='ij')
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H-2*crop_edge, W-2*crop_edge, 1, 3)
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p


def normalization(data):
    """
    Normalize normal direction.

    Args:
        data(np.array): (H,W,3)

    Returns:
        directions(np.array): _description_
    """
    l2_norm = np.sqrt(np.multiply(data[:, :, 0], data[:, :, 0])+np.multiply(
        data[:, :, 1], data[:, :, 1])+np.multiply(data[:, :, 2], data[:, :, 2]))
    l2_norm = np.dstack((l2_norm, l2_norm, l2_norm))
    return data/(l2_norm+1e-10)


def masked_psnr(img1, img2, mask):
    mse = torch.mean((img1[mask] - img2[mask]) ** 2)
    if mse == 0:
        return 100
    return - 10 * torch.log10(mse)


def cal_depth_continuity(depth_map, row, col, patch_size=5):
    """
    Calculate the continuity of the depth map.

    Args:
        depth_map (tensor): depth map
        row (int): row index
        col (int): col index
        patch_size (int): patch size
    """
    row_start = max(row - patch_size // 2, 0)
    row_end = min(row + patch_size // 2, depth_map.shape[0])
    col_start = max(col - patch_size // 2, 0)
    col_end = min(col + patch_size // 2, depth_map.shape[1])
    patch = depth_map[row_start:row_end, col_start:col_end]

    std = torch.std(patch)
    return std


def get_npc_input_pcl(npc):
    cloud_pos = np.array(npc.input_pos())
    cloud_rgb = np.array(npc.input_rgb())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_pos)
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb/255.0)

    return pcd


def clone_kf_dict(keyframes_dict):
    cloned_keyframes_dict = []
    for keyframe in keyframes_dict:
        cloned_keyframe = {}
        for key, value in keyframe.items():
            cloned_value = value.clone()
            cloned_keyframe[key] = cloned_value
        cloned_keyframes_dict.append(cloned_keyframe)
    return cloned_keyframes_dict


# def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = pcd.voxel_down_sample(voxel_size)

#     if not pcd.has_normals():
#         radius_normal = voxel_size * 2
#         print(":: Estimate normal with search radius %.3f." % radius_normal)
#         pcd_down.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#     radius_feature = 0.15
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh

def preprocess_point_cloud(pcd, voxel_size, camera_location):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))

    pcd_down.orient_normals_towards_camera_location(
        camera_location=camera_location)

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, global_iter, conf):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    # print("   Downsampling voxel size is %.3f," % voxel_size)
    # print("   Using a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(global_iter, conf))
    return result

def pairwise_registration(segment_source, segment_target, method="gt", global_iter=10000000, conf=0.99999):

    max_correspondence_distance_coarse = 0.3
    max_correspondence_distance_fine = 0.03

    source_points = segment_source["points"]
    target_points = segment_target["points"]

    source_colors = segment_source["points_color"]
    target_colors = segment_target["points_color"]

    cloud_source = o3d.geometry.PointCloud()
    cloud_source.points = o3d.utility.Vector3dVector(np.array(source_points))
    cloud_source.colors = o3d.utility.Vector3dVector(np.array(source_colors))
    cloud_source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    keyframe_source = segment_source["keyframe"]
    camera_location_source = keyframe_source[:3, 3].cpu().numpy()
    cloud_source.orient_normals_towards_camera_location(
        camera_location=camera_location_source)

    cloud_target = o3d.geometry.PointCloud()
    cloud_target.points = o3d.utility.Vector3dVector(np.array(target_points))
    cloud_target.colors = o3d.utility.Vector3dVector(np.array(target_colors))
    cloud_target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    keyframe_target = segment_target["keyframe"]
    camera_location_target = keyframe_target[:3, 3].cpu().numpy()
    cloud_target.orient_normals_towards_camera_location(
        camera_location=camera_location_target)
    
    output = dict()
    if method == "gt":

        gt_source = segment_source["gt_camera"]
        gt_target = segment_target["gt_camera"]

        delta_gt = gt_source@gt_target.inverse()

        delta = delta_gt@keyframe_target@keyframe_source.inverse()

        output["transformation"] = np.array(delta)
    elif method == "icp":
        icp_coarse = o3d.pipelines.registration.registration_icp(
            cloud_source, cloud_target, max_correspondence_distance_coarse, np.identity(
                4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            cloud_source, cloud_target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        delta = icp_fine.transformation
        output["transformation"] = np.array(delta)
    elif method == "robust_icp":
        voxel_size = 0.04
        sigma = 0.01

        source_down, source_fpfh = preprocess_point_cloud(
            cloud_source, voxel_size, camera_location_source)
        target_down, target_fpfh = preprocess_point_cloud(
            cloud_target, voxel_size, camera_location_target)
        
        tic = time.perf_counter()

        result_ransac = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size, global_iter, conf)

        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        icp_fine = o3d.pipelines.registration.registration_icp(
            cloud_source, cloud_target, max_correspondence_distance_fine,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
        
        toc = time.perf_counter()
        delta = icp_fine.transformation

        # compute success to gt delta
        gt_source = segment_source["gt_camera"]
        gt_target = segment_target["gt_camera"]
        rel_gt = gt_source@gt_target.inverse()
        delta_gt = rel_gt@keyframe_target@keyframe_source.inverse()
        
        output["transformation_gt"] = np.array(delta_gt)
        output["transformation_gt_mag"] = torch.abs(get_tensor_from_camera(delta_gt)).mean().item()
        output["transformation_mag"] = torch.abs(
            torch.tensor(delta)).mean().item()
        output["transformation_transl_mag"] = torch.abs(torch.tensor(delta[:3, -1])).mean().item()
        output["transformation_transl_err"] = torch.abs(torch.tensor(delta[:3, -1]) - torch.tensor(delta_gt[:3, -1])).mean().item()
        output["transformation"] = np.array(delta)
        output["fitness"] = icp_fine.fitness
        output["inlier_rmse"] = icp_fine.inlier_rmse
        output["registration_time"] = toc-tic

    elif method == "colored_icp":
        try:
            voxel_radius = [0.04, 0.02, 0.01]
            max_iter = [50, 30, 14]
            current_transformation = np.identity(4)
            print("3. Colored point cloud registration")
            for scale in range(3):
                iter = max_iter[scale]
                radius = voxel_radius[scale]
                print([iter, radius, scale])

                print("3-1. Downsample with a voxel size %.2f" % radius)
                source_down = cloud_source.voxel_down_sample(radius)
                target_down = cloud_target.voxel_down_sample(radius)

                print("3-2. Estimate normal.")
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                print("3-3. Applying colored point cloud registration")
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                      relative_rmse=1e-6,
                                                                      max_iteration=iter))
                delta = result_icp.transformation
                output["transformation"] = np.array(delta)
        except:
            icp_coarse = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_coarse, np.identity(
                    4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            delta = icp_fine.transformation
            output["transformation"] = np.array(delta)
    elif method == "identity":
        print("identity")
        delta = np.identity(4)
        output["transformation"] = delta

    else:
        raise NotImplementedError

    output["information"] = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        cloud_source,
        cloud_target,
        max_correspondence_distance_fine,
        np.array(delta)
    )

    output["n_points"] = min(len(cloud_source.points),
                             len(cloud_target.points))

    return output


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, init_trans):
    distance_threshold = voxel_size
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
    return result


def compute_cos_rel_rot(c2w_1, c2w_2):
    rot1_c2w = c2w_1[:3, :3]
    rot2_c2w = c2w_2[:3, :3].to(c2w_1.device)
    optical_axis = torch.zeros(3).to(c2w_1.device)
    optical_axis[2] = 1
    c1_optical_axis_w = torch.matmul(rot1_c2w, optical_axis)
    c2_optical_axis_w = torch.matmul(rot2_c2w, optical_axis)
    vec_c1 = c1_optical_axis_w
    vec_c2 = c2_optical_axis_w
    cos = torch.dot(vec_c1, vec_c2)
    return cos


def compute_rel_trans(c1, c2):

    t_c1 = c1[:3, -1]
    t_c2 = c2[:3, -1].to(c1.device)

    return (t_c2-t_c1).norm(2)


class matching_result:

    def __init__(self, s, t):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = np.identity(4)
        self.information = np.identity(6)
        self.transformation_gt = np.identity(4)
        self.fitness = -1
        self.inlier_rmse = -1
        self.transformation_gt_mag = -1
        self.transformation_mag = -1
        self.transformation_transl_mag = -1
        self.transformation_transl_err = -1

    def __str__(self) -> str:
        return f"source : {self.s}, target : {self.t}, success : {self.success}, \
            transformation : {self.transformation}, information : {self.information}, \
            fitness : {self.fitness}, inlier_rmse : {self.inlier_rmse}, transformation_gt_mag : {self.transformation_gt_mag}, \
            transformation_mag : {self.transformation_mag}, transformation_gt : {self.transformation_gt}, \
            transformation_transl_mag : {self.transformation_transl_mag}, transformation_transl_err : {self.transformation_transl_err}"

    def __repr__(self) -> str:
        return self.__str__()


def update_posegraph_for_scene(s, t, transformation, information,
                               pose_graph):
    if t == s + 1:  # odometry case
        # always identity in our case
        # odometry = np.dot(transformation, odometry)
        # odometry_inv = np.linalg.inv(odometry)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=True))
    return pose_graph

def register_point_cloud_pair(s, t, s_segment, t_segment, method, global_iter, conf):
    output = pairwise_registration(s_segment, t_segment, method, global_iter, conf)

    if t != s + 1:
        if (output["transformation"].trace() == 4.0) or output["information"][5, 5] / output["n_points"] < 0.3:
            output["success"] = False
            output["transformation"] = np.identity(4)
            output["information"] = np.identity(6)
            return output

    output["success"] = True
    return output
