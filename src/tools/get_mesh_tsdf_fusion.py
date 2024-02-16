import sys
sys.path.append('.')
from src.common import as_intrinsics_matrix
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.Point_SLAM import Point_SLAM
from src import config
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import torch
import numpy as np
import argparse
import random
import os
import subprocess
import traceback
from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
import cv2



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_neural_point_cloud(slam, ckpt, device, use_exposure=False):

    slam.npc._cloud_pos = ckpt['cloud_pos']
    slam.npc._input_pos = ckpt['input_pos']
    slam.npc._input_rgb = ckpt['input_rgb']
    slam.npc._input_normal = ckpt['input_normal']
    slam.npc._input_normal_cartesian = ckpt['input_normal_cartesian']
    slam.npc._pts_num = len(ckpt['cloud_pos'])
    slam.npc.geo_feats = ckpt['geo_feats'].to(device)
    slam.npc.col_feats = ckpt['col_feats'].to(device)
    if use_exposure:
        assert 'exposure_feat_all' in ckpt.keys(
        ), 'Please check if exposure feature is encoded.'
        slam.mapper.exposure_feat_all = ckpt['exposure_feat_all'].to(device)

    cloud_pos = torch.tensor(ckpt['cloud_pos'], device=device)
    slam.npc.index_train(cloud_pos)
    slam.npc.index.add(cloud_pos)

    print(
        f'Successfully loaded neural point cloud, {slam.npc.index.ntotal} points in total.')


def load_ckpt(cfg, slam):
    """
    Saves mesh of already reconstructed model from checkpoint file. Makes it 
    possible to remesh reconstructions with different settings and to draw the cameras
    """

    assert cfg['mapping']['save_selected_keyframes_info'], 'Please save keyframes info to help run this code.'

    ckptsdir = f'{slam.output}/ckpts'
    device = cfg['mapping']['device']
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('\nGet ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
        else:
            raise ValueError(f'Check point directory {ckptsdir} is empty.')
    else:
        raise ValueError(f'Check point directory {ckptsdir} not found.')

    return ckpt


class DepthImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('depth_')])
        self.image_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('color_')])

        indices = []
        for depth_file in self.depth_files:
            base, ext = os.path.splitext(depth_file)
            index = int(base[-5:])
            indices.append(index)
        self.indices = indices

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = np.load(self.depth_files[idx])
        image = np.load(self.image_files[idx])

        if self.transform:
            depth = self.transform(depth)
            image = self.transform(image)

        return depth, image


def main():
    parser = argparse.ArgumentParser(
        description="Configs for Point-SLAM."
    )
    parser.add_argument(
        "config", type=str, help="Path to config file.",
    )
    parser.add_argument("--input_folder", type=str,
                        help="input folder, this have higher priority, can overwrite the one in config file.",
                        )
    parser.add_argument("--output", type=str,
                        help="output folder, this have higher priority, can overwrite the one in config file.",
                        )
    parser.add_argument("--name", type=str,
                        help="specify the name of the mesh",
                        )
    parser.add_argument("--no_render", default=False, action='store_true',
                        help="if to render frames from checkpoint for constructing the mesh.",
                        )
    parser.add_argument("--exposure_avail", default=False, action='store_true',
                        help="if the exposure information is available for rendering.",
                        )
    parser.add_argument("-s", "--silent", default=False, action='store_true',
                        help="if to print status message.",
                        )
    parser.add_argument("--no_eval", default=False, action='store_true',
                        help="if to evaluate the mesh by 2d and 3d metrics.",
                        )

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')

    def optional_int(string):
        return None if string == "None" else int(string)
    parser.add_argument("--stop", type=optional_int, help="stop after n frames")

    args = parser.parse_args()
    assert torch.cuda.is_available(), 'GPU required for reconstruction.'
    cfg = config.load_config(args.config, "configs/point_slam.yaml")
    device = cfg['mapping']['device']

    # define variables for dynamic query radius computation
    radius_add_max = cfg['pointcloud']['radius_add_max']
    radius_add_min = cfg['pointcloud']['radius_add_min']
    radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
    color_grad_threshold = cfg['pointcloud']['color_grad_threshold']

    slam = Point_SLAM(cfg, args, share_npc=False,
                      share_decoders=False if args.no_render else True)
    slam.output = cfg['data']['output'] if args.output is None else args.output
    print("args : ", args.output, "out :", slam.output)
    ckpt = load_ckpt(cfg, slam)

    render_frame = not args.no_render
    use_exposure = args.exposure_avail
    if render_frame:
        load_neural_point_cloud(slam, ckpt, device, use_exposure=use_exposure)
        idx = 0
        frame_cnt = 0

        try:
            slam.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])
            if not args.silent:
                print('Successfully loaded decoders.')
        except Exception as e:
            print(e)
        frame_reader = get_dataset(cfg, args, device=device)
        visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                vis_dir=os.path.join(slam.output, 'rendered_every_frame'), renderer=slam.renderer_map,
                                verbose=slam.verbose, device=device, wandb=False)

        if not args.silent:
            print('Starting to render frames...')
        last_idx = (ckpt['idx']+1) if (ckpt['idx'] +
                                       1) < len(frame_reader) else len(frame_reader)
        while idx < last_idx:
            _, gt_color, gt_depth, gt_c2w = frame_reader[idx]
            cur_c2w = ckpt['estimate_c2w_list'][idx].to(device)

            if use_exposure:
                try:
                    state_dict = torch.load(f'{slam.output}/ckpts/color_decoder/{idx:05}.pt',
                                            map_location=device)
                    slam.shared_decoders.color_decoder.load_state_dict(
                        state_dict)
                except Exception as e:
                    print(e)
                    raise ValueError(
                        f'Cannot load per mapping-frame color decoder at frame {idx}.')

            ratio = radius_query_ratio
            intensity = rgb2gray(gt_color.cpu().numpy())
            grad_y = filters.sobel_h(intensity)
            grad_x = filters.sobel_v(intensity)
            color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
            color_grad_mag = np.clip(
                color_grad_mag, 0.0, color_grad_threshold)  # range 0~1

            fn_map_r_query = interp1d([0, 0.01, color_grad_threshold], [
                                        ratio*radius_add_max, ratio*radius_add_max, ratio*radius_add_min])
            dynamic_r_query = fn_map_r_query(color_grad_mag)
            dynamic_r_query = torch.from_numpy(dynamic_r_query).to(device)

            cur_frame_depth, cur_frame_color = visualizer.vis_value_only(idx, 0, gt_depth, gt_color, cur_c2w, slam.npc, slam.shared_decoders,
                                                                         slam.npc.geo_feats, slam.npc.col_feats, freq_override=True,
                                                                         dynamic_r_query=dynamic_r_query, exposure_feat=slam.mapper.exposure_feat_all[
                                                                             idx // cfg["mapping"]["every_frame"]
                                                                         ].to(device) if use_exposure else None)
            cur_frame_depth[gt_depth == 0] = 0
            np.save(f'{slam.output}/rendered_every_frame/depth_{idx:05d}',
                    cur_frame_depth.cpu().numpy())
            np.save(f'{slam.output}/rendered_every_frame/color_{idx:05d}',
                    cur_frame_color.cpu().numpy())
            img = cv2.cvtColor(cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(
                    f'{slam.output}/rendered_image', f'frame_{idx:05d}_notload.png'), img)
            idx += cfg['mapping']['every_frame']
            frame_cnt += 1
        if not args.silent:
            print(f'Finished rendering {frame_cnt} frames.')

    dataset = DepthImageDataset(root_dir=slam.output+'/rendered_every_frame')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_folder = cfg['data']['input_folder'] if args.input_folder is None else args.input_folder
    scene_name = input_folder.split('/')[-1]
    mesh_name = f'{scene_name}_pred_mesh.ply' if args.name is None else args.name
    mesh_out_file = f'{slam.output}/mesh/{mesh_name}'
    mesh_align_file = f'{slam.output}/mesh/mesh_tsdf_fusion_aligned.ply'

    H, W, fx, fy, cx, cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    if not args.silent:
        print('Starting to integrate the mesh...')
    cam_points = []
    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)
    
    os.makedirs(f'{slam.output}/mesh/mid_mesh', exist_ok=True)

    for i, (depth, color) in enumerate(dataloader):
        index = dataset.indices[i]
        depth = depth[0].cpu().numpy()
        color = color[0].cpu().numpy()
        c2w = ckpt['estimate_c2w_list'][index].cpu().numpy()

        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0
        w2c = np.linalg.inv(c2w)
        cam_points.append(c2w[:3, 3])

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(
            np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

        if i > 0 and ((i+1) % cfg["meshing"]["mesh_freq"]) == 1:
            o3d_mesh = volume.extract_triangle_mesh()
            o3d_mesh = o3d_mesh.translate(compensate_vector)
            o3d.io.write_triangle_mesh(
                f'{slam.output}/mesh/mid_mesh/frame_{5*i}_mesh.ply', o3d_mesh)
            print(f"saved intermediate mesh until frame {5*i}.")

    o3d_mesh = volume.extract_triangle_mesh()
    np.save(os.path.join(f'{slam.output}/mesh',
            'vertices_pos.npy'), np.asarray(o3d_mesh.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    if not args.silent:
        print('🕹️ Meshing finished.')

    eval_recon = not args.no_eval
    if eval_recon:
        try:
            if cfg['dataset'] == 'replica':
                print('Evaluating...')
                result_recon_obj = subprocess.run(['python', '-u', 'src/tools/eval_recon.py', '--rec_mesh',
                                                   mesh_out_file,
                                                   '--gt_mesh', f'cull_replica_mesh/{scene_name}.ply', '-3d', '-2d'],
                                                  text=True, check=True, capture_output=True)
                result_recon = result_recon_obj.stdout
                print(result_recon)
                print('✨ Successfully evaluated 3D reconstruction.')
            else:
                print('Current dataset not supported for evaluating 3D reconstruction.')
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print('Failed to evaluate 3D reconstruction.')


if __name__ == "__main__":
    main()

