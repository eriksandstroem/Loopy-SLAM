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
import time
from viz import SLAMFrontend
from tqdm import tqdm
import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ckpt(cfg, output):
    """
    Saves mesh of already reconstructed model from checkpoint file. Makes it 
    possible to remesh reconstructions with different settings and to draw the cameras
    """

    assert cfg['mapping']['save_selected_keyframes_info'], 'Please save keyframes info to help run this code.'

    ckptsdir = f'{output}/ckpts'
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

# Define a custom sorting key function
def get_number(filename):
    return int(filename.split("/")[-1].split(".")[0])

class DepthImageDataset(Dataset):
    def __init__(self, root_dir, png_scale=None):
        self.root_dir = root_dir
        self.png_scale = png_scale
        self.depth_files = sorted([os.path.join(root_dir, "depth", f) for f in os.listdir(
            root_dir + "/depth")], key=get_number)
        self.image_files = sorted([os.path.join(root_dir, "color", f) for f in os.listdir(
            root_dir + "/color")], key=get_number)

        indices = []
        for depth_file in self.depth_files:
            base, ext = os.path.splitext(depth_file)
            index = base.split("/")[-1]
            indices.append(index)
        self.indices = indices

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        depth = depth.astype(np.float32) / self.png_scale
        H, W = depth.shape
        image = cv2.resize(image, (W, H))

        return depth, image


def main():
    parser = argparse.ArgumentParser(
        description="Configs for Point-SLAM."
    )
    parser.add_argument(
        "config", type=str, help="Path to config file.",
    )
    parser.add_argument("--output", type=str,
                        help="output folder, this have higher priority, can overwrite the one in config file.",
                        )

    args = parser.parse_args()
    cfg = config.load_config(args.config, "configs/point_slam.yaml")


    output = cfg['data']['output'] if args.output is None else args.output
    print("args : ", args.output, "out :", output)
    # load ckpt to get the start and end of submap indices

    ckpt = load_ckpt(cfg, output)

    # extract start and end indices of submaps from ckpt
    submap_indices = []
    for fragment in ckpt["fragments"].keys():
        submap_indices.append(ckpt["fragments"][fragment]["start_idx"])

    # load the final poses which is where we will anchor the submap meshes
    gt_c2w_list = ckpt['gt_c2w_list'] 
    estimate_c2w_list = ckpt['estimate_c2w_list']      

    # path to RGBD data:
    # rgbd_root = "/home/esandstroem/scratch-second/data/scannet/scene0054_00/frames"
    # # dataset = DepthImageDataset(root_dir=output+'/mapping_vis')
    # dataset = DepthImageDataset(root_dir=rgbd_root, png_scale=cfg["cam"]["png_depth_scale"])

    # H, W, fx, fy, cx, cy = cfg["cam"]["H"], cfg["cam"]["W"], cfg["cam"]["fx"], cfg["cam"]["fy"], cfg["cam"]["cx"], cfg["cam"]["cy"]

    # To make the submap meshes
    # for k in range(len(submap_indices)):
    #     volume = o3d.pipelines.integration.ScalableTSDFVolume(
    #         voxel_length=5.0 * scale / 512.0,
    #         sdf_trunc=0.04 * scale,
    #         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    #     print('Starting to integrate the mesh for submap: ', submap_indices[k])

    #     os.makedirs(f'{output}/mesh/mid_mesh/submap_' + str(k), exist_ok=True)
    #     startIdx = submap_indices[k]
    #     if k == len(submap_indices) - 1:
    #         endIdx = len(dataset)
    #     else:
    #         endIdx = submap_indices[k+1]

    #     for i in range(startIdx, endIdx):
    #         if i % 5 == 0:
    #             depth, color = dataset[i]
    #             c2w = ckpt['estimate_c2w_list'][i]

    #             c2w[:3, 1] *= -1.0
    #             c2w[:3, 2] *= -1.0
    #             w2c = np.linalg.inv(c2w)

    #             depth = o3d.geometry.Image(depth.astype(np.float32))
    #             color = o3d.geometry.Image(
    #                 np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

    #             intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    #             rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #                 color,
    #                 depth,
    #                 depth_scale=1.0,
    #                 depth_trunc=30,
    #                 convert_rgb_to_intensity=False)
    #             volume.integrate(rgbd, intrinsic, w2c)
                
    #             o3d_mesh = volume.extract_triangle_mesh()
    #             o3d.io.write_triangle_mesh(
    #                 f'{output}/mesh/mid_mesh/submap_{k}/frame_{i}_mesh.ply', o3d_mesh)

    #     o3d_mesh = volume.extract_triangle_mesh()
    #     o3d.io.write_triangle_mesh(f'{output}/mesh/mid_mesh/submap_{k}/frame_{i}_mesh.ply', o3d_mesh)


    # The next step is to stitch the submap meshes together and move them according to the reconstruction at runtime

    # First we load the keyframe pose history
    # We want the pose of each global keyframe and how it evolves over time.

    # segm_idx = len(self.fragments_dict.keys())-1
    # path = os.path.join(os.path.join(output, 'ckpts'), '{:05d}_before_pgo.tar'.format(segm_idx))

    ckptsdir = f'{output}/ckpts'
    ckpt_list = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'pgo.tar' in f]
    
    keyframe_poses = dict()
    for i in range(0, len(ckpt_list), 2):
        nbr_segments = int(ckpt_list[i].split("/")[-1].split("_")[0]) + 1 # the naming is 1 less than the actual nbr
        before_pgo = torch.load(ckpt_list[i+1], map_location="cpu")['estimate_c2w_list']
        after_pgo = torch.load(ckpt_list[i], map_location="cpu")['estimate_c2w_list']
        for k in range(nbr_segments):
            keyframe_idx = submap_indices[k]
            if "segment_" + str(k) not in keyframe_poses:
                keyframe_poses["segment_" + str(k)] = [before_pgo[keyframe_idx].numpy()]
            else:
                keyframe_poses["segment_" + str(k)].append(before_pgo[keyframe_idx].numpy())

            keyframe_poses["segment_" + str(k)].append(after_pgo[keyframe_idx].numpy())

    # Given the poses, we can now start making the visualization with open3d

    estimate_c2w_list = estimate_c2w_list.numpy()
    gt_c2w_list = gt_c2w_list.numpy()

    # init_pose = copy.deepcopy(o3d.io.read_pinhole_camera_parameters('/home/esandstroem/scratch-second/esandstroem/point-slam-loop/src/tools/camera_pos.json').extrinsic)
    init_pose = copy.deepcopy(o3d.io.read_pinhole_camera_parameters('/home/notchla/Downloads/camera_pos.json').extrinsic)

    frontend = SLAMFrontend(output, init_pose=init_pose, cam_scale=0.3,
                            save_rendering=True, near=0, gt_c2w_list=gt_c2w_list).start()
    meshfile = f'{output}/mesh/scene0054_00_pred_mesh.ply'
    # frontend.update_mesh(meshfile)


    # get list of "before_pgo.tar" files with which we will update the global trajectory vis with
    ckpt_list_cam = [os.path.join(ckptsdir, f)
                for f in sorted(os.listdir(ckptsdir)) if 'before_pgo.tar' in f]
    
    # loop over the frames for the first cam_list
    cam_list = ckpt_list_cam[0]
    cam_tensor_list = torch.load(cam_list, map_location="cpu")['estimate_c2w_list']
    last_pgo_index = None
    submap_index_tracker = 0
    for i, cam in enumerate(cam_tensor_list):
        time.sleep(0.06)
        if torch.abs(cam).sum() == 0:
            # we have reached the end of this "before pgo" camera file.
            # we now want to update the camera trajectory to the last index
            frontend.update_cam_trajectory(cam_tensor_list , i, gt=False)
            frontend.update_cam_trajectory(cam_tensor_list , i, gt=True)
            last_pgo_index = i
            break
        # if we don't do deepcopy here, there whole cam_tensor_list and gt_c2w_list will be multiplied by -1 in the third column
        frontend.update_pose(1, copy.deepcopy(cam), gt=False)
        frontend.update_pose(1, copy.deepcopy(gt_c2w_list[i]), gt=True)
        # the visualizer might get stuck if update every frame
        # with a long sequence (10000+ frames)
        if i > submap_indices[submap_index_tracker]:
            mesh_idx_name = submap_indices[submap_index_tracker + 1] - 1
            # load new mesh to scene
            meshfile = f'{output}/mesh/mid_mesh/submap_{submap_index_tracker}/frame_{mesh_idx_name}_mesh.ply'
            # frontend.update_mesh(meshfile)
            default_pose = copy.deepcopy(estimate_c2w_list[submap_indices[submap_index_tracker]])
            default_pose[:3, 2] *= -1
            default_pose[:3, 1] *= -1
            default_pose = np.linalg.inv(default_pose)

            first_pose = copy.deepcopy(cam_tensor_list[submap_indices[submap_index_tracker]])
            first_pose[:3, 2] *= -1
            first_pose[:3, 1] *= -1
            first_pose = np.linalg.inv(first_pose) # w2c

            frontend.add_mesh(meshfile, first_pose, default_pose)
            submap_index_tracker += 1

            
        if i % 10 == 0: # c2w_list, i, is_gt
            frontend.update_cam_trajectory(cam_tensor_list , i, gt=False)
            frontend.update_cam_trajectory(cam_tensor_list , i, gt=True)


    # loop over ckpt_list_cam .tar files
    for cam_list in ckpt_list_cam[1:]:
        # load the camera list:
        cam_tensor_list = torch.load(cam_list, map_location="cpu")['estimate_c2w_list']
        # compute the start idx of the cam_tensor_list
        # snap the old camera centers to their new location after PGO
        frontend.update_cam_trajectory(cam_tensor_list, last_pgo_index, gt=False)
        frontend.update_cam_trajectory(cam_tensor_list, last_pgo_index, gt=True)
        # update existing submap meshes according to the PGO
        # just feed a dict of the absolute poses extracted from the "cam_tensor_list" up to the last_pgo_index
        # and the keys are the same keys we use for the meshes in the draw_trajectory function in the viz.py file
        transformations = {}
        for k in range(submap_index_tracker):
            pose = copy.deepcopy(cam_tensor_list[submap_indices[k]])
            pose[:3, 2] *= -1
            pose[:3, 1] *= -1
            pose = np.linalg.inv(pose) # w2c
            transformations[f'submap_{k}'] = pose
        frontend.update_mesh_poses(transformations)

        cam_tensor_list_loop = cam_tensor_list[last_pgo_index:]
        for i, cam in enumerate(cam_tensor_list_loop):

            time.sleep(0.06)
            if torch.abs(cam).sum() == 0:
                last_pgo_index += i
                frontend.update_cam_trajectory(cam_tensor_list , last_pgo_index, gt=False)
                frontend.update_cam_trajectory(cam_tensor_list , last_pgo_index, gt=True)
                break

            frontend.update_pose(1, copy.deepcopy(cam), gt=False)
            frontend.update_pose(1, copy.deepcopy(gt_c2w_list[i+last_pgo_index]), gt=True)
            # the visualizer might get stucked if update every frame
            # with a long sequence (10000+ frames)
            if (i + last_pgo_index) > submap_indices[submap_index_tracker]:
                mesh_idx_name = submap_indices[submap_index_tracker + 1] - 1
                # load new mesh to scene
                meshfile = f'{output}/mesh/mid_mesh/submap_{submap_index_tracker}/frame_{mesh_idx_name}_mesh.ply'
                # frontend.update_mesh(meshfile)
                # the default pose is the pose with which we integrated the submaps with using tsdf fusion
                # the default pose is the final pose of the keyframe associated with the submap
                default_pose = copy.deepcopy(estimate_c2w_list[submap_indices[submap_index_tracker]])
                default_pose[:3, 2] *= -1
                default_pose[:3, 1] *= -1
                default_pose = np.linalg.inv(default_pose)

                first_pose = copy.deepcopy(cam_tensor_list[submap_indices[submap_index_tracker]])
                first_pose[:3, 2] *= -1
                first_pose[:3, 1] *= -1
                first_pose = np.linalg.inv(first_pose) # w2c

                frontend.add_mesh(meshfile, first_pose, default_pose)
                submap_index_tracker += 1
                    
            if i % 10 == 0: # c2w_list, i, is_gt
                frontend.update_cam_trajectory(cam_tensor_list, i+last_pgo_index, gt=False)
                frontend.update_cam_trajectory(cam_tensor_list, i+last_pgo_index, gt=True)
            
            if (i+last_pgo_index) % 500 == 0:
                print(i+last_pgo_index)



    # then update the poses using the final "estimate_c2w_list". First jump to the correct idx
    # and then loop over the remaining frames

    frontend.update_cam_trajectory(estimate_c2w_list, last_pgo_index, gt=False)
    frontend.update_cam_trajectory(estimate_c2w_list, last_pgo_index, gt=True)
    estimate_c2w_list_loop = estimate_c2w_list[last_pgo_index:]

    # we know that no more PGOs are taking place so we need not update the geometry here.
    
    for i, cam in enumerate(estimate_c2w_list_loop):
        time.sleep(0.06)
        frontend.update_pose(1, copy.deepcopy(cam), gt=False)
        frontend.update_pose(1, copy.deepcopy(gt_c2w_list[i+last_pgo_index]), gt=True)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if submap_index_tracker >= len(submap_indices) - 1:
            # we want to load the last submap
            mesh_files_list = os.listdir(f'{output}/mesh/mid_mesh/submap_{submap_index_tracker}')
            # take the last one which contains the highest frame count
            mesh_file = mesh_files_list[-1]
            # load new mesh to scene
            meshfile = f'{output}/mesh/mid_mesh/submap_{submap_index_tracker}/{mesh_file}'
            # frontend.update_mesh(meshfile)
            # the default pose is the pose with which we integrated the submaps with using tsdf fusion
            # the default pose is the final pose of the keyframe associated with the submap
            default_pose = copy.deepcopy(estimate_c2w_list[submap_indices[submap_index_tracker]])
            default_pose[:3, 2] *= -1
            default_pose[:3, 1] *= -1
            default_pose = np.linalg.inv(default_pose)

            first_pose = copy.deepcopy(cam_tensor_list[submap_indices[submap_index_tracker]])
            first_pose[:3, 2] *= -1
            first_pose[:3, 1] *= -1
            first_pose = np.linalg.inv(first_pose) # w2c

            frontend.add_mesh(meshfile, first_pose, default_pose)
            # frontend.add_mesh(meshfile)
        elif (i + last_pgo_index) > submap_indices[submap_index_tracker]:
            mesh_idx_name = submap_indices[submap_index_tracker + 1] - 1
            # load new mesh to scene
            meshfile = f'{output}/mesh/mid_mesh/submap_{submap_index_tracker}/frame_{mesh_idx_name}_mesh.ply'
            # frontend.update_mesh(meshfile)
            # the default pose is the pose with which we integrated the submaps with using tsdf fusion
            # the default pose is the final pose of the keyframe associated with the submap
            default_pose = copy.deepcopy(estimate_c2w_list[submap_indices[submap_index_tracker]])
            default_pose[:3, 2] *= -1
            default_pose[:3, 1] *= -1
            default_pose = np.linalg.inv(default_pose)

            first_pose = copy.deepcopy(cam_tensor_list[submap_indices[submap_index_tracker]])
            first_pose[:3, 2] *= -1
            first_pose[:3, 1] *= -1
            first_pose = np.linalg.inv(first_pose) # w2c

            frontend.add_mesh(meshfile, first_pose, default_pose)
            # frontend.add_mesh(meshfile)
            submap_index_tracker += 1
            
        if i % 10 == 0: # c2w_list, i, is_gt
            frontend.update_cam_trajectory(cam_tensor_list, i+last_pgo_index, gt=False)
            frontend.update_cam_trajectory(cam_tensor_list, i+last_pgo_index, gt=True)

    time.sleep(1)
    os.system(
        f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")

if __name__ == "__main__":
    main()

