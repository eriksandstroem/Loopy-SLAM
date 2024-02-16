import torch
import numpy as np
import numpy.ma as ma
import warnings
import random
import open3d as o3d
import cv2
import pydbow3 as bow
import os
import matplotlib.pyplot as plt

import faiss
import faiss.contrib.torch_utils
from src.common import setup_seed, clone_kf_dict, compute_cos_rel_rot, compute_rel_trans, pairwise_registration, matching_result, update_posegraph_for_scene, register_point_cloud_pair
from typing import Literal, Union, Optional
import open3d as o3d
import pickle
from multiprocessing import Pool
from src.utils.datasets import get_dataset
from src.common import get_tensor_from_camera

from joblib import Parallel, delayed
import multiprocessing
import time
import psutil
from operator import itemgetter


class NeuralPointCloud(object):
    def __init__(self, cfg, slam, args):
        self.cfg = cfg
        self.c_dim = cfg['model']['c_dim']
        self.device = cfg['mapping']['device']
        self.cuda_id = 0
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.nn_num = cfg['pointcloud']['nn_num']

        self.nlist = cfg['pointcloud']['nlist']
        self.radius_add = cfg['pointcloud']['radius_add']
        self.radius_min = cfg['pointcloud']['radius_min']
        self.radius_query = cfg['pointcloud']['radius_query']
        self.radius_mesh = cfg['pointcloud']['radius_mesh']
        self.fix_interval_when_add_along_ray = cfg['pointcloud']['fix_interval_when_add_along_ray']

        self.N_surface = cfg['rendering']['N_surface']
        self.N_add = cfg['pointcloud']['N_add']
        self.near_end_surface = cfg['pointcloud']['near_end_surface']
        self.far_end_surface = cfg['pointcloud']['far_end_surface']
        self.fixed_segment_size = cfg["mapping"]["fixed_segment_size"]
        self.segment_strategy = cfg["mapping"]["segment_strategy"]
        self.segment_rel_trans = cfg["mapping"]["segment_rel_trans"]
        self.segment_rot_cos = cfg["mapping"]["segment_rot_cos"]
        self.min_dist = cfg["tracking"]["min_dist"]
        self.kval = cfg["tracking"]["kval"]

        self._cloud = []
        self._input_pos = []     # to save locations of the rgbd input
        self._input_rgb = []
        self._input_normal = []
        self._input_normal_cartesian = []
        self._pts_num = 0
        self.keyframe_dict = []

        # {segment_0 : {keyframe : Tensor, points : List, points_color : List, gt_color : Tensor, npc : List, geo_feats : Tensor, col_feats : Tensor}, ...}
        self.fragments_dict = None

        self.resource = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.resource,
                                            self.cuda_id,
                                            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
        self.index.nprobe = cfg['pointcloud']['nprobe']
        self.nprobe = cfg['pointcloud']['nprobe']
        self.end_geo_feats = None
        self.end_col_feats = None

        self.orb = cv2.ORB_create()
        self.voc = bow.Vocabulary()
        self.voc.load(cfg["orbvoc"])
        self.db = bow.Database()
        self.db.setVocabulary(self.voc)

        # for correction
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.output = slam.output
        self.new_segment = False
        self.max_correspondence_distance_coarse = 0.3
        self.max_correspondence_distance_fine = 0.03
        self.frame_reader = get_dataset(cfg, args, "cpu")
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.use_old_segments_only = True

        self.odometry_errors_after = []
        self.odometry_errors_before = []
        self.loop_errors_after = []
        self.loop_errors_before = []
        self.registration_errors = []

        self.registration_method = "robust_icp"
        self.filter = cfg["tracking"]["filter"] # read from configs
        self.orb_filter = cfg["tracking"]["dbow_filter"]
        self.dbow_scores = []
        self.registration_times = []
        
        self.mult_dbow = cfg["tracking"]["mult_dbow"]
        self.prune_pgo = cfg["tracking"]["prune_pgo"]
        self.lc_pref = cfg["tracking"]["lc_pref"]
        self.std_threshold = cfg["tracking"]["std_threshold"]
        self.newnew_trans_mag_filter = cfg["tracking"]["newnew_trans_mag_filter"]
        self.norm_trans_mag_thresh = cfg["tracking"]["norm_trans_mag_thresh"]
        self.fitness_thresh = cfg["tracking"]["fitness_thresh"]
        self.gt_thresh = cfg["tracking"]["gt_thresh"]
        self.gt_filtering = cfg["tracking"]["gt_filtering"]
        self.trans_mag_percentile = cfg["tracking"]["trans_mag_percentile"]
        self.gt_constraints = cfg["tracking"]["gt_constraints"]
        self.iter_std_thresh = cfg["tracking"]["iter_std_thresh"]
        self.global_iter = cfg["tracking"]["global_iter"]
        self.conf = cfg["tracking"]["global_reg_conf"]
        self.old_trans_mag_filter = cfg["tracking"]["old_trans_mag_filter"]
        self.distance_thresholding = cfg["tracking"]["distance_thresholding"]
        
        self.pgo_times = []


        setup_seed(cfg["setup_seed"])

    def extract_orb_features(self, img, index):
        img = np.float32(img.cpu().numpy()*255)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(np.uint8(gray_img), None)
        img2 = cv2.drawKeypoints(
            np.uint8(gray_img), kp, None, color=(0, 255, 0), flags=0)
        cv2.imwrite(os.path.join(f"{self.output}/orb",
                    "segment"+str(index))+".png", img2)
        return des

    def add_orb_features(self, features):
        self.db.add(features)

    def query_bow(self, features, n_matches):
        results = self.db.query(features, n_matches)
        print([(result.Id, result.Score) for result in results])
        return results

    def apply_correction(self, pose_graph):
        fragments_names = list(self.fragments_dict.keys())

        # save camera list
        segm_idx = len(self.fragments_dict.keys())-1
        path = os.path.join(os.path.join(self.output, 'ckpts'), '{:05d}_before_pgo.tar'.format(segm_idx))
        torch.save({'estimate_c2w_list': self.estimate_c2w_list}, path, _use_new_zipfile_serialization=False)
        with torch.no_grad():
            for i, segment in enumerate(self.fragments_dict.keys()):
                # if(i < len(self.fragments_dict.keys())-1):
                #     continue
                print(i, segment)
                fragment = self.fragments_dict[segment]
                start_idx = fragment["start_idx"]
                end_idx = self.fragments_dict[fragments_names[i+1]]["start_idx"] if i+1 < len(
                    fragments_names) else start_idx+1

                print("start/end idx", start_idx, end_idx)
                points = fragment["npc"]

                if i == len(fragments_names)-1 and self.use_old_segments_only:
                    print("last segment")
                    transformation = torch.from_numpy(
                        pose_graph.nodes[i-1].pose.copy()).float().cuda()
                else:
                    transformation = torch.from_numpy(
                        pose_graph.nodes[i].pose.copy()).float().cuda()

                print("transformation", transformation)

                ones = torch.ones(len(points), 1).cuda()
                homo_points = torch.cat(
                    [torch.tensor(points).cuda(), ones], dim=1).cuda().float()
                homo_points = homo_points@torch.t(transformation)
                pts = homo_points[:, :3]
                z = homo_points[:, -1:]
                pts = pts / z

                fragment["npc"] = pts.tolist()

                points_sensor = fragment["points"]

                ones = torch.ones(len(points_sensor), 1).cuda()
                homo_points = torch.cat(
                    [torch.tensor(points_sensor).cuda(), ones], dim=1).cuda().float()
                homo_points = homo_points@torch.t(transformation)
                pts_sensor = homo_points[:, :3]
                z = homo_points[:, -1:]
                pts_sensor = pts_sensor / z

                fragment["points"] = pts_sensor.tolist()

                cameras = self.estimate_c2w_list[start_idx:end_idx]

                # print(cameras, cameras.shape)
                # print(self.camera_before_correction)

                cameras_tensor = cameras.cuda()

                new_cameras_tensor = transformation@cameras_tensor

                new_cameras_tensor = new_cameras_tensor.cpu()

                new_cameras_tensor[:, 3] = torch.tensor([0, 0, 0, 1.0])

                # print(new_cameras_tensor, cameras)

                self.estimate_c2w_list[start_idx:end_idx] = new_cameras_tensor

                fragment["keyframe"] = self.estimate_c2w_list[start_idx].clone().cpu()

                gt_camera = fragment["gt_camera"]
                keyframe = fragment["keyframe"]

                t_error = (gt_camera[:3, 3:] - keyframe[:3, 3:]).norm(2)

                try:
                    with open(f'{self.output}/segments_error/segment_{i}', "rb") as f:
                        errors = pickle.load(f)
                except:
                    errors = []
                with open(f'{self.output}/segments_error/segment_{i}', "wb") as f:
                    errors.append(t_error.item())
                    pickle.dump(errors, f)

            # save camera list
            path = os.path.join(os.path.join(self.output, 'ckpts'), '{:05d}_after_pgo.tar'.format(segm_idx))
            torch.save({'estimate_c2w_list': self.estimate_c2w_list}, path, _use_new_zipfile_serialization=False)
            return pts, self.estimate_c2w_list[start_idx]

    def compute_odometry_errors(self, after_correction=False, iter=None):
        segment_list = list(self.fragments_dict.keys())
        n_segments = len(segment_list)
        errors = []
        for i, segment_k in enumerate(segment_list):
            if (i + 1) < n_segments:
                current_s = self.fragments_dict[segment_k]
                next_s = self.fragments_dict[segment_list[i+1]]

                gt_current = current_s["gt_camera"].detach().clone()
                gt_next = next_s["gt_camera"].detach().clone()

                delta_gt = gt_next@gt_current.inverse()

                e_current = current_s["keyframe"].detach().clone()
                e_next = next_s["keyframe"].detach().clone()

                delta_e = e_next@e_current.inverse()

                delta_gt_tensor = get_tensor_from_camera(delta_gt)
                delta_e_tensor = get_tensor_from_camera(delta_e)

                loss_camera_tensor = torch.abs(delta_gt_tensor-delta_e_tensor)
                camera_tensor_error = loss_camera_tensor.mean().item()
                camera_quad_error = loss_camera_tensor[:4].mean().item()
                camera_pos_error = loss_camera_tensor[-3:].mean().item()
                error_dict = {"tensor": camera_tensor_error, "quad": camera_quad_error,
                              "pos": camera_pos_error, "source": i, "target": i+1, "odometry": True}
                errors.append(error_dict)
        # import pickle

        # before_or_after = "after" if after_correction else "before"
        # pickle.dump(errors, open(os.path.join(
        #     f'{self.output}/segments_error/', f"odometry_errors_{before_or_after}_iter{iter}"), "wb"))
        if after_correction:
            self.odometry_errors_after.append(errors)
        else:
            self.odometry_errors_before.append(errors)
        # self.create_plot(errors, after_correction, iter)

    def compute_loop_errors(self, pairs, after_correction=False, iter=None):
        segment_list = list(self.fragments_dict.keys())
        errors = []
        for pair in pairs:
            if pair.s+1 != pair.t:
                source_s = self.fragments_dict[segment_list[pair.s]]
                target_s = self.fragments_dict[segment_list[pair.t]]

                gt_current = source_s["gt_camera"].detach().clone()
                gt_next = target_s["gt_camera"].detach().clone()

                delta_gt = gt_next@gt_current.inverse()

                e_current = source_s["keyframe"].detach().clone()
                e_next = target_s["keyframe"].detach().clone()

                delta_e = e_next@e_current.inverse()

                delta_gt_tensor = get_tensor_from_camera(delta_gt)
                delta_e_tensor = get_tensor_from_camera(delta_e)

                loss_camera_tensor = torch.abs(delta_gt_tensor-delta_e_tensor)
                camera_tensor_error = loss_camera_tensor.mean().item()
                camera_quad_error = loss_camera_tensor[:4].mean().item()
                camera_pos_error = loss_camera_tensor[-3:].mean().item()
                error_dict = {"tensor": camera_tensor_error, "quad": camera_quad_error,
                              "pos": camera_pos_error, "source": pair.s, "target": pair.t, "odometry": False}
                errors.append(error_dict)
        import pickle

        before_or_after = "after" if after_correction else "before"
        pickle.dump(errors, open(os.path.join(
            f'{self.output}/segments_error/', f"loop_errors_{before_or_after}_iter{iter}"), "wb"))
        if after_correction:
            self.loop_errors_after.append(errors)
        else:
            self.loop_errors_before.append(errors)

    def compute_registration_errors(self, pairs):
        segment_list = list(self.fragments_dict.keys())
        errors = []
        for pair in pairs:
            if pair.s+1 != pair.t:
                source_s = self.fragments_dict[segment_list[pair.s]]
                target_s = self.fragments_dict[segment_list[pair.t]]

                gt_current = source_s["gt_camera"].detach().clone()
                gt_next = target_s["gt_camera"].detach().clone()

                delta_gt = gt_next@gt_current.inverse()

                e_current = source_s["keyframe"].detach().clone()

                transformation = torch.from_numpy(pair.transformation).float()

                e_current = transformation@e_current
                e_next = target_s["keyframe"].detach().clone()

                delta_e = e_next@e_current.inverse()

                delta_gt_tensor = get_tensor_from_camera(delta_gt)
                delta_e_tensor = get_tensor_from_camera(delta_e)

                loss_camera_tensor = torch.abs(delta_gt_tensor-delta_e_tensor)
                camera_tensor_error = loss_camera_tensor.mean().item()
                camera_quad_error = loss_camera_tensor[:4].mean().item()
                camera_pos_error = loss_camera_tensor[-3:].mean().item()
                error_dict = {"tensor": camera_tensor_error, "quad": camera_quad_error,
                              "pos": camera_pos_error, "source": pair.s, "target": pair.t, "odometry": False}
                errors.append(error_dict)
        self.registration_errors.append(errors)

    def plot_deltas_old(self, bool_filtering, transformation_errs, fitnesses, inlier_rmses, trans_mags, trans_gt_mags, distances, filename):
        # Create a figure with subplots based on the number of input lists
        fig, axs = plt.subplots(3, 2, figsize=(8, 9))

        # Plot first list
        axs[0, 0].plot(range(len(transformation_errs)),
                       transformation_errs, marker='o', markersize=8, linestyle='-')
        axs[0, 0].set_xlabel('Loop Edge')
        axs[0, 0].set_ylabel('Transformation Error')

        # Plot the second list
        axs[1, 0].plot(range(len(fitnesses)), fitnesses,
                       marker='o', markersize=8, linestyle='-')
        axs[1, 0].set_xlabel('Loop Edge')
        axs[1, 0].set_ylabel('Fitness')

        # Plot the third list
        axs[2, 0].plot(range(len(inlier_rmses)), inlier_rmses,
                       marker='o', markersize=8, linestyle='-')
        axs[2, 0].set_xlabel('Loop Edge')
        axs[2, 0].set_ylabel('Inlier RMSE')

        # Plot the fourth list
        axs[0, 1].plot(range(len(trans_gt_mags)), trans_gt_mags,
                       marker='o', markersize=8, linestyle='-')
        axs[0, 1].set_xlabel('Loop Edge')
        axs[0, 1].set_ylabel('GT Transformation Magnitude')

        # Plot the fifth list
        axs[1, 1].plot(range(len(trans_mags)), trans_mags,
                       marker='o', markersize=8, linestyle='-')
        axs[1, 1].set_xlabel('Loop Edge')
        axs[1, 1].set_ylabel('Transformation Magnitude')

        # Plot the sixth list
        axs[2, 1].plot(range(len(distances)), distances,
                       marker='o', markersize=8, linestyle='-')
        axs[2, 1].set_xlabel('Loop Edge')
        axs[2, 1].set_ylabel('Loop Distance')

        plt.tight_layout()

        # Save the figure as a PNG file
        plt.savefig(filename)

    def plot_deltas(self, bool_filtering, transformation_errs, fitnesses, inlier_rmses, trans_mags, trans_gt_mags, distances, filename):
        # Create a figure with subplots based on the number of input lists
        fig, axs = plt.subplots(3, 2, figsize=(8, 9))

        # Plot first list with lines and markers
        axs[0, 0].plot(range(len(transformation_errs)),
                        transformation_errs, marker='o', markersize=8, linestyle='-', color='blue')
        for i, b in enumerate(bool_filtering):
            if b:
                axs[0, 0].plot(i, transformation_errs[i], marker='o', markersize=8, linestyle='-', color='green')

        axs[0, 0].set_xlabel('Loop Edge')
        axs[0, 0].set_ylabel('Transformation Error')

        # Plot the second list with lines and markers
        axs[1, 0].plot(range(len(fitnesses)), fitnesses,
                        marker='o', markersize=8, linestyle='-', color='blue')
        for i, b in enumerate(bool_filtering):
            if b:
                axs[1, 0].plot(i, fitnesses[i], marker='o', markersize=8, linestyle='-', color='green')

        axs[1, 0].set_xlabel('Loop Edge')
        axs[1, 0].set_ylabel('Fitness')

        # Plot the third list with lines and markers
        axs[2, 0].plot(range(len(inlier_rmses)), inlier_rmses,
                        marker='o', markersize=8, linestyle='-', color='blue')
        for i, b in enumerate(bool_filtering):
            if b:
                axs[2, 0].plot(i, inlier_rmses[i], marker='o', markersize=8, linestyle='-', color='green')

        axs[2, 0].set_xlabel('Loop Edge')
        axs[2, 0].set_ylabel('Inlier RMSE')

        # Plot the fourth list with lines and markers
        axs[0, 1].plot(range(len(trans_gt_mags)), trans_gt_mags,
                        marker='o', markersize=8, linestyle='-', color='blue')
        for i, b in enumerate(bool_filtering):
            if b:
                axs[0, 1].plot(i, trans_gt_mags[i], marker='o', markersize=8, linestyle='-', color='green')

        axs[0, 1].set_xlabel('Loop Edge')
        axs[0, 1].set_ylabel('GT Transformation Magnitude')

        # Plot the fifth list with lines and markers
        axs[1, 1].plot(range(len(trans_mags)), trans_mags,
                        marker='o', markersize=8, linestyle='-', color='blue')
        for i, b in enumerate(bool_filtering):
            if b:
                axs[1, 1].plot(i, trans_mags[i], marker='o', markersize=8, linestyle='-', color='green')

        axs[1, 1].set_xlabel('Loop Edge')
        axs[1, 1].set_ylabel('Transformation Magnitude')

        # Plot the sixth list with lines and markers
        axs[2, 1].plot(range(len(distances)), distances,
                        marker='o', markersize=8, linestyle='-', color='blue')
        for i, b in enumerate(bool_filtering):
            if b:
                axs[2, 1].plot(i, distances[i], marker='o', markersize=8, linestyle='-', color='green')

        axs[2, 1].set_xlabel('Loop Edge')
        axs[2, 1].set_ylabel('Loop Distance')

        plt.tight_layout()

        # Save the figure as a PNG file
        plt.savefig(filename)

    def save_plots(self, iters, values, title, loop=False):
        x = np.arange(len(iters))
        width = 0.25
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in values.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_ylabel('Error')
        ax.set_title('Loop errors' if loop else "Odometry errors")
        ax.set_xticks(x + width, iters)
        ax.legend(loc='upper left', ncols=3 if loop else 2)

        plt.savefig(os.path.join(f'{self.output}/segments_error/', title))

    def create_plots_odometry(self, n_segments):
        iters = list(range(0, len(self.odometry_errors_after)))
        values_before = {
            "Tensor": [],
            "Quad": [],
            "Pos": []
        }
        values_after = {
            "Tensor": [],
            "Quad": [],
            "Pos": []
        }

        # after mean error
        for iter in self.odometry_errors_after:
            mean_tensor_error_iter = np.mean([e["tensor"] for e in iter])
            mean_quad_error_iter = np.mean([e["quad"] for e in iter])
            mean_pos_error_iter = np.mean([e["pos"] for e in iter])
            values_after["Tensor"].append(mean_tensor_error_iter)
            values_after["Quad"].append(mean_quad_error_iter)
            values_after["Pos"].append(mean_pos_error_iter)

        # before error mean
        for iter in self.odometry_errors_before:
            mean_tensor_error_iter = np.mean([e["tensor"] for e in iter])
            mean_quad_error_iter = np.mean([e["quad"] for e in iter])
            mean_pos_error_iter = np.mean([e["pos"] for e in iter])
            values_before["Tensor"].append(mean_tensor_error_iter)
            values_before["Quad"].append(mean_quad_error_iter)
            values_before["Pos"].append(mean_pos_error_iter)

        tensor_values = {
            "Tensor_b": values_before["Tensor"],
            "Tensor_a": values_after["Tensor"]
        }

        pos_values = {
            "Pos_b": values_before["Pos"],
            "Pos_a": values_after["Pos"],
        }

        quad_values = {
            "Quad_b": values_before["Quad"],
            "Quad_a": values_after["Quad"]
        }

        self.save_plots(iters, tensor_values,
                        "odometry_tensor" + str(n_segments) + ".png")
        self.save_plots(iters, pos_values, "odometry_pos" +
                        str(n_segments) + ".png")
        self.save_plots(iters, quad_values, "odometry_quad" +
                        str(n_segments) + ".png")

    def create_plots_loop(self, n_segments):
        iters = list(range(0, len(self.loop_errors_after)))
        values_before = {
            "Tensor": [],
            "Quad": [],
            "Pos": []
        }
        values_after = {
            "Tensor": [],
            "Quad": [],
            "Pos": []
        }

        values_registration = {
            "Tensor": [],
            "Quad": [],
            "Pos": []
        }
        # after mean error
        for iter in self.loop_errors_after:
            mean_tensor_error_iter = np.mean([e["tensor"] for e in iter])
            mean_quad_error_iter = np.mean([e["quad"] for e in iter])
            mean_pos_error_iter = np.mean([e["pos"] for e in iter])
            values_after["Tensor"].append(mean_tensor_error_iter)
            values_after["Quad"].append(mean_quad_error_iter)
            values_after["Pos"].append(mean_pos_error_iter)

        # before error mean
        for iter in self.loop_errors_before:
            mean_tensor_error_iter = np.mean([e["tensor"] for e in iter])
            mean_quad_error_iter = np.mean([e["quad"] for e in iter])
            mean_pos_error_iter = np.mean([e["pos"] for e in iter])
            values_before["Tensor"].append(mean_tensor_error_iter)
            values_before["Quad"].append(mean_quad_error_iter)
            values_before["Pos"].append(mean_pos_error_iter)

        # registration error
        for iter in self.registration_errors:
            mean_tensor_error_iter = np.mean([e["tensor"] for e in iter])
            mean_quad_error_iter = np.mean([e["quad"] for e in iter])
            mean_pos_error_iter = np.mean([e["pos"] for e in iter])
            values_registration["Tensor"].append(mean_tensor_error_iter)
            values_registration["Quad"].append(mean_quad_error_iter)
            values_registration["Pos"].append(mean_pos_error_iter)

        tensor_values = {
            "Tensor_b": values_before["Tensor"],
            "Tensor_a": values_after["Tensor"],
            "Tensor_r": values_registration["Tensor"]
        }

        pos_values = {
            "Pos_b": values_before["Pos"],
            "Pos_a": values_after["Pos"],
            "Pos_r": values_registration["Pos"]
        }

        quad_values = {
            "Quad_b": values_before["Quad"],
            "Quad_a": values_after["Quad"],
            "Quad_r": values_registration["Quad"]
        }

        self.save_plots(iters, tensor_values, "loop_tensor" +
                        str(n_segments) + ".png", loop=True)
        self.save_plots(iters, pos_values, "loop_pos" +
                        str(n_segments) + ".png", loop=True)
        self.save_plots(iters, quad_values, "loop_quad" +
                        str(n_segments) + ".png", loop=True)

    def compute_correction(self):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        matching_results = []
        if self.use_old_segments_only:
            segments_list = list(self.fragments_dict.keys())[
                :-1]  # remove last segment
        else:
            segments_list = list(self.fragments_dict.keys())
        n_segments = len(segments_list)
        # check if last segment has any loop edges. If it does not,
        last_segment_loop_edge = False
        # we can guarantee that the graph is identical to how it was in the previous LC
        # therefore, we do not trigger LC.
        for _ in range(n_segments):  # add all the nodes
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
        for s, dbow_score in zip(range(n_segments), self.dbow_scores):
            if s+1 < n_segments:
                matching_results.append(matching_result(s, s+1))
            fragment = self.fragments_dict[segments_list[s]]
            features = self.get_keyframe_orb(fragment["gt_color"])
            dbow_results = self.query_bow(
                features, self.kval)  # read K from the configs
            if self.orb_filter:
                print(f"used dbow score for keyframe-{s} is {dbow_score}")
            if self.use_old_segments_only:
                if self.orb_filter:
                    dbow_results = [x for x in dbow_results if abs(
                        x.Id - s) > self.min_dist and x.Id < n_segments and x.Score > self.mult_dbow*dbow_score]
                else:
                    dbow_results = [x for x in dbow_results if abs(
                        x.Id - s) > self.min_dist and x.Id < n_segments]
            else:
                dbow_results = [
                    x for x in dbow_results if abs(x.Id - s) > self.min_dist]
            ids_dbow = [result.Id for result in dbow_results]
            for t in ids_dbow:
                # if we connect to or from the last segment from any segment
                if t == n_segments - 1 or s == n_segments - 1:
                    # we want to trigger LC.
                    last_segment_loop_edge = True
                matching_results.append(matching_result(s, t))

        if not last_segment_loop_edge:
            return None

        os.environ['OMP_NUM_THREADS'] = '2'
        MAX_THREADS = 8
        current_process = psutil.Process()
        subproc_before = set(
            [p.pid for p in current_process.children(recursive=True)])
        tic = time.perf_counter()
        results = Parallel(n_jobs=MAX_THREADS)(delayed(register_point_cloud_pair)(matching_results[r].s, matching_results[r].t, self.fragments_dict[segments_list[matching_results[r].s]], self.fragments_dict[
            segments_list[matching_results[r].t]], method=self.registration_method if matching_results[r].s+1 != matching_results[r].t else "identity", global_iter=self.global_iter, conf=self.conf) for r in range(len(matching_results)))
        toc = time.perf_counter()
        subproc_after = set(
            [p.pid for p in current_process.children(recursive=True)])
        for subproc in subproc_after - subproc_before:
            print('Killing process with pid {}'.format(subproc))
            psutil.Process(subproc).terminate()
        print(f"elapsed time {toc - tic}")

        for i in range(len(matching_results)):
            matching_results[i].success = results[i]["success"]
            matching_results[i].transformation = results[i]["transformation"]
            matching_results[i].information = results[i]["information"]
            if matching_results[i].s+1 != matching_results[i].t:
                matching_results[i].success = results[i]["success"]
                matching_results[i].fitness = results[i]["fitness"]
                matching_results[i].inlier_rmse = results[i]["inlier_rmse"]
                matching_results[i].transformation_gt_mag = results[i]["transformation_gt_mag"]
                matching_results[i].transformation_mag = results[i]["transformation_mag"]
                matching_results[i].transformation_transl_mag = results[i]["transformation_transl_mag"]
                matching_results[i].transformation_transl_err = results[i]["transformation_transl_err"]
                matching_results[i].transformation_gt = results[i]["transformation_gt"]

                self.registration_times.append(results[i]["registration_time"])

        if self.filter:
            # compute LC distances
            distances = np.array([np.abs(
                matching_result.s - matching_result.t) for matching_result in matching_results])
            # filter away the odometry edges
            
            filter_mask = distances > 1
            distances = distances[distances > 1]
            # compute threshold for top X percent distance
            distance_thresh = np.percentile(distances, 90)
            distance_low_thresh = np.percentile(distances, 10)
            filter_basedon_dist_and_fitness = distances > distance_thresh


            # compute 90th percentile transformation error
            transformation_errs = np.array(
                [torch.abs(get_tensor_from_camera(result.transformation) - get_tensor_from_camera(result.transformation_gt)).mean().item()
                 for result in matching_results])
            # remove the ones from odometry edges
            transformation_errs = transformation_errs[filter_mask]
            nonempty = transformation_errs.size
            if nonempty:
                # compute rotation and translation magnitudes and the corresponding GT values for plotting and filtering
                transformation_transl_errs = np.array(
                    [torch.abs(get_tensor_from_camera(result.transformation)[4:] - get_tensor_from_camera(result.transformation_gt)[4:]).mean().item()
                        for result in matching_results])
                transformation_transl_errs = transformation_transl_errs[filter_mask]

                transformation_gt_transl_mags = np.array(
                    [torch.abs(get_tensor_from_camera(result.transformation_gt)[4:]).mean().item()
                        for result in matching_results])
                transformation_gt_transl_mags = transformation_gt_transl_mags[filter_mask]

                transformation_transl_mags = np.array(
                    [torch.abs(get_tensor_from_camera(result.transformation)[4:]).mean().item()
                        for result in matching_results])
                transformation_transl_mags = transformation_transl_mags[filter_mask]
                max_trans_transl_mag = max(transformation_transl_mags)

                transformation_rot_errs = np.array(
                    [torch.abs(get_tensor_from_camera(result.transformation)[:4] - get_tensor_from_camera(result.transformation_gt)[:4]).mean().item()
                        for result in matching_results])
                transformation_rot_errs = transformation_rot_errs[filter_mask]

                transformation_gt_rot_mags = np.array(
                    [torch.abs(get_tensor_from_camera(result.transformation_gt)[:4]).mean().item()
                        for result in matching_results])
                transformation_gt_rot_mags = transformation_gt_rot_mags[filter_mask]

                transformation_rot_mags = np.array(
                    [torch.abs(get_tensor_from_camera(result.transformation)[:4]).mean().item()
                        for result in matching_results])
                transformation_rot_mags = transformation_rot_mags[filter_mask]

                #threshold = np.percentile(transformation_errs, 15)
                # pair up the transformation error, fitness and inlier rmse for ordering and plotting
                # Pair up elements from all three lists and sort based on the first list

                fitnesses = np.array(
                    [result.fitness for result in matching_results])
                # remove the ones from odometry edges

                fitnesses = fitnesses[filter_mask]
                # based on the fitness score, keep the high distance edges or potentially remove them
                # update the filter mask for this
                fitness_low_thresh = np.percentile(fitnesses, 20)
                filter_basedon_dist_and_fitness = np.logical_and(filter_basedon_dist_and_fitness, fitnesses > fitness_low_thresh)

                # threshold = np.percentile(fitnesses, 40)
                inlier_rmses = np.array(
                    [result.inlier_rmse for result in matching_results])
                # remove the ones from odometry edges
                inlier_rmses = inlier_rmses[filter_mask]

                transformation_mags = np.array(
                    [result.transformation_mag for result in matching_results])
                # remove the ones from odometry edges
                transformation_mags = transformation_mags[filter_mask]
                std = transformation_mags.std()
                
                std_rot = transformation_rot_mags.std()
                max_trans_mag = max(transformation_mags)
                # old
                # threshold = np.percentile(transformation_mags, self.trans_mag_percentile)
                # new
                threshold = np.percentile(transformation_transl_mags, self.trans_mag_percentile)

                # compute iter_std
                # remove the edges that are too distant and have a high enough fitness score
                transformation_transl_mags_f = transformation_transl_mags[~filter_basedon_dist_and_fitness]
                std_trans = transformation_transl_mags_f.std() # compute std after removing the good far away loop edges

                iter_std = transformation_transl_mags_f.std()
                transformation_transl_mags_temp = transformation_transl_mags_f
                percentile = 97.5
                mag_thresh_temp = max(transformation_transl_mags_temp)
                while iter_std > self.iter_std_thresh and percentile > 0:
                    mag_thresh_temp = np.percentile(transformation_transl_mags_temp, percentile)
                    transformation_transl_mags_temp = transformation_transl_mags_temp[transformation_transl_mags_temp < mag_thresh_temp]
                    iter_std = transformation_transl_mags_temp.std()
                    percentile -= 2.5


                transformation_gt_mags = np.array(
                    [result.transformation_gt_mag for result in matching_results])
                # remove the ones from odometry edges
                transformation_gt_mags = transformation_gt_mags[filter_mask]

                sorted_six = sorted(
                    zip(transformation_errs, fitnesses, inlier_rmses, transformation_mags, transformation_gt_mags, distances))
                # Unpack the sorted triplets
                transformation_errs, fitnesses_s, inlier_rmses_s, transformation_mags, transformation_gt_mags, distances_s = zip(
                    *sorted_six)

                # plot the numbers in three subplots in one figure
                # -1 because we trigger lc for the i-1 segments at segment i.
                lc_idx = len(self.fragments_dict) - 2 # (same as n_segments - 1)

                # self.plot_deltas(transformation_errs, fitnesses_s, inlier_rmses_s, transformation_mags, transformation_gt_mags, distances_s,
                #                     os.path.join(f'{self.output}/segments_error/error_plot_loop_closure_' + str(lc_idx) + '_std_' + str(std) + '_full.png'))

                sorted_seven = sorted(
                    zip(transformation_transl_errs, fitnesses, inlier_rmses, transformation_transl_mags, 
                        filter_basedon_dist_and_fitness, transformation_gt_transl_mags, distances))
                # Unpack the sorted triplets
                transformation_transl_errs, fitnesses_s, inlier_rmses_s, transformation_transl_mags, filter_basedon_dist_and_fitness, transformation_gt_transl_mags, distances_s = zip(*sorted_seven)

                # uncomment to plot statistics about the loop registrations
                # self.plot_deltas(filter_basedon_dist_and_fitness, transformation_transl_errs, fitnesses_s, inlier_rmses_s, transformation_transl_mags, transformation_gt_transl_mags, distances_s,
                #                     os.path.join(f'{self.output}/segments_error/error_plot_loop_closure_' + str(lc_idx) + '_std_' + str(std_trans) + '_' + str(mag_thresh_temp) + '_transl.png'))

                # sorted_six = sorted(
                #     zip(transformation_rot_errs, fitnesses, inlier_rmses, transformation_rot_mags, transformation_gt_rot_mags, distances))
                # Unpack the sorted triplets
                # transformation_rot_errs, fitnesses_s, inlier_rmses_s, transformation_rot_mags, transformation_gt_rot_mags, distances_s = zip(
                #     *sorted_six)
                # self.plot_deltas(transformation_rot_errs, fitnesses_s, inlier_rmses_s, transformation_rot_mags, transformation_gt_rot_mags, distances_s,
                #                     os.path.join(f'{self.output}/segments_error/error_plot_loop_closure_' + str(lc_idx) + '_std_' + str(std_rot) + '_rot.png'))

            # check if last segment has valid registration
            valid_last_segment_registration = False
            for r in range(len(matching_results)):
                if matching_results[r].s+1 != matching_results[r].t:  # loop edge
                    if self.newnew_trans_mag_filter:
                        if matching_results[r].success:
                            if std_trans <= self.std_threshold:
                                if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                                    valid_last_segment_registration = True
                                if self.distance_thresholding:
                                    if np.abs(matching_results[r].s - matching_results[r].t) > distance_low_thresh:
                                        pose_graph = update_posegraph_for_scene(
                                            matching_results[r].s, matching_results[r].t,
                                            matching_results[r].transformation,
                                            matching_results[r].information, pose_graph)
                                else:
                                    pose_graph = update_posegraph_for_scene(
                                        matching_results[r].s, matching_results[r].t,
                                        matching_results[r].transformation,
                                        matching_results[r].information, pose_graph)
                            elif np.abs(matching_results[r].s - matching_results[r].t) > distance_thresh and matching_results[r].fitness >= fitness_low_thresh:
                                if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                                    valid_last_segment_registration = True
                                if self.distance_thresholding:
                                    if np.abs(matching_results[r].s - matching_results[r].t) > distance_low_thresh:
                                        pose_graph = update_posegraph_for_scene(
                                            matching_results[r].s, matching_results[r].t,
                                            matching_results[r].transformation,
                                            matching_results[r].information, pose_graph)
                                else:
                                    pose_graph = update_posegraph_for_scene(
                                        matching_results[r].s, matching_results[r].t,
                                        matching_results[r].transformation,
                                        matching_results[r].information, pose_graph)
                            else:
                                if matching_results[r].transformation_transl_mag < mag_thresh_temp and matching_results[r].fitness >= self.fitness_thresh:
                                    if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                                        valid_last_segment_registration = True
                                    if self.distance_thresholding:
                                        if np.abs(matching_results[r].s - matching_results[r].t) > distance_low_thresh:
                                            pose_graph = update_posegraph_for_scene(
                                                matching_results[r].s, matching_results[r].t,
                                                matching_results[r].transformation,
                                                matching_results[r].information, pose_graph)
                                    else:
                                        pose_graph = update_posegraph_for_scene(
                                            matching_results[r].s, matching_results[r].t,
                                            matching_results[r].transformation,
                                            matching_results[r].information, pose_graph)
                    elif self.old_trans_mag_filter:
                        if matching_results[r].success and std_trans <= self.std_threshold:
                            if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                                valid_last_segment_registration = True
                            pose_graph = update_posegraph_for_scene(
                                matching_results[r].s, matching_results[r].t,
                                matching_results[r].transformation,
                                matching_results[r].information, pose_graph)
                        elif matching_results[r].success and matching_results[r].transformation_transl_mag < mag_thresh_temp and matching_results[r].fitness >= self.fitness_thresh:
                            if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                                valid_last_segment_registration = True
                            pose_graph = update_posegraph_for_scene(
                                matching_results[r].s, matching_results[r].t,
                                matching_results[r].transformation,
                                matching_results[r].information, pose_graph)

                    elif self.gt_filtering and matching_results[r].transformation_transl_err <= self.gt_thresh: 
                        if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                            valid_last_segment_registration = True
                        pose_graph = update_posegraph_for_scene(
                            matching_results[r].s, matching_results[r].t,
                            matching_results[r].transformation,
                            matching_results[r].information, pose_graph)
                    else:
                        # if matching_results[r].success and matching_results[r].transformation_mag <= threshold: # old
                        if matching_results[r].success and matching_results[r].transformation_transl_mag <= threshold:
                            if matching_results[r].s == n_segments - 1 or matching_results[r].t == n_segments - 1:
                                valid_last_segment_registration = True
                            pose_graph = update_posegraph_for_scene(
                                matching_results[r].s, matching_results[r].t,
                                matching_results[r].transformation,
                                matching_results[r].information, pose_graph)

                # odometry
                elif matching_results[r].success:
                    pose_graph = update_posegraph_for_scene(
                        matching_results[r].s, matching_results[r].t,
                        matching_results[r].transformation,
                        matching_results[r].information, pose_graph)

        else:
            valid_last_segment_registration = True
            for r in range(len(matching_results)):
                if matching_results[r].success:
                    pose_graph = update_posegraph_for_scene(
                        matching_results[r].s, matching_results[r].t,
                        matching_results[r].transformation,
                        matching_results[r].information, pose_graph)
                    
        if valid_last_segment_registration:

            option = o3d.pipelines.registration.GlobalOptimizationOption(
                # read from config file
                max_correspondence_distance=self.max_correspondence_distance_fine,
                edge_prune_threshold=self.prune_pgo,
                preference_loop_closure=self.lc_pref,
                reference_node=0)
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Info) as cm:
                tic = time.perf_counter()
                o3d.pipelines.registration.global_optimization(
                    pose_graph,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    option)
                toc = time.perf_counter()
                self.pgo_times.append(toc - tic)
        else:
            return None


        self.compute_odometry_errors(iter=n_segments)
        self.compute_loop_errors(matching_results, iter=n_segments)

        pts, camera = self.apply_correction(pose_graph)

        self.compute_odometry_errors(after_correction=True, iter=n_segments)
        self.compute_loop_errors(
            matching_results, after_correction=True, iter=n_segments)
        self.compute_registration_errors(matching_results)
        self.create_plots_odometry(n_segments - 1)
        self.create_plots_loop(n_segments - 1)
        return pts, camera

    def get_keyframe_orb(self, img):
        img = np.float32(img.cpu().numpy()*255)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(np.uint8(gray_img), None)
        return des

    @torch.no_grad()
    def compute_tsdf(self):
        old_segment_key = list(self.fragments_dict.keys())[-2]
        last_segment_key = list(self.fragments_dict.keys())[-1]
        old_segment = self.fragments_dict[old_segment_key]
        last_segment = self.fragments_dict[last_segment_key]

        start_idx = old_segment["start_idx"]
        end_idx = last_segment["start_idx"]

        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        # compensate_vector = (-0.0 * scale / 512.0, 2.5 *
        #                  scale / 512.0, -2.5 * scale / 512.0)

        print(start_idx, end_idx)

        for idx in range(start_idx, end_idx):
            # print(idx)
            _, gt_color, gt_depth, _ = self.frame_reader[idx]
            depth = gt_depth.cpu().numpy()
            color = gt_color.cpu().numpy()
            c2w = self.estimate_c2w_list[idx].clone().cpu()

            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(
                np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.W, self.H, self.fx, self.fy, self.cx, self.cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1.0,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        o3d_mesh = volume.extract_triangle_mesh()
        # o3d_mesh = o3d_mesh.translate(compensate_vector)

        # o3d.io.write_triangle_mesh(os.path.join(f"{self.output}/mesh", f"mesh_{old_segment_key}.ply"), o3d_mesh)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d_mesh.vertices
        # pcd.colors = o3d_mesh.vertex_colors

        old_segment["points"] = np.asarray(o3d_mesh.vertices)
        old_segment["points_color"] = np.asarray(o3d_mesh.vertex_colors)

        # o3d.io.write_point_cloud(os.path.join(f"{self.output}/mesh", f"pcd_{old_segment_key}.ply"), pcd)

    def compute_tsdf_current(self):
        last_segment_key = list(self.fragments_dict.keys())[-1]
        last_segment = self.fragments_dict[last_segment_key]

        end_idx = last_segment["start_idx"]
        start_idx = end_idx - 5  # set in config

        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        # compensate_vector = (-0.0 * scale / 512.0, 2.5 *
        #                  scale / 512.0, -2.5 * scale / 512.0)

        print(start_idx, end_idx)

        for idx in range(start_idx, end_idx):
            # print(idx)
            _, gt_color, gt_depth, _ = self.frame_reader[idx]
            depth = gt_depth.cpu().numpy()
            color = gt_color.cpu().numpy()
            c2w = self.estimate_c2w_list[idx].clone().cpu()

            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(
                np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.W, self.H, self.fx, self.fy, self.cx, self.cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1.0,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        o3d_mesh = volume.extract_triangle_mesh()
        # o3d_mesh = o3d_mesh.translate(compensate_vector)

        # o3d.io.write_triangle_mesh(os.path.join(f"{self.output}/mesh", f"mesh_{old_segment_key}.ply"), o3d_mesh)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d_mesh.vertices
        # pcd.colors = o3d_mesh.vertex_colors

        last_segment["points"] = np.asarray(o3d_mesh.vertices).tolist()
        last_segment["points_color"] = np.asarray(
            o3d_mesh.vertex_colors).tolist()

    
    def compute_dbow_score(self):
        old_segment_key = list(self.fragments_dict.keys())[-2]
        last_segment_key = list(self.fragments_dict.keys())[-1]
        old_segment = self.fragments_dict[old_segment_key]
        last_segment = self.fragments_dict[last_segment_key]

        start_idx = old_segment["start_idx"]
        end_idx = last_segment["start_idx"]

        # end_idx = min(start_idx + 20, end_idx) #read x frames from config

        print(start_idx, end_idx)

        voc = bow.Vocabulary()
        voc.load(self.cfg["orbvoc"])
        db = bow.Database()
        db.setVocabulary(self.voc)

        for idx in range(start_idx+1, end_idx):
            _, gt_color, _ , _ = self.frame_reader[idx]
            features = self.get_keyframe_orb(gt_color)
            db.add(features)

        _, gt_color_keyframe, _ , _ = self.frame_reader[start_idx]
        features = self.get_keyframe_orb(gt_color_keyframe)
        results = db.query(features, end_idx-(start_idx+1))

        score_list = [result.Score for result in results]
        if len(score_list) > 0:
            self.dbow_scores.append(min(score_list))
        else:
            self.dbow_scores.append(-1)
    
    def apply_transformation(self):
        # needs to be 4, because we trigger pgo when we just started the 4th one on the 3 existing ones
        if self.new_segment:
            # compute tsdf fusion of old segment
            self.compute_tsdf()
            self.compute_dbow_score()
            self.new_segment = False
            if len(self.fragments_dict) > 3:
                # if not self.use_old_segments_only:
                #     self.compute_tsdf_current()
                output = self.compute_correction()
                if output is None:
                    return None
                else:
                    pts, camera = output

                del self.index
                self.index = faiss.index_cpu_to_gpu(self.resource,
                                                    self.cuda_id,
                                                    faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
                self.index.nprobe = self.nprobe
                self.index.train(pts)
                self.index.add(pts)
                return camera
            else:
                return None
        else:
            return None

    def update_fragments(self, method: Literal["fixed", "rot_trans"], idx: torch.Tensor, cur_c2w: torch.Tensor, pts: torch.Tensor, pts_color: torch.Tensor, gt_color: torch.Tensor, gt_depth: torch.Tensor, npc: torch.Tensor, geo_feats: torch.Tensor, col_feats: torch.Tensor, gt_camera: torch.Tensor, init: dict):
        if method == "fixed":
            index_pc = idx.item() // self.fixed_segment_size
            new = idx.item() % self.fixed_segment_size

            if self.fragments_dict is None:  # first fragment
                self.fragments_dict = {f"segment_{index_pc}": {"keyframe": cur_c2w.detach().clone().cpu(), "points": pts.cpu().tolist(), "points_color": pts_color.cpu().tolist(), "gt_color": gt_color.detach().clone(
                ).cpu(), "gt_depth": gt_depth.detach().clone().cpu(), "npc": npc.tolist(), "geo_feats": geo_feats, "col_feats": col_feats, "gt_camera": gt_camera.detach().clone().cpu(), "start_idx": idx.item(), "idx_start_segment_features" : 0, "mask": None}}
                # extract and add orb features
                features = self.extract_orb_features(
                    self.fragments_dict["segment_0"]["gt_color"], index_pc)
                self.add_orb_features(features)
            else:
                # new fragment
                if new == 0 and f"segment_{index_pc}" not in self.fragments_dict.keys():
                    init_npc = init["npc"].tolist()
                    idx_start_features = len(init_npc)
                    npc = init_npc + npc.tolist()
                    geo_feats = torch.cat([init["geo_feats"], geo_feats], 0)
                    col_feats = torch.cat([init["col_feats"], col_feats], 0)
                    self.fragments_dict[f"segment_{index_pc-1}"]["mask"] = init["mask"]

                    self.fragments_dict[f"segment_{index_pc}"] = {"keyframe": cur_c2w.detach().clone().cpu(), "points": pts.cpu().tolist(), "points_color": pts_color.cpu().tolist(), "gt_color": gt_color.detach().clone(
                    ).cpu(), "gt_depth": gt_depth.detach().clone().cpu(), "npc": npc, "geo_feats": geo_feats, "col_feats": col_feats, "gt_camera": gt_camera.detach().clone().cpu(), "start_idx": idx.item(), "idx_start_segment_features" : idx_start_features, "mask": None}
                    # extract, add and query orb features
                    print("new segment")
                    features = self.extract_orb_features(
                        self.fragments_dict[f"segment_{index_pc}"]["gt_color"], index_pc)
                    # results = self.query_bow(features, 5)
                    self.add_orb_features(features)
                    # pts = self.compute_correction(index_pc, results)
                    self.new_segment = True
                else:
                    self.fragments_dict[f"segment_{index_pc}"]["points"] += pts.cpu(
                    ).tolist()
                    self.fragments_dict[f"segment_{index_pc}"]["points_color"] += pts_color.cpu(
                    ).tolist()
                    self.fragments_dict[f"segment_{index_pc}"]["npc"] += npc.tolist()
                    self.fragments_dict[f"segment_{index_pc}"]["geo_feats"] = torch.cat(
                        [self.fragments_dict[f"segment_{index_pc}"]["geo_feats"], geo_feats], 0)
                    self.fragments_dict[f"segment_{index_pc}"]["col_feats"] = torch.cat(
                        [self.fragments_dict[f"segment_{index_pc}"]["col_feats"], col_feats], 0)
        elif method == "rot_trans":
            if self.fragments_dict is None:  # first segment
                self.fragments_dict = {"segment_0": {"keyframe": cur_c2w.detach().clone().cpu(), "idx": idx, "points": pts.tolist(), "points_color": pts_color.tolist(), "gt_color": gt_color.detach().clone(
                ), "gt_depth": gt_depth.detach().clone().cpu(), "npc": npc.tolist(), "geo_feats": geo_feats, "col_feats": col_feats, "gt_camera": gt_camera.detach().clone().cpu(), "start_idx": idx.item(), "idx_start_segment_features" : 0, "mask": None}}
                features = self.extract_orb_features(
                    self.fragments_dict["segment_0"]["gt_color"], 0)
                self.add_orb_features(features)
            else:
                last_segment = list(self.fragments_dict.keys())[-1]
                last_keyframe = self.fragments_dict[last_segment]["keyframe"]
                if compute_rel_trans(last_keyframe, cur_c2w) > self.segment_rel_trans or compute_cos_rel_rot(last_keyframe, cur_c2w) < self.segment_rot_cos:  # 20°
                    print("new segment")

                    init_npc = init["npc"].tolist()
                    idx_start_features = len(init_npc)
                    npc = init_npc + npc.tolist()
                    geo_feats = torch.cat([init["geo_feats"], geo_feats], 0)
                    col_feats = torch.cat([init["col_feats"], col_feats], 0)

                    seg_idx_old = int(last_segment.split("_")[-1])
                    self.fragments_dict[f"segment_{seg_idx_old}"]["mask"] = init["mask"]
                    seg_idx = int(last_segment.split("_")[-1]) + 1
                    self.fragments_dict[f"segment_{seg_idx}"] = {"keyframe": cur_c2w.detach().clone().cpu(), "idx": idx, "points": pts.tolist(), "points_color": pts_color.tolist(), "gt_color": gt_color.detach(
                    ).clone(), "gt_depth": gt_depth.detach().clone().cpu(), "npc": npc, "geo_feats": geo_feats, "col_feats": col_feats, "gt_camera": gt_camera.detach().clone().cpu(), "start_idx": idx.item(), "idx_start_segment_features" : idx_start_features, "mask": None}
                    features = self.extract_orb_features(
                        self.fragments_dict[f"segment_{seg_idx}"]["gt_color"], seg_idx)
                    # results = self.query_bow(features, 5)
                    self.add_orb_features(features)
                    self.new_segment = True
                else:
                    self.fragments_dict[last_segment]["points"] += pts.tolist()
                    self.fragments_dict[last_segment]["points_color"] += pts_color.tolist()
                    self.fragments_dict[last_segment]["npc"] += npc.tolist()
                    self.fragments_dict[last_segment]["geo_feats"] = torch.cat(
                        [self.fragments_dict[last_segment]["geo_feats"], geo_feats], 0)
                    self.fragments_dict[last_segment]["col_feats"] = torch.cat(
                        [self.fragments_dict[last_segment]["col_feats"], col_feats], 0)
        else:
            raise NotImplementedError

    def init_segment(self, cur_c2w):
        last_segment = list(self.fragments_dict.keys())[-1]
        last_segment = self.fragments_dict[last_segment]
        c2w = cur_c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        pts = np.asarray(last_segment["npc"])
        ones = np.ones_like(pts[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [pts, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[self.fx, .0, self.cx], [.0, self.fy, self.cy],
                      [.0, .0, 1.0]]).reshape(3, 3)
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)
        edge = 20
        mask = (uv[:, 0] < self.W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < self.H-edge)*(uv[:, 1] > edge)
        mask = mask.reshape(-1)
        pts = pts[mask]
        geo_feats = last_segment["geo_feats"].detach().clone()
        col_feats = last_segment["col_feats"].detach().clone()
        mask_torch = torch.from_numpy(mask)
        geo_feats = geo_feats[mask_torch]
        col_feats = col_feats[mask_torch]
        self.index.train(torch.tensor(pts.tolist(), device=self.device))
        self.index.add(torch.tensor(pts.tolist(), device=self.device))
        print(f"init added {pts.shape} points")
        return {"npc": pts, "geo_feats": geo_feats, "col_feats": col_feats, "mask" : mask}

    def get_cloud_pos(self, end=False):
        if end:
            npc = []
            first_fragment = list(self.fragments_dict.keys())[0]
            npc_old = np.array(self.fragments_dict[first_fragment]["npc"])
            mask_old = np.array([False] * len(npc_old))
            counter_old = np.array([0]*len(npc_old))
            for fragment in list(self.fragments_dict.keys())[:-1]:
                npc_s = np.array(self.fragments_dict[fragment]["npc"])
                counter_s = np.array([1]*len(npc_s))
                mask_s = self.fragments_dict[fragment]["mask"]
                idx = self.fragments_dict[fragment]["idx_start_segment_features"]
                counter_s[:idx] += counter_old[mask_old]
                npc_s[:idx] += npc_old[mask_old]
                npc += (npc_s[~mask_s] / counter_s[..., np.newaxis][~mask_s]).tolist()
                npc_old = npc_s
                mask_old = mask_s
                counter_old = counter_s
            last_segment = list(self.fragments_dict.keys())[-1]
            npc_last = np.array(self.fragments_dict[last_segment]["npc"])
            counter_last = np.array([1]*len(npc_last))
            idx = self.fragments_dict[last_segment]["idx_start_segment_features"]
            counter_last[:idx] += counter_old[mask_old]
            npc_last[:idx] += npc_old[mask_old]
            npc += (npc_last / counter_last[..., np.newaxis]).tolist()
            print(f"final number of points: {len(npc)}")
            return npc
        else:
            last_segment = list(self.fragments_dict.keys())[-1]
            return self.fragments_dict[last_segment]["npc"]

    def check_index(self, method: Literal["fixed", "rot_trans"], idx: Optional[torch.Tensor], cur_c2w: Optional[torch.Tensor] = None):
        if method == "fixed":
            index_pc = idx.item() // self.fixed_segment_size
            new = idx.item() % self.fixed_segment_size

            if self.fragments_dict is None:
                return None
            elif new == 0 and f"segment_{index_pc}" not in self.fragments_dict.keys():
                #self.resource = faiss.StandardGpuResources()
                del self.index
                self.index = faiss.index_cpu_to_gpu(self.resource,
                                                    self.cuda_id,
                                                    faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
                self.index.nprobe = self.nprobe
                init = self.init_segment(cur_c2w)
                return init
            return None
        elif method == "rot_trans":
            if self.fragments_dict is None:
                return None
            last_segment = list(self.fragments_dict.keys())[-1]
            last_keyframe = self.fragments_dict[last_segment]["keyframe"]
            if compute_rel_trans(last_keyframe, cur_c2w) > self.segment_rel_trans or compute_cos_rel_rot(last_keyframe, cur_c2w) < self.segment_rot_cos:
                del self.index
                self.index = faiss.index_cpu_to_gpu(self.resource,
                                                    self.cuda_id,
                                                    faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
                self.index.nprobe = self.nprobe
                init = self.init_segment(cur_c2w)
                return init
            return None
        else:
            raise NotImplementedError

    def check_rot_trans(self, cur_c2w):
        if self.fragments_dict is None:
            return True
        else:
            last_segment = list(self.fragments_dict.keys())[-1]
            last_keyframe = self.fragments_dict[last_segment]["keyframe"]
            if compute_rel_trans(last_keyframe, cur_c2w) > self.segment_rel_trans or compute_cos_rel_rot(last_keyframe, cur_c2w) < self.segment_rot_cos:
                return True
            else:
                return False

    def cloud(self, index=None):
        if index is None:
            return self._cloud
        return self._cloud[index]

    def append_cloud(self, value):
        self._cloud.append(value)

    # def cloud_pos(self, index=None):
    #     if index is None:
    #         return self._cloud_pos
    #     return self._cloud_pos[index]

    def input_pos(self):
        return self._input_pos

    def input_rgb(self):
        return self._input_rgb

    def input_normal(self):
        return self._input_normal

    def input_normal_cartesian(self):
        return self._input_normal_cartesian

    # def append_cloud_pos(self, value):
    #     self._cloud_pos.append(value)

    def pts_num(self):
        return self._pts_num

    def add_pts_num(self):
        self._pts_num += 1

    def set_pts_num(self, value):
        self._pts_num = value

    def set_keyframe_dict(self, value):
        self.keyframe_dict = value

    def get_keyframe_dict(self):
        return clone_kf_dict(self.keyframe_dict)

    def get_index(self):
        return self.index

    def index_is_trained(self):
        return self.index.is_trained

    def index_train(self, xb):
        assert torch.is_tensor(xb), 'use tensor to train FAISS index'
        self.index.train(xb)
        return self.index.is_trained

    def train_index_global(self):
        #self.resource = faiss.StandardGpuResources()
        del self.index
        self.index = faiss.index_cpu_to_gpu(self.resource,
                                            self.cuda_id,
                                            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2))
        self.index.nprobe = self.nprobe
        self.index.train(torch.tensor(
            self.get_cloud_pos(end=True), device=self.device))
        self.index.add(torch.tensor(
            self.get_cloud_pos(end=True), device=self.device))

    def index_ntotal(self):
        return self.index.ntotal

    def index_set_nprobe(self, value):
        self.index.nprobe = value

    def index_get_nprobe(self):
        return self.index.nprobe

    def get_device(self):
        return self.device

    def get_c_dim(self):
        return self.c_dim

    def get_radius_query(self):
        return self.radius_query

    def get_radius_add(self):
        return self.radius_add

    def set_nn_num(self, n):
        self.nn_num = n

    def get_fragments(self):
        return self.fragments_dict

    def set_fragment_npc(self, segment, npc):
        self.fragments_dict[segment]["npc"] = npc

    def get_segments_keyframe_dict(self):
        keyframe_dict = {}
        keyframe_list = []
        idx = 0
        for fragment in self.fragments_dict.keys():
            keyframe_dict[idx] = {"depth": self.fragments_dict[fragment]["gt_depth"].to(self.device), "color": self.fragments_dict[fragment]["gt_color"].to(
                self.device), "est_c2w": self.fragments_dict[fragment]["keyframe"].to(self.device), "frame": self.fragments_dict[fragment]["start_idx"]}
            idx += 1
            keyframe_list.append(idx)
        return keyframe_dict, keyframe_list

    def get_geo_feats(self, end=False):
        if end:
            if self.end_geo_feats is not None:
                return self.end_geo_feats
            geo_feats = []
            #first segment
            first_fragment = list(self.fragments_dict.keys())[0]
            feats_old = self.fragments_dict[first_fragment]["geo_feats"].detach().cpu()
            mask_old = torch.tensor([False] * len(feats_old))
            counter_old = torch.tensor([0]*len(feats_old))
            for fragment in list(self.fragments_dict.keys())[:-1]:
                # print(fragment)
                feats_s = self.fragments_dict[fragment]["geo_feats"].detach().cpu()
                counter_s = torch.tensor([1]*len(feats_s))
                mask_s = torch.from_numpy(self.fragments_dict[fragment]["mask"])
                idx = self.fragments_dict[fragment]["idx_start_segment_features"]
                counter_s[:idx] += counter_old[mask_old]
                feats_s[:idx] += feats_old[mask_old]
                geo_feats += (feats_s[~mask_s] / counter_s[~mask_s].unsqueeze(-1))
                feats_old = feats_s
                mask_old = mask_s
                counter_old = counter_s
            last_segment = list(self.fragments_dict.keys())[-1]
            feats_last = self.fragments_dict[last_segment]["geo_feats"].detach().cpu()
            counter_last = torch.tensor([1]*len(feats_last))
            idx = self.fragments_dict[last_segment]["idx_start_segment_features"]
            counter_last[:idx] += counter_old[mask_old]
            feats_last[:idx] += feats_old[mask_old]
            geo_feats += (feats_last / counter_last.unsqueeze(-1))
            self.end_geo_feats = torch.vstack(geo_feats).cuda()
            print(f"final number of geo_feats: {self.end_geo_feats.shape}")
            return self.end_geo_feats
        else:
            last_segment = list(self.fragments_dict.keys())[-1]
            return self.fragments_dict[last_segment]["geo_feats"]

    def get_col_feats(self, end=False):
        if end:
            if self.end_col_feats is not None:
                return self.end_col_feats
            col_feats = []
            first_fragment = list(self.fragments_dict.keys())[0]
            feats_old = self.fragments_dict[first_fragment]["col_feats"].detach().cpu()
            mask_old = torch.tensor([False] * len(feats_old))
            counter_old = torch.tensor([0]*len(feats_old))
            for fragment in list(self.fragments_dict.keys())[:-1]:
                # print(fragment)
                feats_s = self.fragments_dict[fragment]["col_feats"].detach().cpu()
                counter_s = torch.tensor([1]*len(feats_s))
                mask_s = torch.from_numpy(self.fragments_dict[fragment]["mask"])
                idx = self.fragments_dict[fragment]["idx_start_segment_features"]
                counter_s[:idx] += counter_old[mask_old]
                feats_s[:idx] += feats_old[mask_old]
                col_feats += (feats_s[~mask_s] / counter_s[~mask_s].unsqueeze(-1))
                feats_old = feats_s
                mask_old = mask_s
                counter_old = counter_s
            last_segment = list(self.fragments_dict.keys())[-1]
            feats_last = self.fragments_dict[last_segment]["col_feats"].detach().cpu()
            counter_last = torch.tensor([1]*len(feats_last))
            idx = self.fragments_dict[last_segment]["idx_start_segment_features"]
            counter_last[:idx] += counter_old[mask_old]
            feats_last[:idx] += feats_old[mask_old]
            col_feats += (feats_last / counter_last.unsqueeze(-1))
            self.end_col_feats = torch.vstack(col_feats).cuda()
            print(f"final number of col_feats: {self.end_col_feats.shape}")
            return self.end_col_feats
        else:
            last_segment = list(self.fragments_dict.keys())[-1]
            return self.fragments_dict[last_segment]["col_feats"]
    
    def get_registration_times(self):
        return self.registration_times
    
    def get_pgo_times(self):
        return self.pgo_times
    
    def update_geo_feats(self, feats, indices=None, end=False):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if end:
            if indices is not None:
                self.end_geo_feats[indices] = feats.clone().detach()
            else:
                assert feats.shape[0] == self.end_geo_feats.shape[0], 'feature shape[0] mismatch'
                self.end_geo_feats = feats.clone().detach()
        else:
            last_segment = list(self.fragments_dict.keys())[-1]
            if indices is not None:
                self.fragments_dict[last_segment]["geo_feats"][indices] = feats.clone(
                ).detach()
            else:
                assert feats.shape[0] == self.fragments_dict[last_segment]["geo_feats"].shape[0], 'feature shape[0] mismatch'
                self.fragments_dict[last_segment]["geo_feats"] = feats.clone(
                ).detach()

    def update_col_feats(self, feats, indices=None, end=False):
        assert torch.is_tensor(feats), 'use tensor to update features'
        if end:
            if indices is not None:
                self.end_col_feats[indices] = feats.clone().detach()
            else:
                assert feats.shape[0] == self.end_col_feats.shape[0], 'feature shape[0] mismatch'
                self.end_col_feats = feats.clone().detach()
        else:
            last_segment = list(self.fragments_dict.keys())[-1]
            if indices is not None:
                self.fragments_dict[last_segment]["col_feats"][indices] = feats.clone(
                ).detach()
            else:
                assert feats.shape[0] == self.fragments_dict[last_segment]["col_feats"].shape[0], 'feature shape[0] mismatch'
                self.fragments_dict[last_segment]["col_feats"] = feats.clone(
                ).detach()

    def cart2sph(self, xyz):
        # transform normals from cartesian to sphere angles
        # xyz should be tensor of N*3, and normalized
        normals_sph = torch.zeros(xyz.shape[0], 2, device=xyz.device)
        xy = xyz[:, 0]**2 + xyz[:, 1]**2
        normals_sph[:, 0] = torch.atan2(torch.sqrt(xy), xyz[:, 2])
        normals_sph[:, 1] = torch.atan2(xyz[:, 1], xyz[:, 0])
        return normals_sph

    def add_neural_points(self, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                          train=False, is_pts_grad=False, dynamic_radius=None, idx=None, gt_color=None, gt_depth=None, cur_c2w=None, gt_camera=None):
        """
        Add multiple neural points, will use depth filter when getting these samples.

        Args:
            batch_rays_o (tensor): (N,3)
            batch_rays_d (tensor): (N,3)
            batch_gt_depth (tensor): (N,)
            batch_gt_color (tensor): (N,3)
        """

        if batch_rays_o.shape[0]:
            init = self.check_index(method=self.segment_strategy,
                                    idx=idx, cur_c2w=cur_c2w)
            mask = batch_gt_depth > 0
            batch_gt_color = batch_gt_color*255
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                batch_rays_o[mask], batch_rays_d[mask], batch_gt_depth[mask], batch_gt_color[mask]

            pts_gt = batch_rays_o[..., None, :] + batch_rays_d[...,
                                                               None, :] * batch_gt_depth[..., None, None]
            mask = torch.ones(pts_gt.shape[0], device=self.device).bool()
            pts_gt = pts_gt.reshape(-1, 3)

            if self.index.is_trained:
                _, _, neighbor_num_gt = self.find_neighbors_faiss(
                    pts_gt, step='add', is_pts_grad=is_pts_grad, dynamic_radius=dynamic_radius)
                mask = (neighbor_num_gt == 0)

            self._input_pos.extend(pts_gt[mask].tolist())
            self._input_rgb.extend(batch_gt_color[mask].tolist())

            gt_depth_surface = batch_gt_depth.unsqueeze(
                -1).repeat(1, self.N_add)
            t_vals_surface = torch.linspace(
                0.0, 1.0, steps=self.N_add, device=self.device)

            if self.fix_interval_when_add_along_ray:
                # add along ray, interval unrelated to depth
                intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                           device=self.device).unsqueeze(0)
                z_vals = gt_depth_surface + intervals
            else:  # add along ray, interval related to depth
                z_vals_surface = self.near_end_surface*gt_depth_surface * (1.-t_vals_surface) + \
                    self.far_end_surface * \
                    gt_depth_surface * (t_vals_surface)
                z_vals = z_vals_surface

            pts = batch_rays_o[..., None, :] + \
                batch_rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts[mask]  # use mask from pts_gt for auxiliary points
            pts = pts.reshape(-1, 3)

            #self._cloud_pos += pts.tolist()
            self._pts_num += pts.shape[0]

            geo_feats = torch.zeros(
                [pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)
            col_feats = torch.zeros(
                [pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)

            self.update_fragments(method=self.segment_strategy, idx=idx, cur_c2w=cur_c2w, pts=pts_gt, pts_color=batch_gt_color,
                                  gt_color=gt_color, gt_depth=gt_depth, npc=pts, geo_feats=geo_feats, col_feats=col_feats, gt_camera=gt_camera,
                                  init=init)

            if train or not self.index.is_trained:
                self.index.train(pts)
            self.index.train(torch.tensor(
                self.get_cloud_pos(), device=self.device))
            self.index.add(pts)

            return torch.sum(mask)
        else:
            return 0

    # def update_points(self, idxs, geo_feats, col_feats, detach=True):
    #     """
    #     Update point cloud features.

    #     Args:
    #         idxs (list):
    #         geo_feats (list of tensors):
    #         col_feats (list of tensors):
    #     """
    #     assert len(geo_feats) == len(col_feats), "feature size mismatch"
    #     for _, idx in enumerate(idxs):
    #         if detach:
    #             self._cloud[idx].geo_feat = geo_feats[_].clone().detach()
    #             self._cloud[idx].col_feat = col_feats[_].clone().detach()
    #         else:
    #             self._cloud[idx].geo_feat = geo_feats[_]
    #             self._cloud[idx].col_feat = col_feats[_]
    #     geo_feats = torch.cat(geo_feats, 0)
    #     col_feats = torch.cat(col_feats, 0)
    #     if detach:
    #         self.geo_feats[idxs] = geo_feats.clone().detach()
    #         self.col_feats[idxs] = col_feats.clone().detach()
    #     else:
    #         self.geo_feats[idxs] = geo_feats
    #         self.col_feats[idxs] = col_feats

    def find_neighbors_faiss(self, pos, step='add', retrain=False, is_pts_grad=False, dynamic_radius=None):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            step (str): 'add'|'query'
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points choosen based on color grad, will use smaller radius when looking for neighbors
            dynamic_radius (tensor, optional): choose every radius differently based on its color gradient

        Returns:
            indices: list of variable length list of neighbors index, [] if none
        """
        if (not self.index.is_trained) or retrain:
            print("retrain")
            self.index.train(self.get_cloud_pos())

        assert step in ['add', 'query', 'mesh']
        split_pos = torch.split(pos, 65000, dim=0)
        D_list = []
        I_list = []
        for split_p in split_pos:
            D, I = self.index.search(split_p.float(), self.nn_num)
            D_list.append(D)
            I_list.append(I)
        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        if step == 'query':
            radius = self.radius_query
        elif step == 'add':
            if not is_pts_grad:
                radius = self.radius_add
            else:
                radius = self.radius_min
        else:
            radius = self.radius_mesh

        if dynamic_radius is not None:
            try:
                assert pos.shape[0] == dynamic_radius.shape[0], 'shape mis-match for input points and dynamic radius'
                neighbor_num = (D < dynamic_radius.reshape(-1, 1)
                                ** 2).sum(axis=-1).int()
            except:
                neighbor_num = (D < radius**2).sum(axis=-1).int()
        else:
            neighbor_num = (D < radius**2).sum(axis=-1).int()

        return D, I, neighbor_num

    def merge_points(self, keyframe_list):
        pass

    def get_feature_at_pos(self, p, feat_name):  # not used, use this in decoder
        if torch.is_tensor(p):
            p = p.detach().cpu().numpy().reshape((-1, 3)).astype(np.float32)
        else:
            p = np.asarray(p).reshape((-1, 3)).astype(np.float32)
        D, I, neighbor_num = self.find_neighbors_faiss(p, step='query')
        D, I, neighbor_num = [torch.from_numpy(i).to(
            self.device) for i in (D, I, neighbor_num)]

        c = torch.zeros([p.shape[0], self.c_dim],
                        device=self.device).normal_(mean=0, std=0.01)
        has_neighbors = neighbor_num > 0

        c_temp = torch.cat([torch.sum(torch.cat([getattr(self._cloud[I[i, j].item()], feat_name) / (torch.sqrt(D[i, j])+1e-10)
                                                 for j in range(neighbor_num[i].item())], 0), 0) / torch.sum(1.0 / (torch.sqrt(D[i, :neighbor_num[i]])+1e-10))
                            for i in range(p.shape[0]) if neighbor_num[i].item() > 0], dim=0)
        c_temp = c_temp.reshape(-1, self.c_dim)
        c[has_neighbors] = c_temp

        return c, has_neighbors

    def sample_near_pcl(self, rays_o, rays_d, near, far, num):
        """
        For pixels with 0 depth readings, preferably sample near point cloud.

        Args:
            rays_o (tensor): _description_
            rays_d (tensor): _description_
            near : near end for sampling along this ray
            far: far end
            num (int): stratified sampling num between near and far
        """
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        n_rays = rays_d.shape[0]
        intervals = 25
        z_vals = torch.linspace(near, far, steps=intervals, device=self.device)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)

        if torch.is_tensor(far):
            far = far.item()
        z_vals_section = np.linspace(near, far, intervals)
        z_vals_np = np.linspace(near, far, num)
        z_vals_total = np.tile(z_vals_np, (n_rays, 1))

        pts_split = torch.split(pts, 65000)  # limited by faiss bug
        Ds, Is, neighbor_nums = [], [], []
        for pts_batch in pts_split:
            D, I, neighbor_num = self.find_neighbors_faiss(
                pts_batch, step='query')
            D, I, neighbor_num = D.cpu().numpy(), I.cpu().numpy(), neighbor_num.cpu().numpy()
            Ds.append(D)
            Is.append(I)
            neighbor_nums.append(neighbor_num)
        D = np.concatenate(Ds, axis=0)
        I = np.concatenate(Is, axis=0)
        neighbor_num = np.concatenate(neighbor_nums, axis=0)

        neighbor_num = neighbor_num.reshape((n_rays, -1))
        neighbor_num_bool = neighbor_num.reshape((n_rays, -1)).astype(bool)
        invalid = neighbor_num_bool.sum(axis=-1) < 2

        if invalid.sum(axis=-1) < n_rays:
            r, c = np.where(neighbor_num[~invalid].astype(bool))
            idx = np.concatenate(
                ([0], np.flatnonzero(r[1:] != r[:-1])+1, [r.size]))
            out = [c[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
            z_vals_valid = np.asarray([np.linspace(
                z_vals_section[item[0]], z_vals_section[item[1]], num=num) for item in out])
            z_vals_total[~invalid] = z_vals_valid

        invalid_mask = torch.from_numpy(invalid).to(self.device)
        return torch.from_numpy(z_vals_total).float().to(self.device), invalid_mask


class NeuralPoint(object):
    def __init__(self, idx, pos, c_dim, normal=None, device=None):
        """
        Init a neural point
        Args:
            idx (int): _description_
            pos (tensor): _description_
            c_dim (int): _description_
            normal (_type_, optional): _description_. Defaults to None.
            device (_type_, optional): _description_. Defaults to None.
        """
        assert torch.is_tensor(pos)
        self.position = pos.to(device=device)
        self.idx = idx
        if normal is not None:
            self.normal = torch.tensor(normal).to(device)
        self.geo_feat = torch.zeros(
            [1, c_dim], device=device).normal_(mean=0, std=0.01)
        self.col_feat = torch.zeros(
            [1, c_dim], device=device).normal_(mean=0, std=0.01)
