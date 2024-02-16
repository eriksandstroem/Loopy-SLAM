import open3d as o3d
import open3d.core as o3c
import numpy as np


class VisualOdometer(object):

    def __init__(self, cam_params: [], device="CUDA:0"):
        self.device = o3c.Device(device)
        _, _, fx, fy, cx, cy = cam_params
        self.intrinsics = o3d.core.Tensor(
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), o3d.core.Dtype.Float64)
        self.est_abs_poses = []
        self.est_rel_poses = []
        self.infos = []
        self.last_abs_pose = None
        self.last_frame = None
        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(
                100, relative_fitness=0.0001),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(50),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(30)]
        self.method = o3d.t.pipelines.odometry.Method.Hybrid
        self.max_depth = 10.0
        self.depth_scale = 1.0

    def _observations2frame(self, image: np.ndarray, depth: np.ndarray):
        """ Estimate the pose of the current frame
        Args:
            image: RGB image of type int in [0, 255] range
            depth: Depth image of type float in any range (max range considered as 10m)
        """
        return o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(image).to(self.device),
            o3d.t.geometry.Image(depth).to(self.device))

    def get_last_frame(self):
        return self.last_frame

    def update_est_abs_pose(self, pose):
        self.last_abs_pose = pose

    def set_init_state(self, image: np.ndarray, depth: np.ndarray, pose: np.ndarray):
        """ Estimate the pose of the current frame
        Args:
            image: RGB image of type int in [0, 255] range
            depth: Depth image of type float in any range (max range considered as 10m)
            pose: Camera pose in the form of 4x4 transformation matrix in camera space
        """
        self.last_frame = self._observations2frame(image, depth)
        self.last_abs_pose = pose

    def estimate_pose(self, image: np.ndarray, depth: np.ndarray):
        """ Estimate the pose of the current frame
        Args:
            image: RGB image of type int in [0, 255] range
            depth: Depth image of type float in any range (max range considered as 10m)
        """
        current_frame = self._observations2frame(image, depth)
        res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            self.last_frame, current_frame, self.intrinsics, o3c.Tensor(
                np.eye(4)),
            self.depth_scale, self.max_depth, self.criteria_list, self.method)

        rel_transofrm = res.transformation.cpu().numpy()
        # To accomodate the original data coordinate system
        rel_transofrm[0, 3] *= -1
        rel_transofrm[1, 2] *= -1
        rel_transofrm[2, 1] *= -1

        self.last_frame = current_frame.clone()
        self.last_abs_pose = self.last_abs_pose @ rel_transofrm
        self.est_rel_poses.append(rel_transofrm)
        return self.last_abs_pose
