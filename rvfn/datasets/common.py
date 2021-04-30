import numpy as np
from nuscenes.nuscenes import Box
from typing import List, Tuple
from pyquaternion import Quaternion

from rvfn.config.dataset_config import VoxelGridConfig


class PointCloud:
    """
    Class representing a d-dimensional pointcloud where each point is stored
     as a column in the data matrix, and the first 3 dimensions represent
     x, y, z coordinates in space.
    This class takes code from and is compatible with Nuscenes devkit
     PointCloud class:https://github.com/nutonomy/nuscenes-devkit/blob/master/\
     python-sdk/nuscenes/utils/data_classes.py
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize a pointcloud and check correctness of dimensions.
        Args:
            points: <np.float: d, n>, n points of d-dimensions
        """
        assert points.ndim == 2
        assert points.shape[0] >= 3, \
            'Pointcloud must have at least 3 dimensions'
        self.points = points

    def nbr_dims(self) -> int:
        """
        Returns: number of dimensions of points in pointcloud
        """
        return self.points.shape[0]

    def nbr_points(self) -> int:
        """
        Returns: number of points in pointcloud
        """
        return self.points.shape[1]

    def add_dims(self, arr: np.ndarray):
        """
        Adds the values in arr as a new feature dimensions to each point
        Args:
            arr: <np.float: d', n> d' values for each point
        """
        assert arr.shape[1] == self.nbr_points(), 'Array shape: ' + \
                                                  str(arr.shape) + \
                                                  ' -- #points: ' + \
                                                  str(self.nbr_points())

        self.points = np.vstack((self.points, arr))

    def add_points(self, arr: np.ndarray, auto_pad=False):
        """
        Adds points to pointcloud.
        Args:
            arr: <np.float: d', n'> n' points with d' dimensions. If auto_pad
             is false, d' must equal pointcloud nbr_dims
            auto_pad: bool, if true the pointcloud dimensions will be padded
             with zeros to match arr dimensions. d' must be greater or equal to
             pointcloud nbr_dims
        """
        if auto_pad:
            assert arr.shape[0] >= self.nbr_dims()
            diff_dims = arr.shape[0] - self.nbr_dims()
            if diff_dims > 0:
                self.add_dims(np.zeros((arr.shape[0] - self.nbr_dims(),
                                        self.nbr_points())))
        else:
            assert arr.shape[0] == self.nbr_dims()

        self.points = np.concatenate((self.points, arr), axis=1)

    def clone(self):
        """
        Returns: A copy of the pointcloud
        """
        return PointCloud(self.points.copy())

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        Args:
            ratio: Fraction of points to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()),
                                        size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def translate(self, t: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        Args:
            t: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + t[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation matrix.
        Args:
            rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transformation.
        Args:
            transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        homogeneous_points = np.vstack(
            (self.points[:3, :], np.ones(self.nbr_points())))
        self.points[:3, :] = transf_matrix.dot(homogeneous_points)[:3, :]

    def shuffle(self):
        """
        Randomly shuffles the points in the pointcloud
        """
        np.random.shuffle(self.points.T)


class VoxelGrid:
    def __init__(self, pc: PointCloud, config: VoxelGridConfig = None):
        """
        Creates a VoxelGrid from a PointCloud. This may change the input
        pointcloud by changing the points order.
        Args:
            pc: PointCloud, input pointcloud
        """
        self.config = config if config is not None else VoxelGridConfig()

        # Randomly shuffle the points
        pc.shuffle()

        # voxel coordinates for each point in pointcloud
        self.voxels = np.floor(
            pc.points[:3, :].T / self.config.voxel_size).astype(np.int32)

        self.voxels, self.points_map, self.num_points_in_voxels = np.unique(
            self.voxels, axis=0,
            return_inverse=True,
            return_counts=True)

        self.max_points = self.config.max_points_per_voxel

        # features array per point
        # (+ 3 relative coordinates)
        self.features = np.zeros(
            shape=(len(self.voxels), self.max_points, pc.nbr_dims() + 3),
            dtype=np.float32)

        for i in range(len(self.voxels)):
            # Features of points that fall within the ith voxel
            points_in_voxel = pc.points[:, self.points_map == i]

            if self.num_points_in_voxels[i] > self.max_points:
                points_in_voxel = points_in_voxel[:, :self.max_points]
                self.num_points_in_voxels[i] = self.max_points

            pts_xyz = points_in_voxel[:3, :].T
            pts_offset_xyz = pts_xyz - np.mean(pts_xyz, axis=0)

            self.features[i, :self.num_points_in_voxels[i], :] = \
                np.concatenate((pts_offset_xyz, points_in_voxel.T), axis=1)


class VehicleBox:
    def __init__(self, xyz: List[float], wlh: List[float], yaw: float):
        self.xyz = xyz
        self.wlh = wlh
        self.yaw = yaw

    @classmethod
    def from_nuscenes_box(cls, box: Box):
        xyz = list(box.center)
        wlh = list(box.wlh)
        yaw, _, _ = box.orientation.yaw_pitch_roll
        return cls(xyz, wlh, yaw)

    @classmethod
    def from_list(cls, box: List[float]):
        """
        Args:
            box: [x, y, z, w, l, h, yaw]
        """
        box = list(box)
        xyz = box[:3]
        wlh = box[3:6]
        yaw = box[6]
        return cls(xyz, wlh, yaw)

    def to_list(self):
        return self.xyz + self.wlh + [self.yaw]

    def rotation(self) -> Quaternion:
        """
        Get rotation as a quaternion
        """
        return Quaternion(axis=(0.0, 0.0, 1.0), radians=self.yaw)

    def rotate(self, angle) -> None:
        """
        Rotate the box around its center z axis
        Args:
            angle: Angle change in radians
        """
        self.yaw += angle

    def translate(self, t: np.ndarray) -> None:
        """
        Applies a translation.
        :param t: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.xyz += t

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        References:
            Taken from nuscenes.data_classes.Box
        Args:
            wlh_factor: Multiply w, l, h by a factor to scale the box.
        Returns:
            <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = np.array(self.wlh) * wlh_factor

        # 3D bounding box corners.
        # (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.rotation().rotation_matrix, corners)

        # Translate
        x, y, z = self.xyz
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def distance(self, other, criteria='center2d'):
        """
        Function for calculating a "distance" measure between two bboxes
        Args:
            criteria: Distance criteria. Can be one of:
                - center2d: L2 distance of box centers in the xy plane.
                    This is what Nuscenes official evaluation uses for
                    calculating MAP.
        """

        if criteria == 'center2d':
            # L2 distance of box centers in the xy plane
            return np.linalg.norm(
                np.array(self.xyz[:2]) - np.array(other.xyz[:2]))
        else:
            raise NotImplementedError

    def __repr__(self):
        return str(self.to_list())


def bboxes_to_arrays(bboxes: List[Box]):
    ret = np.empty((len(bboxes), 7))

    for i in range(len(bboxes)):
        ret[i] = VehicleBox.from_nuscenes_box(bboxes[i]).to_list()

    return ret


def coords_in_range(coords, ranges):
    """
    Returns True if a 3D coordinate lies inside a range.
    Args:
        coords: list or tuple of 3 numbers
        ranges: list or tuple of 3 pairs of numbers indicating the range limits

    """
    for ax in range(3):
        if not ranges[ax][0] <= coords[ax] <= ranges[ax][1]:
            return False
    return True


def shadow_boxes_corners(bboxes):
    """
    Convert 3D bboxes to their shadow in the xy plane represented by the 4
     corner points.

    Args:
        bboxes: (N, 7) boxes in format x, y, z, w, l, h, yaw

    Returns: (N, 4, 2) 2D shadow bbox corners

    """
    num_boxes = bboxes.shape[0]

    lengths = bboxes[:, 4]
    widths = bboxes[:, 3]
    yaws = bboxes[:, -1]

    # The shadows of bboxes centered at 0
    shadows = np.array([[-lengths / 2, -lengths / 2, lengths / 2, lengths / 2],
                        [widths / 2, -widths / 2, -widths / 2, widths / 2]])
    # (2, 4, num_boxes) -> (num_boxes, 2, 4)
    shadows = shadows.transpose((2, 0, 1))

    # Rotation matrix to match the shadows to bboxes
    rot = np.array([
        [np.cos(yaws), -np.sin(yaws)],
        [np.sin(yaws), np.cos(yaws)]])
    # (2, 2, num_boxes) -> (num_boxes, 2, 2)
    rot = rot.transpose((2, 0, 1))

    for i in range(rot.shape[0]):
        shadows[i] = np.dot(rot[i], shadows[i])

    # (num_boxes, 2, 4) -> (num_boxes, 4, 2)
    shadows = shadows.transpose((0, 2, 1))

    # Move shadows to their original bbox center location
    translation = bboxes[:, :2]
    translation = np.tile(translation, (1, 4)).reshape((num_boxes, 4, 2))
    shadows += translation

    return shadows


def upright_boxes(boxes_corner):
    """
    Converts 2D bounding boxes to their smallest circumscribing box with
     sides orthogonal to the axes.
    Changes format from 4x2 corner coordinates to 4 coordinates indicating
     min_x, min_y, max_x and max_y

    Taken from:
     https://github.com/skyhehe123/VoxelNet-pytorch/blob/master/utils.py

    Args:
        boxes_corner: (N, 4, 2) bbox corner coordinates

    Returns:
        (N, 4) same bboxes in format br_x, br_y, tl_x, tl_y

    """
    N = boxes_corner.shape[0]

    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d


def upright_shadows(bboxes):
    """
    Converts 7 coded 3D bboxes to smallest upright (orthogonal to axes) boxes
     circumscribing their shadow in the xy plane.

    Args:
        bboxes: (N, 7) bboxes in format x, y, z, w, l , h, yaw

    Returns:
        (N, 4) 2D bboxes in format br_x, br_y, tl_x, tl_y
    """
    bboxes_corner = shadow_boxes_corners(bboxes)
    return upright_boxes(bboxes_corner)


def test():
    boxes = np.array([[1, 2, 3, 4, 5, 6, 0.5],
                      [3, 4, 0, 1, 2, 1, 2.0],
                      [5, 5, 5, 1, 3, 4, 0.6]])

    shadows = shadow_boxes_corners(boxes)
    print(shadows)


if __name__ == '__main__':
    test()
