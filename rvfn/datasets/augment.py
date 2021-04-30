from rvfn.datasets.common import PointCloud, VehicleBox, \
    upright_shadows
from rvfn.config.dataset_config import AugmentationConfig

from nuscenes.utils.data_classes import Box

from typing import List, Tuple
import numpy as np
from pyquaternion import Quaternion
from torchvision.ops import box_iou
import torch


def contained_points(pc: PointCloud, box: Box):
    """
    Finds the points contained within box.

    Returns: Mask of points in pc which are within bounds of box
    """
    corners = box.corners(1.2)

    min_x = np.min(corners[0, :])
    min_y = np.min(corners[1, :])
    min_z = np.min(corners[2, :])

    max_x = np.max(corners[0, :])
    max_y = np.max(corners[1, :])
    max_z = np.max(corners[2, :])

    bound_x = np.logical_and(pc.points[0, :] >= min_x,
                             pc.points[0, :] <= max_x)
    bound_y = np.logical_and(pc.points[1, :] >= min_y,
                             pc.points[1, :] <= max_y)
    bound_z = np.logical_and(pc.points[2, :] >= min_z,
                             pc.points[2, :] <= max_z)

    return np.logical_and(np.logical_and(bound_x, bound_y), bound_z)


def z_rotation_matrix(yaw):
    """
    Rotation matrix for rotating "yaw" radians around the z axis
    Args:
        yaw: Rotation angle in radians

    Returns: np.array<3, 3> rotation matrix
    """
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0][0] = np.cos(yaw)
    rotation_matrix[0][1] = -np.sin(yaw)
    rotation_matrix[1][0] = np.sin(yaw)
    rotation_matrix[1][1] = np.cos(yaw)
    rotation_matrix[2][2] = 1

    return rotation_matrix


def augment(pc: PointCloud, boxes: List[Box],
            cfg: AugmentationConfig = None) -> Tuple[PointCloud, List[Box]]:
    """
    Apply random augmentations to the pointcloud and ground truth boxes
     according to: https://arxiv.org/pdf/1711.06396.pdf
    Augmentations performed are:
     - Random rotation on boxes along with the points that lie inside them
     - Random translation on boxes along with the points that lie inside them
     - Random global rotation on the entire pointcloud and boxes
     - Random global scaling on the entire pointcloud and boxes

    Args:
        pc: Pre-filtered pointcloud
        boxes: Ground-truth boxes
        cfg: Augmentation configurations

    Returns: augmented point cloud and boxes
    """
    cfg = cfg if cfg is not None else AugmentationConfig()

    np.random.seed()

    # Perturbate the bounding boxes along with the points inside them
    for idx in range(len(boxes)):
        box = boxes[idx]

        rotation = np.random.uniform(-cfg.box_rotation,
                                     cfg.box_rotation)
        # Normal distribution with std of 1 for x and y, and std of 0.3 for z
        translation = np.array([np.random.normal(0, cfg.box_translation[0]),
                                np.random.normal(0, cfg.box_translation[1]),
                                np.random.normal(0, cfg.box_translation[2])])

        center = box.center.copy()

        # Keep a copy in case we want to abort the change
        old_box = box.copy()

        # Mask of points in the pointcloud contained in the box
        contained_mask = contained_points(pc, box)

        # Transform the bounding box
        box.translate(-center)
        box.rotate(Quaternion(axis=(0.0, 0.0, 1.0), radians=rotation))
        box.translate(center)

        box.translate(translation)

        # Detect collision with other boxes, and undo the changes if detected
        vehicle_box = VehicleBox.from_nuscenes_box(box)
        box_upright = upright_shadows(np.array([vehicle_box.to_list()]))

        collision = False
        for other_idx in range(len(boxes)):
            if other_idx == idx:
                continue
            other_box = VehicleBox.from_nuscenes_box(boxes[other_idx])
            other_box_upright = upright_shadows(
                np.array([other_box.to_list()]))

            with torch.no_grad():
                iou = box_iou(torch.as_tensor(box_upright, dtype=torch.float),
                              torch.as_tensor(other_box_upright,
                                              dtype=torch.float))

            if iou > 0:
                collision = True
                break

        if collision:
            boxes[idx] = old_box
            continue

        # Rotation matrix for rotating "rotation" radians around the z axis
        rotation_matrix = z_rotation_matrix(rotation)

        # Continue if there are no points in this bbox
        if pc.points[:3, contained_mask].shape[1] == 0:
            continue

        # Rotate the points inside the box around
        #  the box's center in the z axis
        pc.points[:3, contained_mask] -= center.reshape((3, 1))
        pc.points[:3, contained_mask] = np.dot(rotation_matrix,
                                               pc.points[:3, contained_mask])
        pc.points[:3, contained_mask] += center.reshape((3, 1))

        # Translate the points inside the box
        pc.points[:3, contained_mask] += translation.reshape((3, 1))

    # Rotate the entire point cloud and boxes
    global_rotation = np.random.uniform(-cfg.global_rotation,
                                        cfg.global_rotation)

    global_rotation_mat = z_rotation_matrix(global_rotation)
    pc.rotate(global_rotation_mat)

    for box in boxes:
        box.rotate(Quaternion(axis=(0.0, 0.0, 1.0), radians=global_rotation))

    # Scale the entire point cloud and boxes
    global_scale = np.random.uniform(1.0 - cfg.global_scale,
                                     1.0 + cfg.global_scale)
    pc.points[:3, :] *= global_scale
    for box in boxes:
        box.center *= global_scale
        box.wlh *= global_scale

    return pc, boxes
