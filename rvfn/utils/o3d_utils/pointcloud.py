"""
Helper functions for visualizing fused pointclouds
"""

from matplotlib import cm
import numpy as np
import open3d as o3d

from rvfn.datasets.common import PointCloud
from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset


def make_radar_lineset(radar_points, feats, color):
    """
    Make an Open3D lineset for radar velocities
    Args:
        radar_points: <np.float f, n> pointcloud points that contain radar
         features according to the feats dict.
        feats: dict specifying the index for each feature in the input
         pointcloud. The indices for x, y, and z must be sequential.
        color: color for the lines.

    Returns: An Open3D lineset
    """
    lineset_points = []
    lineset_lines = []

    for p in radar_points.T:
        vx = p[feats['vx']]
        vy = p[feats['vy']]
        source = p[feats['x']:feats['z'] + 1]
        dest = [p[feats['x']] + vx, p[feats['y']] + vy, p[feats['z']]]
        lineset_points.append(source)
        lineset_points.append(dest)

    for i in range(0, len(lineset_points), 2):
        lineset_lines.append([i, i + 1])

    colors = [color for _ in range(len(lineset_lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lineset_points),
        lines=o3d.utility.Vector2iVector(lineset_lines),
    )

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def make_o3d_pcd(in_pcd: PointCloud, feats: dict, mode='rgb', radar=True):
    """
    Makes an Open3D pointcloud and lineset for visualization of a fused
    pointcloud
    Args:
        in_pcd: PointCloud, input pointcloud.
        feats: dict specifying the index for each feature in the input
         pointcloud. The indices for x, y, and z must be sequential. Same for
         r, g, and b if applicable.
        mode: One of:
         - 'rgb': The point colors will be set based on RGB features
         - 'intensity': The point colors will be set based on Lidar intensity
         - None: The point colors won't be set
        radar: If True points with radar dimensions will be colored and a
         lineset denoting velocity of each of those points will be created

    Returns: an o3d pointcloud, and an o3d lineset. If radar==False, lineset
     will be None.
    """
    pcd = in_pcd.clone()
    xyz = pcd.points[feats['x']:feats['z'] + 1, :].T
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    colors = []
    radar_color = [0, 0.5, 1]  # radar color in middle blue
    lineset = None
    if mode == 'rgb':
        colors = pcd.points[feats['r']:feats['b'] + 1, :].T
        colors = np.array([[c / 255 for c in color] for color in colors])
    elif mode == 'intensity':
        intensities = pcd.points[feats['intensity'], :].T

        # Similar scaling as done in nuscenes-devkit for visualization
        # https://github.com/nutonomy/nuscenes-devkit/blob/20ab9c7df6b9fb731\
        # 9f32ffb5758dd5005a4d2ea/python-sdk/nuscenes/nuscenes.py#L532
        intensities = (intensities - np.min(intensities)) / (
                np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)

        viridis = cm.get_cmap('viridis', 12)
        colors = np.array([viridis(c)[:3] for c in intensities])
    elif mode is None:
        colors = None

    if radar:
        radar_mask = np.ones(pcd.nbr_points(), dtype=bool)
        radar_mask = np.logical_and(radar_mask,
                                    pcd.points[feats['rcs'], :] != 0)

        radar_mask = np.logical_or(radar_mask,
                                   pcd.points[feats['vx'], :] != 0)
        radar_mask = np.logical_or(radar_mask,
                                   pcd.points[feats['vy'], :] != 0)

        if colors is not None:
            colors[radar_mask] = radar_color

        radar_points = pcd.points[:, radar_mask]
        lineset = make_radar_lineset(radar_points, feats, radar_color)

    if colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pcd, lineset
