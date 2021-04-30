"""
Classes and functions for visualizing LineSets in Open3D
"""

import open3d as o3d
import numpy as np
import itertools
from typing import List

from rvfn.datasets.common import VehicleBox


class LineSetBox:
    def __init__(self, corners: np.ndarray):
        """
        LineSetBox from box corners
        Args:
            corners: <np.float: 8, 3>, coordinates of the 8 corners of the box.
            first four points must belong to the same face, and in a CW or CCW
            order. And the second four points must be of the opposite face in
            the same order. If the direction of the box matters, the first four
            points must represent the front plane.
        """
        assert corners.shape == (8, 3), \
            'Input must be 8 points with 3 coordinates each.' \
            'got {}'.format(corners.shape)
        self.corners = corners

    @classmethod
    def from_voxel(cls, idx: List[int], voxel_size: List[float]):
        """
        LineSetBox from a voxel
        Args:
            idx: voxel index (coordinates within the voxel grid)
            voxel_size: Dimensions of each voxel (x, y, z)
        """
        assert len(idx) == len(
            voxel_size) == 3, 'Got idx: {}, size: {}'.format(idx, voxel_size)

        coords = np.zeros((3, 2))
        for ax in range(3):
            coords[ax][0] = idx[ax] * voxel_size[ax]
            coords[ax][1] = (idx[ax] + 1) * voxel_size[ax]

        combinations = list(itertools.product([0, 1], repeat=3))

        # Swap nodes positions to have the faces in a CCW order
        combinations[2], combinations[3] = combinations[3], combinations[2]
        combinations[6], combinations[7] = combinations[7], combinations[6]

        corners = [[coords[i][combinations[count][i]] for i in range(3)] for
                   count in range(8)]
        return cls(np.array(corners))

    @classmethod
    def from_list(cls, box: List[float]):
        return cls(VehicleBox.from_list(box).corners().T)

    def get_o3d_lineset(self, front_cross=False,
                        color=None) -> o3d.geometry.LineSet:
        """
        Makes an o3d LineSet representing the box.
        Args:
            front_cross: whether to mark the front plane with a cross
            color: color of the edges

        Returns: A LineSet containing 12 line segments representing the box.
         if front_cross==True there will be 2 extra lines in a cross shape
         on the front plane.
        """
        lines = [
            # Front plane
            [0, 1], [0, 3], [1, 2], [2, 3],
            # Back plane
            [4, 5], [4, 7], [5, 6], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

        if front_cross:
            lines += [[0, 2], [1, 3]]

        if color is None:
            color = [1, 0.6, 0]  # orange

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.corners),
            lines=o3d.utility.Vector2iVector(lines),
        )

        line_set.paint_uniform_color(color)

        return line_set


def combine_linesets(linesets: List[o3d.geometry.LineSet],
                     color=None) -> o3d.geometry.LineSet:
    """
    Combine a list of o3d LineSets into a single LineSet.
    This is useful for visualizing a long list of LineSets as Open3D has
    difficulties visualizing more than a certain number of them individually.
    Args:
        linesets: List of LineSets
        color: Color to paint the combined LineSet

    Returns: Combined LineDet

    """
    if color is None:
        color = [1, 0.6, 0]  # orange

    all_points = np.asarray(linesets[0].points)
    all_lines = np.asarray(linesets[0].lines)

    for i in range(1, len(linesets)):
        lines = np.asarray(linesets[i].lines)
        # fix the point indices for lines by adding the number of points
        # entered before it
        lines += all_points.shape[0]
        all_lines = np.vstack((all_lines, lines))

        points = np.asarray(linesets[i].points)
        all_points = np.vstack((all_points, points))

    ret_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(all_lines),
    )

    ret_lineset.paint_uniform_color(color)

    return ret_lineset
