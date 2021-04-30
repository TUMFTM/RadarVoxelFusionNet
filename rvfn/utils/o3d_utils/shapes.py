"""
Helper functions for creating different shapes from points in OpenCV
"""

import numpy as np
import open3d as o3d
import math


pi = math.pi


def make_horizontal_circle_pcd(radius, center=(0, 0, 0), n=200):
    """
    Make a circle of points parallel to the x-y plane
    Args:
        radius: Radius of the circle generated
        center: Center point of the circle [x, y, z]
        n: Number of point on the circle's circumference

    Returns: A pointcloud representing a circle

    """
    return [[math.cos(2 * pi / n * x) * radius + center[0],
             math.sin(2 * pi / n * x) * radius +
             center[1], center[2]] for x in range(0, n + 1)]


def make_line_segment_pcd(start, end, n=10):
    """
    Make a line-segment out of points
    Args:
        start: Starting point of the line-segment [x, y, z]
        end: End point of the line-segment [x, y, z]
        n: Number of points in line-segment

    Returns: A pointcloud representing a line-segment

    """
    x, y, z = [np.arange(start[i], end[i], (end[i] - start[i]) / n) for i in
               start]
    return list(zip(x, y, z))


def make_concentric_circles_pcd(num=10, dist=10, start=10):
    """
    Make equidistant concentric circles centered at the origin using points.
    Args:
        num: Number of circles
        dist: distance between circles
        start: starting radius

    Returns: Pointcloud representing concentric circles

    """
    circles = [make_horizontal_circle_pcd(r) for r in
               np.arange(start, dist * num + start, dist)]

    circles_pcd = [o3d.geometry.PointCloud() for i in range(len(circles))]
    for i in range(len(circles)):
        circles_pcd[i].points = o3d.utility.Vector3dVector(circles[i])
        circles_pcd[i].paint_uniform_color([1, 0.9, 0.5])  # distance circles in light yellow

    return circles_pcd
