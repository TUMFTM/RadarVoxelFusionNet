from itertools import product, chain, repeat
import numpy as np
import open3d as o3d


def back_project(coords, depths, camera_intrinsic, camera_rotation=None,
                 camera_translation=None):
    """
    Back project points from 2D to 3D.
    Args:
        coords: tuple of arrays (np.array<float>(1xn), np.array<float>(1xn)),
            where the first array contain the y(or v) and the second one the
            x(or u) coordinates of the points. This input format is chosen to
            increase performance.
        depths: array np.array<float>(1xn) containing a scalar depth for each
            of the points.
        camera_intrinsic: 3x3 camera intrinsic parameters
        camera_rotation: 3x3 camera rotation matrix
            (with respect to the target coordinate system)
        camera_translation: 1x3 camera translation
            (with respect to the target coordinate system)

    Returns:
        nx3 3D points in the target coordinate system
    """
    '''
    Camera intrinsics of form: [fx,  0, cx]
                               [ 0, fy, cy]
                               [ 0,  0,  1]
    '''

    cx = camera_intrinsic[0][2]
    cy = camera_intrinsic[1][2]

    fx = camera_intrinsic[0][0]
    fy = camera_intrinsic[1][1]

    coords_y, coords_x = coords

    coords_y -= cy
    coords_y *= depths / fy

    coords_x -= cx
    coords_x *= depths / fx

    points = np.column_stack((coords_x, coords_y, depths))

    # None is used as default and check here rather than using np.eye() and
    #  np.zeros() because np.dot() and operator+() still perform the operations
    #  in the trivial cases and reduce the performance.
    if camera_rotation is not None:
        points = np.dot(camera_rotation, points.T).T

    if camera_translation is not None:
        points += np.array(camera_translation)

    return points


def back_project_img(depth, camera_intrinsic, rgb=None,
                     camera_rotation=None,
                     camera_translation=None):
    """
    Back project image to a 3D pointcloud using pixel-by-pixel depth.
    Args:
        rgb: 3 channel pillow image
        depth: Depth matrix with same dimensions as rgb
        camera_intrinsic: 3x3 camera intrinsic parameters
        camera_rotation: 3x3 camera rotation matrix
            (with respect to the target coordinate system)
        camera_translation: 1x3 camera translation
            (with respect to the target coordinate system)

    Returns: List of 3D coordinates in the target coordinate system
             , list of RGB colors
    """

    height, width = depth.shape

    # Create arrays coord_v and coord_u containing all the possible (v, u)
    #  coordinates in the image.

    # width times     width times            width times
    # -----------     -----------     --------------------------
    # 0 0 0 0 0 0 ... 1 1 1 1 1 1 ... height-1 height-1 height-1
    coords_v = np.array(
        list(chain.from_iterable(repeat(i, width) for i in range(height))),
        dtype=float)

    #    height times
    # ------------------
    # 0 1 2 3 .. width-1
    coords_u = np.array(list(range(width)) * height, dtype=float)

    points = back_project((coords_v, coords_u), depth.flatten(),
                          camera_intrinsic,
                          camera_rotation, camera_translation)
    colors = []
    if rgb is not None:
        colors = np.array(rgb).reshape(-1, 3)

    return points, colors


def project_points(points, camera_intrinsics: np.ndarray, image_size,
                   min_dist=0.0):
    """
    Project points on to an image plane using camera intrinsics, and return
    mask of the ones that are within the image size.
    The code is taken from parts of "map_pointcloud" function in nuscenes
    devkit: https://github.com/nutonomy/nuscenes-devkit/blob/20ab9c7df6b9fb731\
    9f32ffb5758dd5005a4d2ea/python-sdk/nuscenes/nuscenes.py#L532
    Args:
        points: <np.float32: d, n> Matrix of points, where each point
         (x, y, z, ...) is along a column.
        camera_intrinsics: <np.float32: 3, 3> camera intrinsics matrix.
        image_size: (int, int) image width and height in pixels.
        min_dist: minimum distance (z) below which the mask will be false.

    Returns: <np.float32: d, m> Matrix of points, transformed and masked.
             <np.bool , n> boolean mask the size of number of input points,
             indicating whether each point was within camera FOV.

    """
    # Save the depths for filtering
    depths = points[2, :]

    viewpad = np.eye(4)
    viewpad[:camera_intrinsics.shape[0], :camera_intrinsics.shape[1]] = \
        camera_intrinsics

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    p_points = np.concatenate((points, np.ones((1, nbr_points))))
    p_points = np.dot(viewpad, p_points)
    p_points = p_points[:3, :]

    p_points = p_points / p_points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    # Indices for 2D pixel dimensions
    u = 0  # along the width axis
    v = 1  # along the height axis

    # Create a mask for choosing points that are at least a certain distance
    # away, and within the camera FOV.
    mask = np.ones(points.shape[1], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, p_points[u, :] >= 0)
    mask = np.logical_and(mask, p_points[u, :] < image_size[0])
    mask = np.logical_and(mask, p_points[v, :] >= 0)
    mask = np.logical_and(mask, p_points[v, :] < image_size[1])

    return p_points, mask


def show_pointcloud(ply_file):
    """
    Display pointcloud from file
    Args:
        ply_file: Pointcloud (.ply) file name
    """
    pcd = o3d.io.read_point_cloud(ply_file)  # Read the point cloud

    o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud


def save_pointcloud(points, colors, ply_file_name):
    """
    Save a pointcloud in ply format.
    Args:
        points: List of 3D point coordinates
        colors: List of RGB values, same length as points
        ply_file_name: Filename to save to

    """

    points_str = []
    for i in range(points.len()):
        points_str.append("%f %f %f %d %d %d\n" % (
            points[i][0], points[i][1], points[i][2],
            colors[i][0], colors[i][1], colors[i][2]))

    file = open(ply_file_name, "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
%s
''' % (len(points), "".join(points_str)))
    file.close()
