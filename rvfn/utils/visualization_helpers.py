"""
Miscellaneous functions and classes to help with visualization
"""
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image


def show_colored_depth(depth_map):
    """
    Displays depth_map with a color pallet
    """
    # Invert non-zero depths for better visibility
    inverted = np.copy(depth_map)
    valid_pixels = (inverted > 0.1)
    inverted[valid_pixels] = \
        100 - inverted[valid_pixels]
    image_jet = cv2.applyColorMap(
        np.uint8(inverted / np.amax(inverted) * 255),
        cv2.COLORMAP_JET)

    cv2.imshow("depths", image_jet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_colors(colour_scheme: str = 'primary'):
    """
    Provides an normalized numpy array of rgb colors according to TUM
        CI standard.

    References:
        https://gitlab.lrz.de/perception/radarseg/radarseg/visu/utils.py

    Arguments:
        colour_scheme: Specifies the returned list of colors, <str>.

    Return:
        colors: Normalized array of rgb values (n, 3), <float>
    """
    # TUM CI primary colormap
    primary = \
        [[0, 101, 189],
         [255, 255, 255],
         [0, 0, 0]]

    # TUM CI secondary colormap
    secondary = \
        [[0, 82, 147],
         [0, 51, 89],
         [88, 88, 88],
         [156, 157, 159],
         [217, 217, 217]]

    # TUM CI accent colormap
    accent = \
        [[218, 215, 203],
         [227, 114, 34],
         [162, 173, 0],
         [152, 198, 234],
         [100, 160, 200]]

    # TUM CI extended colormap
    extended = \
        [[0, 0, 0],
         [105, 8, 90],
         [15, 27, 95],
         [0, 119, 138],
         [0, 124, 48],
         [103, 154, 29],
         [225, 220, 0],
         [249, 186, 0],
         [214, 76, 19],
         [196, 7, 27],
         [156, 13, 22]]

    # TUM blues
    blues = \
        [[0, 0, 0],
         [0, 101, 189],
         [100, 160, 200],
         [152, 198, 234]]

    colors = {'primary': primary, 'secondary': secondary, 'accent': accent,
              'extended': extended, 'blues': blues}
    return np.array(colors[colour_scheme], np.float) / 255


def render_ego_centric_map(scene_map,
                           ego_pose,
                           axes_limits: list = None,
                           ax: Axes = None):
    """
    Renders a map centered around the associated ego pose.

    Reference:
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/\
            nuscenes/nuscenes.py

        Changes are mostly to allow asymmetric axes limits

    Arguments:
        scene_map: nuScenes map object
        ego_pose: nuScenes pose of the ego vehicle
        axes_limits: Axes limit in order left, right, up,
            and down, measured in meters.
        ax: Axes onto which to render
    """

    def crop_image(image: np.array,
                   x_px: int,
                   y_px: int,
                   axes_limit_px: np.array) -> np.array:
        x_min = int(x_px - axes_limit_px[0])
        x_max = int(x_px + axes_limit_px[1])
        y_min = int(y_px - axes_limit_px[2])
        y_max = int(y_px + axes_limit_px[3])

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    if axes_limits is None:
        axes_limits = [5, 55, 20, 20]

    map_mask = scene_map['mask']

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(ego_pose['translation'][0],
                                            ego_pose['translation'][1])
    scaled_limits_px = np.array(axes_limits) / map_mask.resolution
    scaled_limits_px = scaled_limits_px.astype(int)

    first_crop_limits_px = np.ones_like(scaled_limits_px) * np.max(
        scaled_limits_px) * np.sqrt(2)
    first_crop_limits_px = first_crop_limits_px.astype(int)

    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1],
                         first_crop_limits_px)

    # Rotate image.
    ypr_rad = Quaternion(ego_pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

    # Crop image.
    ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2,
                                 rotated_cropped.shape[0] / 2,
                                 scaled_limits_px)

    # Init axes and show image.
    # Set background to white and foreground (semantic prior) to gray.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    foreground_color = int(get_colors('secondary')[4][0] * 255)
    ego_centric_map[ego_centric_map == map_mask.foreground] = foreground_color
    ego_centric_map[ego_centric_map == map_mask.background] = 255
    ax.imshow(ego_centric_map,
              extent=[-axes_limits[0], axes_limits[1], -axes_limits[2],
                      axes_limits[3]],
              cmap='gray', vmin=0, vmax=255)


def render_annotation(annotation,
                      color=None,
                      ax=None,
                      linewidth: float = 1.0,
                      alpha: float = 0.7):
    """
    Renders the given annotation.

    Reference:
        https://github.com/nutonomy/nuscenes-devkit/python-sdk/nuscenes/\
            nuscenes.py

    Arguments:
        annotation: NuScenes annotation object (bounding box).
        color: Color of the bounding box, normalized rgb array(1, 3).
        ax: LIDAR view point, <matplotlib.axes.Axes>.
        linewidth: Width of annotation lines
        alpha: Alpha value for annotation lines.
    """

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color,
                    linewidth=linewidth, alpha=alpha)
            prev = corner

    # Initialize
    color = color if color is not None else np.array([0, 0, 0])

    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)

    # Get annotation corners
    corners = annotation.corners()

    # Draw the sides
    for i in range(4):
        ax.plot([corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=color, linewidth=linewidth, alpha=alpha)

    # Draw front (first 4 corners) and rear (last 4 corners)
    #   rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], color)
    draw_rect(corners.T[4:], color)

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    ax.plot([center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=color, linewidth=linewidth, alpha=alpha)
