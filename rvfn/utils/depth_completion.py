"""
This file contains different functions and classes for producing dense depth
maps from sparse LIDAR data.
Implemented methods:
    MaskedConvDepth:
        Provides arbitrary masked convolution for depth completion.
        Allows use of RGB data for guiding the depth estimation.
        Provides good results for sparse data, but is slow.

    KnnDepth:
        K-Nearest-Neighbours regression for depth completion

    ModifiedIpBasicDepth:
        IP-BASIC (https://arxiv.org/abs/1802.00036,
            https://github.com/kujason/ip_basic),
            modified to work with sparser data.
"""
import numpy as np
import cv2
import math
from sklearn.neighbors import KNeighborsRegressor


def color_distance(color_1, color_2):
    """
    Returns euclidean distance between two RGB colors
    """
    sum_diff = sum((color_1[i] - color_2[i]) ** 2 for i in range(3))
    return math.sqrt(sum_diff) / 255


def gaussian_kernel(l=5, sig_x=1., sig_y=1., normalize=False):
    """
    creates gaussian kernel with side length l and sigmas of sig_x and sig_y
    Adopted from an answer here: https://stackoverflow.com/q/29731726
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (
            np.square(xx) / np.square(sig_x)
            + np.square(yy) / np.square(sig_y)))
    if normalize:
        kernel /= np.sum(kernel)
    return kernel


class KnnDepth:
    """
    Depth completion based on K-nearest-neighbour regression.
    Fills depth array in-place.
    """

    def __init__(self, depth, k=10, weights='distance', n_jobs=-1):
        """
        Prepare data-structures for KNN
        Args:
            depth: Depth array (2D floating point)
             k: K parameter of KNN
            weights: Weight scheme to use for regression.
             See sklearn.KNeighborsRegressor for more details
            n_jobs: Number of parallel jobs to run.
        """
        self.depth = depth
        valid_points = np.transpose(np.nonzero(depth))
        valid_points_depths = [depth[x[0], x[1]] for x in valid_points]

        self.neigh = KNeighborsRegressor(n_neighbors=k, weights=weights,
                                         n_jobs=n_jobs)
        self.neigh.fit(valid_points, valid_points_depths)

    def fill_depth(self, grid=None, mask=None):
        """
        Fill depth in depth image in-place.
        Args:
            grid: Pair of arrays representing coordinates to be filled in format:
             ([v0, v1, v2, ...], [u0, u1, u2, ...]) in which case the pixels
             <u0v0, u1v1, u2v2, ...> will be filled. If None all pixels with value
             equal to zero filtered by mask will be filled.
            mask: If not None, prediction will only be made for pixels where the
             corresponding mask value is True. Must be same shape as depth-map.

        Returns: Filled depth array
        """
        if grid is None:
            # Mask of pixels were prediction is to be made on
            condition = (self.depth == 0)
            if mask:
                assert mask.shape == self.depth.shape
                condition &= mask

            predict = np.nonzero(condition)
        else:
            yy, xx = grid
            predict = (np.ravel(yy), np.ravel(xx))

        self.depth[predict] = self.neigh.predict(np.transpose(predict))

        return self.depth


class MaskedConvDepth:
    """
    Class for depth completion using Masked convolutions on sparse 2D arrays.
    Works by adding the values of each position where mask is set to True to
    the rest of the image, weighted according to the kernel and the RGB image.
    Only works for symmetrical kernels.
    """

    def __init__(self, depth, mask, rgb=None):
        """
        Prepares helper arrays for convolutions.
        Args:
            depth: Depth array
            mask: Which pixels to use. Must be same dimensions as depth.
            rgb: RGB image to use as guidance.
        """
        assert mask.shape == depth.shape
        self.height, self.width = depth.shape

        self.rgb = rgb
        self.depth = depth
        self.mask = mask

        # Array where the valid pixel values times corresponding weights from
        #  the kernel are accumulated.
        self.values = np.zeros(depth.shape)
        # Sum of weights used for each pixel.
        self.weights = np.zeros(depth.shape)

    def fill_depth(self, kernel=None, rgb_weight=0):
        """
        Creates a filled depth image based on kernel and mask.
        Args:
            kernel: Kernel to use for convolutions. If none a gaussian kernel
             with sigmas of 7 in x direction and 15 in y direction will be used
            rgb_weight: Weight of RGB signal. Higher value increases the
             effect of the RGB image on depth estimation.

        Returns: Filled depth array
        """
        if kernel is None:
            # Empirically chosen values for a single sweep of Nuscenes LIDAR.
            # The sparser the data, the bigger the kernel and the sigmas should
            # be. In this case the sparsity in the y direction is higher than
            # x direction, hence the asymmetry in the sigmas.
            kernel = gaussian_kernel(51, 7, 15)

        for i in range(self.height):
            for j in range(self.width):
                if self.mask[i, j]:
                    self.__add_kernel((i, j), kernel, rgb_weight)

        for i in range(self.height):
            for j in range(self.width):
                if self.weights[i, j] > 0:
                    self.depth[i, j] = self.values[i, j] / self.weights[i, j]

        return self.depth

    def __add_kernel(self, point, kernel, rgb_weight=0):
        """
        Adds weights at point according to kernel and rgb_weight
        """
        v, u = point
        point_value = self.depth[v, u]

        top_left_v = v - int(kernel.shape[0] / 2)
        top_left_u = u - int(kernel.shape[1] / 2)

        for i in range(kernel.shape[0]):
            curr_v = i + top_left_v
            if curr_v < 0 or curr_v >= self.height:
                continue
            for j in range(kernel.shape[1]):
                curr_u = j + top_left_u
                if curr_u < 0 or curr_u >= self.width:
                    continue

                weight = kernel[i, j]

                if self.rgb is not None:
                    weight += kernel[i, j] * rgb_weight * color_distance(
                        self.rgb.getpixel((curr_u, curr_v)),
                        self.rgb.getpixel((u, v)))

                self.values[curr_v, curr_u] += weight * point_value
                self.weights[curr_v, curr_u] += weight


class ModifiedIpBasicDepth:
    """
    Multi-scale dilation version of IP-BASIC from: https://github.com/kujason/ip_basic
    Relative to the original this uses bigger kernels to work with sparse data
     and forgoes use of a median blur stage (s4 in the original).
    """
    # Full kernels
    FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
    FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
    FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
    FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
    FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

    # 3x3 cross kernel
    CROSS_KERNEL_3 = np.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.uint8)

    # 5x5 cross kernel
    CROSS_KERNEL_5 = np.asarray(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)

    # 7x7 cross kernel
    CROSS_KERNEL_7 = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.uint8)

    # 5x5 diamond kernel
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)

    # 7x7 diamond kernel
    DIAMOND_KERNEL_7 = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.uint8)

    def __init__(self, max_depth=100.0,
                 dilation_kernel_far=CROSS_KERNEL_3,
                 dilation_kernel_med=CROSS_KERNEL_5,
                 dilation_kernel_near=CROSS_KERNEL_7,
                 extrapolate=False,
                 blur_type='bilateral'):
        """
        IP-BASIC depth completion object.
        Args:
            max_depth: max depth value for inversion
            dilation_kernel_far: dilation kernel to use for 30 < depths
            dilation_kernel_med: dilation kernel to use for 15 < depths < 30m
            dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15m
            extrapolate:whether to extrapolate by extending depths to top of
                the frame, and applying a 31x31 full kernel dilation
            blur_type:
                'gaussian' - provides lower RMSE (use with sparse data)
                'bilateral' - preserves local structure (use with dense data)
        """
        self.max_depth = max_depth
        self.dilation_kernel_far = dilation_kernel_far
        self.dilation_kernel_med = dilation_kernel_med
        self.dilation_kernel_near = dilation_kernel_near
        self.extrapolate = extrapolate
        self.blur_type = blur_type

    def fill_depth(self, depth):
        """
        Fill depths on depth.
        Args:
            depth: Depth array

        Returns: Filled depth array
        """
        # Convert to float32
        depths_in = np.float32(depth)

        # Calculate bin masks before inversion
        valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
        valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
        valid_pixels_far = (depths_in > 30.0)

        # Invert (and offset)
        s1_inverted_depths = np.copy(depths_in)
        valid_pixels = (s1_inverted_depths > 0.1)
        s1_inverted_depths[valid_pixels] = \
            self.max_depth - s1_inverted_depths[valid_pixels]

        # Multi-scale dilation
        dilated_far = cv2.dilate(
            np.multiply(s1_inverted_depths, valid_pixels_far),
            self.dilation_kernel_far)
        dilated_med = cv2.dilate(
            np.multiply(s1_inverted_depths, valid_pixels_med),
            self.dilation_kernel_med)
        dilated_near = cv2.dilate(
            np.multiply(s1_inverted_depths, valid_pixels_near),
            self.dilation_kernel_near)

        # Find valid pixels for each binned dilation
        valid_pixels_near = (dilated_near > 0.1)
        valid_pixels_med = (dilated_med > 0.1)
        valid_pixels_far = (dilated_far > 0.1)

        # Combine dilated versions, starting farthest to nearest
        s2_dilated_depths = np.copy(s1_inverted_depths)
        s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
        s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
        s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

        # Small hole closure
        s3_closed_depths = cv2.morphologyEx(
            s2_dilated_depths, cv2.MORPH_CLOSE, self.FULL_KERNEL_5)

        # Calculate a top mask
        top_mask = np.ones(depths_in.shape, dtype=np.bool)
        for pixel_col_idx in range(s3_closed_depths.shape[1]):
            pixel_col = s3_closed_depths[:, pixel_col_idx]
            top_pixel_row = np.argmax(pixel_col > 0.1)
            top_mask[0:top_pixel_row, pixel_col_idx] = False

        # Get empty mask
        valid_pixels = (s3_closed_depths > 0.1)
        empty_pixels = ~valid_pixels & top_mask

        # Hole fill
        dilated = cv2.dilate(s3_closed_depths, self.FULL_KERNEL_9)
        s4_dilated_depths = np.copy(s3_closed_depths)
        s4_dilated_depths[empty_pixels] = dilated[empty_pixels]

        # Extend highest pixel to top of image or create top mask
        s5_extended_depths = np.copy(s4_dilated_depths)
        top_mask = np.ones(s4_dilated_depths.shape, dtype=np.bool)

        top_row_pixels = np.argmax(s4_dilated_depths > 0.1, axis=0)
        top_pixel_values = s4_dilated_depths[top_row_pixels,
                                             range(s4_dilated_depths.shape[1])]

        for pixel_col_idx in range(s4_dilated_depths.shape[1]):
            if self.extrapolate:
                s5_extended_depths[0:top_row_pixels[pixel_col_idx],
                pixel_col_idx] = top_pixel_values[pixel_col_idx]
            else:
                # Create top mask
                top_mask[0:top_row_pixels[pixel_col_idx],
                pixel_col_idx] = False

        # Fill large holes with masked dilations
        s6_blurred_depths = np.copy(s5_extended_depths)
        for i in range(6):
            empty_pixels = (s6_blurred_depths < 0.1) & top_mask
            dilated = cv2.dilate(s6_blurred_depths, self.FULL_KERNEL_5)
            s6_blurred_depths[empty_pixels] = dilated[empty_pixels]

        # Median blur
        blurred = cv2.medianBlur(s6_blurred_depths, 5)
        valid_pixels = (s6_blurred_depths > 0.1) & top_mask
        s6_blurred_depths[valid_pixels] = blurred[valid_pixels]

        if self.blur_type == 'gaussian':
            # Gaussian blur
            blurred = cv2.GaussianBlur(s6_blurred_depths, (5, 5), 0)
            valid_pixels = (s6_blurred_depths > 0.1) & top_mask
            s6_blurred_depths[valid_pixels] = blurred[valid_pixels]
        elif self.blur_type == 'bilateral':
            # Bilateral blur
            blurred = cv2.bilateralFilter(s6_blurred_depths, 5, 0.5, 2.0)
            s6_blurred_depths[valid_pixels] = blurred[valid_pixels]

        # Invert (and offset)
        s7_inverted_depths = np.copy(s6_blurred_depths)
        valid_pixels = np.where(s7_inverted_depths > 0.1)
        s7_inverted_depths[valid_pixels] = \
            self.max_depth - s7_inverted_depths[valid_pixels]

        depths_out = s7_inverted_depths

        return depths_out
