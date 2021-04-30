from rvfn.config.config import Config

import numpy as np


class VoxelGridConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # In x, y, and z direction in meters.
            'voxel_size': (0.2, 0.2, 0.4),
            'max_points_per_voxel': 40,
        }


class AugmentationConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # Rotation limit for boxes and points inside them.
            # Rotation will be uniformly between -box_rotation and
            #  +box_rotation
            'box_rotation': np.pi / 10,

            # Translation standard deviation in x, y, and z axes
            'box_translation': (1.0, 1.0, 0.3),

            # Rotation limit for the point cloud
            # Rotation will be uniformly between -global_rotation and
            #  +global_rotation
            # This is set to a far lower value than VoxelNet because we are
            #  only using a single camera's FOV. Try increasing it to pi/4
            #  once entire point cloud is used.
            'global_rotation': np.pi / 10,

            # Scaling limit for the point cloud
            # Scaling will be uniformly between 1 - global_scale and
            #  1 + global_scale
            'global_scale': 0.05
        }


class NuscenesDatasetConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'root_path': '/nuscenes',
            'version': 'v1.0-trainval',
            'cam_name': 'CAM_FRONT',
            'radar_name': 'RADAR_FRONT',
            'lidar_sweeps': 3,

            # At least one of use_lidar or use_radar must be True
            'use_lidar': True,
            'use_radar': False,

            # If use_lidar==True, fuse them with RGB info from camera
            'use_rgb': False,

            'radar_sweeps': 3,
            'min_dist': 1.0,
            'fill_type': None,
            'img_size': (1600, 900),

            # In x, y, and z direction in meters.
            # Will remove points and bboxes that are outside this boundary.
            # Can also be None to indicate no point removal.
            'pointcloud_range': (
                (1.0, 50),
                (-20.0, 20.0),
                (-1.0, 3.0)
            ),

            # If True will offset point and bbox coordinates such that all
            #  coordinates become non-negative.
            # Only works if 'pointcloud_range' is not None. This is to avoid
            #  variable offsets per sample.
            'auto_offset': True,

            # How much to offset the coordinates of points and bboxes in each
            #  axis. Will be ignored if 'auto_offset' is True.
            'coord_offsets': (0, +20.0, +1),

            # Maximum number of lidar points in the pointcloud. If there are
            #  more points, a random subsample of them will be selected to fit
            #  within this limit.
            # To maximize the number of radar points, this limit is applied
            #  before adding the radar points.
            'max_points': 40000,

            # Categories of objects.
            # Must be a list of lists, where each sublist indicates categories
            #  that will be mapped to the same label.
            # The label for each sublist will be the index of the sublist.
            # Any category not in this list will be counted as background.
            'categories': [['car']],
            'voxel_grid_config': VoxelGridConfig(),
            'augmentation_config': AugmentationConfig()
        }
