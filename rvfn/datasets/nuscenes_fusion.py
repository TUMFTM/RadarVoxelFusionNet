import numpy as np
from os import path
import json
import logging

import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import PIL.Image as pil

from rvfn.utils.nuscenes_utils import \
    load_and_transform_lidar_to_cam, Category, get_bbox_anns_in_fov
from rvfn.datasets.common import PointCloud, VoxelGrid, \
    bboxes_to_arrays, coords_in_range, VehicleBox
from rvfn.utils.pointcloud import project_points, back_project_img
from rvfn.utils.depth_completion import KnnDepth, MaskedConvDepth, \
    ModifiedIpBasicDepth
from rvfn.utils.visualization_helpers import \
    render_ego_centric_map, render_annotation, get_colors
from rvfn.datasets.augment import augment
from rvfn.config.dataset_config import NuscenesDatasetConfig

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


class NuscenesFusionDataset(Dataset):

    def __init__(self, config: NuscenesDatasetConfig = None,
                 samples_filepath: str = None, perturb=False):
        """
        Args:
            config: Dataset config
            samples_filepath: Path to a file containing a json array of sample
             tokens. Entire dataset will be used if None.
            perturb: Will randomly perturb the samples as a form of on-the-fly
             augmentation if True
        """
        self.config = config if config is not None else NuscenesDatasetConfig()
        self.perturb = perturb

        # Dictionary indicating the feature indices in the output pointcloud
        self.feats = {
            'x': 0, 'y': 1, 'z': 2
        }
        if self.config.use_lidar:
            self.feats['intensity'] = 3
            if self.config.use_rgb:
                self.feats['r'] = 4
                self.feats['g'] = 5
                self.feats['b'] = 6
                if self.config.use_radar:
                    self.feats['rcs'] = 7
                    self.feats['vx'] = 8
                    self.feats['vy'] = 9
            elif self.config.use_radar:
                self.feats['rcs'] = 4
                self.feats['vx'] = 5
                self.feats['vy'] = 6
        elif self.config.use_radar:
            self.feats['rcs'] = 3
            self.feats['vx'] = 4
            self.feats['vy'] = 5
        else:
            raise AttributeError('At least one of use_lidar or use_radar '
                                 'must be True')

        logging.info('Loading dataset...')
        nusc_root = path.expanduser(self.config.root_path)
        self.nusc = NuScenes(version=self.config.version,
                             dataroot=nusc_root, verbose=False)

        if samples_filepath is None:
            self.samples = self.nusc.sample
        else:
            with open(samples_filepath, 'r') as f:
                sample_tokens = json.load(f)
            self.samples = [self.nusc.get('sample', token) for token in
                            sample_tokens]

        self.offsets = np.array(self.config.coord_offsets) \
            if self.config.coord_offsets is not None else None

        if self.config.auto_offset:
            self.offsets = np.array((0, 0, 0))
            ranges = self.config.pointcloud_range
            assert ranges is not None, 'Auto-offset only works if a ' \
                                       'pointcloud-range is specified'

            for ax in range(3):
                self.offsets[ax] = -ranges[ax][0] if ranges[ax][0] < 0 else 0

            logging.info('Coord offset: {}'.format(self.offsets))

        self.categories = [[Category(cat) for cat in label] for
                           label in self.config.categories]

        logging.info('Dataset loaded. Num samples: {}'.format(len(self)))

    def get_sample(self, idx):
        """
        Returns a fused pointcloud and associated annotations for a sample.
        Args:
            idx: sample index

        Returns:
            - A PointCloud where the dimensions are in the order specified
                in self.feats
            - A list of Boxes denoting annotations
            - Token for nuScenes sample

        """
        sample = self.samples[idx]

        if self.config.use_lidar:
            pc = get_camera_fused_pointcloud(self.nusc, sample,
                                             self.config.cam_name,
                                             self.config.min_dist,
                                             self.config.lidar_sweeps,
                                             self.config.use_rgb,
                                             self.config.fill_type)

            # Randomly subsample the pointcloud if there are more points than
            #  maximum allowed.
            if pc.nbr_points() > self.config.max_points:
                pc.subsample(self.config.max_points / pc.nbr_points())

        if self.config.use_radar:
            radar_pc_raw = get_radar_pointcloud(self.nusc, sample,
                                                self.config.radar_name,
                                                self.config.radar_sweeps)

            radar_xyz = radar_pc_raw.points[:3, :]
            radar_rcs = radar_pc_raw.points[5:6, :]
            radar_vel_comp = radar_pc_raw.points[8:10, :]

            radar_pc = PointCloud(radar_xyz)

            if self.config.use_lidar:
                # Zeros for intensity + R, G, and B if applicable
                extra_dims = 4 if self.config.use_rgb else 1
                radar_pc.add_dims(np.zeros((extra_dims,
                                            radar_pc.nbr_points())))
            radar_pc.add_dims(radar_rcs)
            radar_pc.add_dims(radar_vel_comp)

            if self.config.use_lidar:
                pc.add_points(radar_pc.points, auto_pad=True)
            else:
                pc = radar_pc

        # Desired categories of objects. This should be flattened because
        #  sub-lists indicate labeling groups, and we don't care about that
        #  here.
        flatten_categories = [cat for sublist in self.categories for cat
                              in sublist]
        # Get ground-truth bboxes in camera FOV
        boxes = get_bbox_anns_in_fov(self.nusc, sample,
                                     self.config.cam_name,
                                     self.config.img_size,
                                     flatten_categories)

        # Random augmentation
        if self.perturb:
            pc, boxes = augment(pc, boxes, self.config.augmentation_config)

        # Remove points and boxes that are outside config.pointcloud_range
        ranges = self.config.pointcloud_range
        if ranges is not None:
            mask = np.ones(pc.nbr_points())
            for ax in range(3):
                mask = np.logical_and(mask,
                                      ranges[ax][0] < pc.points[ax, :])
                mask = np.logical_and(mask,
                                      pc.points[ax, :] < ranges[ax][1])
            pc.points = pc.points[:, mask]

            boxes = [box for box in boxes if
                     coords_in_range(box.center, ranges)]

        # Translate coordinates by offset
        if self.offsets is not None:
            pc.translate(self.offsets)
            for box in boxes:
                box.translate(self.offsets)

        return pc, boxes, self.samples[idx]['token']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc, boxes, token = self.get_sample(idx)

        # Assigned label to each box based on conf.categories
        labels = []
        for bbox in boxes:
            for c_id, category_group in enumerate(self.categories):
                if Category(bbox.name) in category_group:
                    labels.append(c_id)
                    break
                # boxes that are not in any of the desired categories are
                #  already filtered out in get_sample()

        labels = torch.as_tensor(labels)
        bboxes7d = torch.as_tensor(bboxes_to_arrays(boxes))

        voxel_grid = VoxelGrid(pc, self.config.voxel_grid_config)
        voxels = torch.as_tensor(voxel_grid.voxels)
        features = torch.as_tensor(voxel_grid.features)

        return {'voxels': voxels,
                'features': features,
                'bboxes': bboxes7d,
                'labels': labels,
                'sample_token': token}

    def visualize_sample_bev(self, idx, predicted_bboxes=None,
                             predicted_scores=None):
        """
        Visualize a sample in bird's eye view
        Args:
            idx: Sample index
            predicted_bboxes: List of predicted bounding boxes to draw
            predicted_scores: List of scores for predicted bboxes

        Returns: matplotlib figure

        """
        pc, bboxes, token = self.get_sample(idx)

        # Move point cloud back to their original coordinates
        pc.translate(-self.offsets)

        # Get scene map and ego pose
        scene = self.nusc.get('scene', self.samples[idx]['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        scene_map = self.nusc.get('map', log['map_token'])
        ref_data = self.nusc.get('sample_data',
                                 self.samples[idx]['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', ref_data['ego_pose_token'])

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)

        axes_limits = [5, 55, 20, 20]

        try:
            render_ego_centric_map(scene_map=scene_map,
                                   ego_pose=pose,
                                   axes_limits=axes_limits,
                                   ax=ax)
        except IndexError:
            # This is required since some nuScenes maps are corrupted
            pass

        for box in bboxes:
            box.translate(-self.offsets)
            render_annotation(box, color=get_colors('primary')[2], ax=ax,
                              linewidth=0.7)

        if predicted_bboxes is not None:
            for idx, box in enumerate(predicted_bboxes):
                box = VehicleBox.from_list(box)
                box.translate(-self.offsets)
                # continue
                render_annotation(box, color=get_colors('accent')[1], ax=ax,
                                  linewidth=4.0)
                if predicted_scores is not None:
                    ax.text(box.xyz[0], box.xyz[1],
                            round(predicted_scores[idx], 4),
                            fontsize=9)

        if self.config.use_radar:
            radar_mask = np.zeros(pc.nbr_points(), dtype=bool)
            radar_mask = np.logical_or(radar_mask,
                                       pc.points[self.feats['vx'], :] != 0)
            radar_mask = np.logical_or(radar_mask,
                                       pc.points[self.feats['vy'], :] != 0)

            radar_points = pc.points[:, radar_mask]
            lidar_points = pc.points[:, ~radar_mask]

            ax.quiver(radar_points[0, :], radar_points[1, :],
                      radar_points[self.feats['vx'], :],
                      radar_points[self.feats['vy'], :],
                      color=get_colors('accent')[2],
                      scale=200.0, width=0.005, headwidth=3.0,
                      headlength=3.0, headaxislength=3.0, minlength=0.5)

            ax.scatter(radar_points[0, :], radar_points[1, :],
                       color=get_colors('accent')[2], s=4.0)
        else:
            lidar_points = pc.points

        # Show point clouds
        ax.scatter(lidar_points[0, :], lidar_points[1, :],
                   color=get_colors('primary')[0], s=2.0)

        # Show ego vehicle
        ax.plot(0, 0, 'x', color=get_colors('primary')[2])

        # Limit visible range
        ax.set_xlim(-axes_limits[0], axes_limits[1])
        ax.set_ylim(-axes_limits[2], axes_limits[3])

        # Format axes
        ax.axis('off')
        ax.set_aspect('equal')

        plt.close(fig)
        return fig

    def get_ego_pose(self, idx):
        lidar_token = self.samples[idx]['data']['LIDAR_TOP']
        lidar = self.nusc.get('sample_data', lidar_token)

        return self.nusc.get('ego_pose', lidar['ego_pose_token'])


def find_sample_idx(dataset, token):
    """
    Find sample idx from token in dataset
    """
    for idx, sample in enumerate(dataset.samples):
        if sample['token'] == token:
            return idx

    return None


def make_depth_map(width, height, points, depths):
    """
    Make a 2D array of depths
    Args:
        width: Array width
        height: Array height
        points: 2xn array of points, where the first dimension is u and the
            second v.
        depths: Array of size n, containing the depth of each of the points.

    Returns: 2D depth map
    """
    depth_map = np.zeros((height, width))
    for i in range(len(depths)):
        depth_map[int(points[1][i])][int(points[0][i])] = depths[i]

    return depth_map


def get_camera_fused_pointcloud(nusc: NuScenes, sample,
                                cam_name='CAM_FRONT',
                                min_dist=1.0,
                                nsweeps=5,
                                fuse_camera=True,
                                fill_method=None):
    """
    Loads points from lidar pointcloud, finds the points that are within a
    camera's FOV, and the color of the corresponding points in the camera's
    image. Optionally will estimate depths for more camera pixels based on the
    lidar points and will add them to the returned pointcloud.
    Args:
        nusc: Nuscenes instance
        sample: Nuscenes sample
        cam_name: Name of the camera
        min_dist: Distance in meters below which points will be ignored
        nsweeps: Number of lidar sweeps to use
        fuse_camera: Weather to add color dimension to the pointcloud
        fill_method:
            - None: No depth completion performed
            - 'ipbasic': depth completion based on ip-basic
            - 'knn': depth completion based on KNN regression.
            - 'maskconv': depth completion with masked convolutions.
            This parameter will be ignored if fuse_camera is false

    Returns: PointCloud containing only the points that are within the
     camera's field of view transformed to ego vehicle reference frame.

    Some of the logic in this function is taken from map_pointcloud_to_image
    function in the Nuscenes dev-kit:
    https://github.com/nutonomy/nuscenes-devkit/python-sdk/nuscenes/
        nuscenes.py#L532
    """

    pc = PointCloud(load_and_transform_lidar_to_cam(nusc, sample, cam_name,
                                                    nsweeps).points)

    camera_token = sample['data'][cam_name]
    cam = nusc.get('sample_data', camera_token)
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])

    im = pil.open(path.join(nusc.dataroot, cam['filename']))

    # Nuscenes pointcloud point dimensions start with x, y, and z coordinates.
    depths = pc.points[2, :]

    p_points, mask = project_points(pc.points[:3, :],
                                    np.array(cs_record['camera_intrinsic']),
                                    im.size, min_dist)

    depths = depths[mask]
    p_points = p_points[:, mask]

    pc.points = pc.points[:, mask]

    pc = PointCloud(pc.points)

    if fuse_camera:
        # Get colors of the projected points from the RGB image
        colors = []
        for p in list(zip(p_points[0], p_points[1])):
            colors.append(im.getpixel(p))

        if fill_method is not None:
            depth_map = make_depth_map(im.size[0], im.size[1], p_points,
                                       depths)
            # Create a mask of non-zero values in the old depth map
            # (corresponding to the real lidar points)
            real_lidar_mask = np.ma.masked_not_equal(depth_map, 0)

            if fill_method == 'ipbasic':
                filler = ModifiedIpBasicDepth()
                depth_map = filler.fill_depth(depth_map)
            elif fill_method == 'knn':
                filler = KnnDepth(depth_map)
                depth_map = filler.fill_depth()
            elif fill_method == 'maskconv':
                filler = MaskedConvDepth(depth_map, real_lidar_mask)
                depth_map = filler.fill_depth()
            else:
                print("Fill method not recognized")
                return

            # Set the depth on pixels corresponding to real lidar points to
            #  zero, so that we can filter them out, and add the originals
            #  later. This is to retain the original values in other
            #  dimensions.
            np.putmask(depth_map, real_lidar_mask, 0)

            filled_points, filled_colors = back_project_img(
                depth_map, np.array(cs_record['camera_intrinsic']), im)

            filled_points = np.array(filled_points)
            filled_colors = np.array(filled_colors)

            dist_mask = np.ones(len(filled_points), dtype=bool)
            dist_mask = np.logical_and(dist_mask,
                                       filled_points[:, 2] > min_dist)
            filled_points = filled_points[dist_mask]
            filled_colors = filled_colors[dist_mask]

            filled_points = filled_points.T
            filled_points = np.vstack(
                (filled_points, np.zeros((1, filled_points.shape[1]))))
            pc.points = np.append(pc.points, filled_points, axis=1)

            colors = np.append(colors, filled_colors, axis=0)

        pc.add_dims(np.array(colors).T)

    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    return pc


def get_radar_pointcloud(nusc: NuScenes, sample,
                         radar_name='RADAR_FRONT',
                         nsweeps=5):
    """
    Load radar pointcloud and transform to ego vehicle reference frame
    Args:
        nusc: Nuscenes instance
        sample: Nuscenes sample
        radar_name: Name of the radar sensor
        nsweeps: Number of radar sweeps

    Returns: PointCloud with dimensions in the following order:
        x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid
        ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    """
    sd_token = sample['data'][radar_name]
    sd_record = nusc.get('sample_data', sd_token)
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = radar_name

    pc, _ = RadarPointCloud.from_file_multisweep(nusc,
                                                 sample_rec,
                                                 chan, ref_chan,
                                                 nsweeps=nsweeps)

    pc = PointCloud(pc.points)

    radar_cs_record = nusc.get('calibrated_sensor',
                               sd_record['calibrated_sensor_token'])
    pc.rotate(Quaternion(radar_cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(radar_cs_record['translation']))

    return pc


def get_radar_pointcloud_in_fov(nusc: NuScenes, sample,
                                radar_name='RADAR_FRONT',
                                cam_name='CAM_FRONT',
                                nsweeps=5):
    radar_token = sample['data'][radar_name]
    camera_token = sample['data'][cam_name]

    cam = nusc.get('sample_data', camera_token)
    radar = nusc.get('sample_data', radar_token)

    pc = get_radar_pointcloud(nusc, sample, radar_name, nsweeps)

    # transform to the global frame.
    poserecord = nusc.get('ego_pose', radar['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # transform into the ego vehicle frame for the timestamp of
    # the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    intrinsics = np.array(cs_record['camera_intrinsic'])

    pass
