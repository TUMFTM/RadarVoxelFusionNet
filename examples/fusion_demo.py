"""
This notebook contains visualizations for dataset fusion and various pre-processing steps
"""
# %%
# %load_ext autoreload
# %autoreload 2
from os import path

import open3d as o3d
import logging
import numpy as np
from nuscenes.nuscenes import NuScenes

from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset, \
    NuscenesDatasetConfig
from rvfn.config.dataset_config import AugmentationConfig
from rvfn.utils.o3d_utils.lineset import LineSetBox, combine_linesets
from rvfn.utils.o3d_utils.pointcloud import make_o3d_pcd
from rvfn.datasets.augment import augment

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
# %%
"""
nuScenes dataset path and version
"""
nusc_root = path.expanduser('~/data/sets/nuscenes')
nusc_version = 'v1.0-mini'
# %%
"""
Select a sample
"""
sample_idx = 15

# %%
"""
Visualize sample using nuScenes render tools
"""
nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=False)
my_sample = nusc.sample[sample_idx]
nusc.render_sample_data(
    my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=False)
nusc.render_sample_data(my_sample['data']['CAM_FRONT'])

# %%
"""
Common visualizations. Ground-truth bounding boxes, and coordinate frame.
"""
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=5, origin=[0, 0, 0])


def get_boxes_lineset(boxes, color=None):
    if color is None:
        color = [1.0, 0.5, 0.5]
    lineset = [LineSetBox(box.corners().T).get_o3d_lineset() for box in boxes]
    return combine_linesets(lineset, color=color)


# %%
"""
Visualize lidar pointcloud
"""
lidar_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_rgb': False, 'use_radar': False, 'coord_offsets': None,
     'auto_offset': False, 'pointcloud_range': None,
     'lidar_sweeps': 10, 'categories': [['car']]})
lidar_dataset = NuscenesFusionDataset(lidar_config)

pc, boxes, token = lidar_dataset.get_sample(sample_idx)

o3d_pcd, _ = make_o3d_pcd(pc, lidar_dataset.feats, mode='intensity',
                          radar=False)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries([o3d_pcd, mesh_frame, gt_lineset])
# %%
""" 
Visualize augmentations using the lidar dataset
"""
pc, boxes, token = lidar_dataset.get_sample(sample_idx)

pc, boxes = augment(pc, boxes)

o3d_pcd, _ = make_o3d_pcd(pc, lidar_dataset.feats, mode='intensity',
                          radar=False)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries([o3d_pcd, mesh_frame, gt_lineset])

# %%
"""
Visualize fused pointcloud (rgb + radar)
"""
fusion_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_rgb': True, 'use_radar': True, 'coord_offsets': None,
     'auto_offset': False, 'pointcloud_range': None,
     'lidar_sweeps': 10, 'categories': [['car']]})
fusion_dataset = NuscenesFusionDataset(fusion_config)

pc, boxes, token = fusion_dataset.get_sample(sample_idx)

o3d_pcd, radar_lines = make_o3d_pcd(pc, fusion_dataset.feats, mode='rgb',
                                    radar=True)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries(
    [o3d_pcd, radar_lines, mesh_frame, gt_lineset])

# %%
"""
Visualize depth-completed fused pointcloud
"""
dc_fusion_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_rgb': True, 'use_radar': True, 'coord_offsets': None,
     'auto_offset': False, 'pointcloud_range': None,
     'lidar_sweeps': 10, 'categories': [['car']], 'fill_type': 'ipbasic'})
dc_fusion_dataset = NuscenesFusionDataset(dc_fusion_config)

pc, boxes, token = dc_fusion_dataset.get_sample(sample_idx)

o3d_pcd, radar_lines = make_o3d_pcd(pc, dc_fusion_dataset.feats, mode='rgb',
                                    radar=True)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries(
    [o3d_pcd, radar_lines, mesh_frame, gt_lineset])
# %%
"""
Visualize Voxels
"""
fusion_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_rgb': True, 'use_radar': True, 'coord_offsets': None,
     'auto_offset': False, 'pointcloud_range': None,
     'lidar_sweeps': 10, 'categories': [['car']]})
fusion_dataset = NuscenesFusionDataset(fusion_config)
pc, boxes, token = fusion_dataset.get_sample(sample_idx)

o3d_pcd, radar_lines = make_o3d_pcd(pc, fusion_dataset.feats, mode='rgb',
                                    radar=True)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

sample = fusion_dataset[sample_idx]
voxels = sample['voxels']

o3d_voxels = []
for voxel in voxels:
    o3d_voxels.append(LineSetBox.from_voxel(
        voxel, fusion_config.voxel_grid_config.voxel_size).get_o3d_lineset())

o3d_voxels_ls = combine_linesets(o3d_voxels, color=[0.2, 0.2, 1.0])
o3d.visualization.draw_geometries(
    [o3d_pcd, radar_lines, mesh_frame, gt_lineset, o3d_voxels_ls])

# %%
"""
Visualize pointcloud with offset and range limit
"""
fusion_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_rgb': True, 'use_radar': True,
     'pointcloud_range': (
         (1.0, 50),
         (-20.0, 20.0),
         (-1.0, 3.0)
     ), 'auto_offset': True,
     'lidar_sweeps': 7, 'categories': [['car']]})
fusion_dataset = NuscenesFusionDataset(fusion_config)

pc, boxes, token = fusion_dataset.get_sample(sample_idx)

o3d_pcd, radar_lines = make_o3d_pcd(pc, fusion_dataset.feats, mode='rgb',
                                    radar=True)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries(
    [o3d_pcd, radar_lines, mesh_frame, gt_lineset])
# %%
"""
Visualize pointcloud with offset and range limit with augmentations
"""
aug_cfg = AugmentationConfig({'box_rotation': np.pi / 10,
                              'box_translation': (1.0, 1.0, 0.3),
                              'global_rotation': 0.0,
                              'global_scale': 0.05})
fusion_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_rgb': True, 'use_radar': True,
     'pointcloud_range': (
         (1.0, 50),
         (-20.0, 20.0),
         (-1.0, 3.0)
     ), 'auto_offset': True,
     'lidar_sweeps': 7, 'categories': [['car']],
     'augmentation_config': aug_cfg})
fusion_dataset = NuscenesFusionDataset(fusion_config)

pc, boxes, token = fusion_dataset.get_sample(sample_idx)
pc, boxes = augment(pc, boxes)

o3d_pcd, radar_lines = make_o3d_pcd(pc, fusion_dataset.feats, mode='rgb',
                                    radar=True)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries(
    [o3d_pcd, radar_lines, mesh_frame, gt_lineset])


# %%
"""
Visualize radar-only pointcloud
"""
radar_config = fusion_config = NuscenesDatasetConfig(
    {'root_path': nusc_root, 'version': nusc_version,
     'use_lidar': False, 'use_radar': True,
     'pointcloud_range': (
         (1.0, 50),
         (-20.0, 20.0),
         (-1.0, 3.0)
     ), 'auto_offset': True,
     'categories': [['car']]})

radar_dataset = NuscenesFusionDataset(radar_config)

pc, boxes, token = radar_dataset.get_sample(sample_idx)

o3d_pcd, radar_lines = make_o3d_pcd(pc, radar_dataset.feats, mode=None,
                                    radar=True)
gt_lineset = get_boxes_lineset(boxes, [1.0, 0.5, 0.5])

o3d.visualization.draw_geometries(
    [o3d_pcd, radar_lines, mesh_frame, gt_lineset])
