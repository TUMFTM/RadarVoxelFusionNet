"""
Helper functions for use of Nuscenes dataset
"""

import numpy as np
from typing import List
import open3d as o3d
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import PointCloud, LidarPointCloud, Box
from os import path
import PIL.Image as pil
import copy
from rvfn.utils.pointcloud import project_points


class Category:
    # Nuscenes detection challenge category map according to:
    # https://www.nuscenes.org/object-detection
    CATEGORY_MAP = {
        'car': ['vehicle.car'],
        'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
        'bicycle': ['vehicle.bicycle'],
        'pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child',
                       'human.pedestrian.construction_worker',
                       'human.pedestrian.police_officer'],
        'construction_vehicle': ['vehicle.construction'],
        'motorcycle': ['vehicle.motorcycle'],
        'truck': ['vehicle.truck'],
        'trailer': ['vehicle.trailer'],
        'traffic_cone': ['movable_object.trafficcone'],
        'barrier': ['movable_object.barrier'],
        'ignore': ['animal', 'human.pedestrian.personal_mobility',
                   'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
                   'movable_object.debris', 'movable_object.pushable_pullable',
                   'static_object.bicycle_rack', 'vehicle.emergency.ambulance',
                   'vehicle.emergency.police']
    }

    reverse_map = {}
    for key in CATEGORY_MAP:
        for value in CATEGORY_MAP[key]:
            reverse_map[value] = key

    def __init__(self, name: str):
        """
        Create a category either from the direct name, or from a nuscenes
         sub-category name.
        """
        if name in self.reverse_map:
            name = self.reverse_map[name]
        assert name in self.CATEGORY_MAP, \
            'Category ' + name + ' not recognized'

        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def nuscenes_subcategories(self) -> List[str]:
        return self.CATEGORY_MAP[self.name]


def get_ann_records(nusc: NuScenes, sample, categories: List[Category] = None):
    """
    Get annotation records from a Nuscenes sample
    Args:
        nusc: Nuscenes instance
        sample: Nuscenes sample
        categories: Only return annotations of these categories, returns all
         annotations if equal to None.

    Returns: List of annotation records

    """
    sample_anns_tokens = sample['anns']
    anns = [nusc.get('sample_annotation', ann) for ann in sample_anns_tokens]
    if categories is not None:
        anns = [ann for ann in anns if
                Category(ann['category_name']) in categories]

    return anns


def get_bbox_anns(nusc: NuScenes, sd_token: str,
                  categories: List[Category] = None) -> List[Box]:
    """
    Get a list of bbox annotations for a sensor sample in vehicle coordinates
    Args:
        nusc: Nuscenes instance
        sd_token: Nuscenes sample_data token
        categories: Only return annotations of these categories, returns all
         annotations if equal to None.

    Returns: List of Nusc.Boxes with positions relative to the ego pose.

    """
    sd_record = nusc.get('sample_data', sd_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    boxes = nusc.get_boxes(sd_token)

    for box in boxes:
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

    if categories is not None:
        boxes = [box for box in boxes if Category(box.name) in categories]

    return boxes


def get_bbox_anns_in_fov(nusc: NuScenes, sample, cam_name, img_size,
                         categories: List[Category] = None) -> List[Box]:
    """
    Get a list of bbox annotations for sample that are within a camera's FOV.
    Args:
        nusc: Nuscenes instance
        sample: Nuscenes sample
        cam_name: Camera name (e.g. 'CAM_FRONT')
        img_size: Camera image size (width, height)
        categories: Only return annotations of these categories, returns all
         annotations if equal to None.

    Returns: List of Nusc.Boxes with positions relative to the ego pose.
    """
    sd_token = sample['data'][cam_name]
    sd_record = nusc.get('sample_data', sd_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    intrinsics = np.array(cs_record['camera_intrinsic'])

    bboxes = np.array(
        get_bbox_anns(nusc, sd_token, categories))

    if bboxes.size > 0:
        boxes_copy = copy.deepcopy(bboxes)
        centers = []

        for box in boxes_copy:
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            centers.append(box.center)
        centers = np.array(centers).T

        _, mask = project_points(centers, intrinsics, img_size)

        bboxes = bboxes[mask]

    return bboxes


def get_sensor_transformations(nusc: NuScenes, sample, sensor='CAM_FRONT'):
    """
    Returns sensor transformation and calibration information
    Args:
        nusc: Nuscenes instance
        sample: Nuscenes sample
        sensor: Sensor name

    Returns: 1x3 translation vector,
             1x4 rotation quaternion,
             3x3 calibration matrix if sensor is a camera. None otherwise.
    """

    sd_token = sample['data'][sensor]
    sd_record = nusc.get('sample_data', sd_token)
    sensor_record = nusc.get('calibrated_sensor',
                             sd_record['calibrated_sensor_token'])

    return sensor_record['translation'], sensor_record['rotation'], \
           sensor_record['camera_intrinsic']


def o3d_pointcloud(pointcloud: PointCloud,
                   colors=None) -> o3d.geometry.PointCloud:
    """
    Convert Nuscenes pointcloud to Open3D pointcloud. Will only retain xyz
    coordinates and discard all other information.
    Args:
        pointcloud: Nuscenes pointcloud object
        colors: List containing the RGB color for each of the points in
        pointcloud in order. If length==1, the one color is given to all points

    Returns: Open3D pointcloud object.
    """

    num_points = pointcloud.nbr_points()
    # The first 3 dimensions of Nuscenes pointclouds are xyz coordinates
    points = [pointcloud.points[:3, i] for i in range(num_points)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if len(colors) == 1:
            colors = [colors[0] for _ in range(num_points)]

        assert len(colors) == len(points)

        # o3d colors are in [0, 1] range
        colors = [[c / 255 for c in color] for color in colors]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def get_projected_depth_map(nusc_exp: NuScenesExplorer, sample,
                            pointsensor_name='LIDAR_TOP',
                            cam_name='CAM_FRONT',
                            min_dist=1.0):
    pointsensor_token = sample['data'][pointsensor_name]
    camera_token = sample['data'][cam_name]
    points, colors, im = nusc_exp.map_pointcloud_to_image(pointsensor_token,
                                                          camera_token,
                                                          min_dist)

    depth_map = np.zeros((im.size[1], im.size[0]))
    for i in range(len(colors)):
        depth_map[int(points[1][i])][int(points[0][i])] = colors[i]

    return depth_map


def load_and_transform_lidar_to_cam(nusc: NuScenes, sample,
                                    cam_name='CAM_FRONT',
                                    nsweeps=5):
    """
    Load LIDAR points and transform to camera coordinates
    Args:
        nusc: Nuscenss instance
        sample: Nuscenes sample
        cam_name: Camera name to transform into
        nsweeps: Number of LIDAR sweeps to use

    Returns: LidarPointCloud in the camera coordinates system

    """
    pointsensor_token = sample['data']['LIDAR_TOP']
    camera_token = sample['data'][cam_name]

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    # pcl_path = path.join(nusc.dataroot, pointsensor['filename'])

    chan = pointsensor['channel']
    ref_chan = 'LIDAR_TOP'

    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample, chan, ref_chan,
                                                 nsweeps=nsweeps)

    # First step: transform the point-cloud to the ego vehicle frame for the
    # timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform into the ego vehicle frame for the timestamp of
    # the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    return pc


def transform_cam_to_lidar(nusc: NuScenes, sample, pc,
                           cam_name='CAM_FRONT'):
    """
    Transform a PointCloud from camera coordinates to LIDAR coordinates
    Args:
        nusc: Nuscenss instance
        sample: Nuscenes sample
        pc: Nuscenes.PointCloud
        cam_name: Camera name from the coordinates of which we transform

    Returns: PointCloud in LIDAR coordinates system
    """
    pointsensor_token = sample['data']['LIDAR_TOP']
    camera_token = sample['data'][cam_name]

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)

    # First step: transform to ego vehicle coordinates.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform global coordinates.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform to ego vehicle at timestamp of the sweep.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform to LIDAR frame.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    return pc


if __name__ == '__main__':
    nusc_root = path.expanduser('~/data/sets/nuscenes')
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=True)
    sample_token = 'c59e60042a8643e899008da2e446acca'  # sample with vehicle close by

    my_sample = nusc.get('sample', sample_token)

    cam_front_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    img_path = path.join(nusc_root, cam_front_data['filename'])
    img = pil.open(img_path).convert('RGB')

    pc, colors = get_camera_fused_pointcloud(nusc, my_sample, nsweeps=3,
                                             fill_method='knn')

    # pcd = o3d_pointcloud(pc, colors)
    #
    # o3d.visualization.draw_geometries([pcd])
