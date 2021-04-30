"""
Provides function for converting the networks predictions to and from nuScenes
 results dicts.
"""
import numpy as np

from pyquaternion import Quaternion


def make_result_dict(sample_token, bbox, offsets, ego_pose, score,
                     detection_name='car'):
    """

    Args:
        sample_token: nuScenes sample token
        bbox: Prediction to convert
        offsets: Dataset offsets
        ego_pose: nuScenes ego pose for sample
        score: prediction score
        detection_name: The class of the predictions

    Returns: result dict

    """
    ego_rotation = Quaternion(ego_pose['rotation'])
    orig_bbox = bbox[:3] - offsets
    translation = ego_rotation.rotate(orig_bbox) + ego_pose['translation']

    rotation = Quaternion(axis=(0.0, 0.0, 1.0),
                          radians=bbox[6]) * ego_rotation
    return {
        'sample_token': sample_token,
        'translation': list(translation),
        'translation_ego': list(ego_pose['translation']),
        # not in nuScenes spec
        'size': list(bbox[3:6]),
        'rotation': list(rotation),
        'velocity': [0, 0],  # not predicted
        'detection_name': detection_name,
        'detection_score': float(score),
        'attribute_name': ''  # not predicted
    }


def get_prediction_from_result_dict(result_dict, ego_pose, offsets):
    global_rotation = Quaternion(result_dict['rotation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    rotation = global_rotation * ego_rotation.inverse
    bbox = np.empty(7)
    bbox_local = np.array(result_dict['translation']) - \
        np.array(ego_pose['translation'])

    bbox[:3] = ego_rotation.inverse.rotate(bbox_local) + offsets
    bbox[3:6] = result_dict['size']
    bbox[6] = rotation.yaw_pitch_roll[0]

    return bbox, result_dict['detection_score']

