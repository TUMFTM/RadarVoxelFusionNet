"""
Creates Json files containing sample tokens for train and validation splits.
Selects scenes for each split and creates a subset that contains only samples
 that have a minimum number of objects of desired categories within a desired
 range.
On "random" mode splits a nuScenes dataset by randomly selecting a configurable
 number of scenes for validation.
On "rain_night" mode selects only rain or night labeled scenes according to the
offical nuScenes trainval split.
On "official" mode it uses the official nuScenes trainval split.
"""
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from os import path
import argparse
from typing import Tuple, List
import random
import json

from rvfn.config.pipeline_config import PipelineConfig
from rvfn.utils.nuscenes_utils import Category, get_bbox_anns_in_fov
from rvfn.datasets.common import coords_in_range


def make_subset(conf: PipelineConfig,
                min_objects: int, split_select_mode: str = "official",
                val_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    """
    Splits the dataset to train and val by selecting val_ratio of scenes at
     random for the validation set. Removes the samples which do not contain
     at least min_objects within the parameters of conf.

    Args:
        conf: Besides the path and version of the dataset, this is used to
            limit the area and object categories we look for within the samples
            using conf.pointcloud_range and conf.categories.
        min_objects: Minimum number of objects of relevant categories that
            must be present in a sample to be included.
        split_select_mode: Either official, random, or rain_night.
        val_ratio: Ratio of dataset scenes used for validation set for
            random split.

    Returns: List of sample tokens for training set, and list of sample tokens
        for validation.

    """
    categories = [[Category(cat) for cat in label] for label in
                  conf.categories]
    categories = [cat for sublist in categories for cat in sublist]

    nusc_root = path.expanduser(conf.root_path)
    nusc = NuScenes(version=conf.version, dataroot=nusc_root, verbose=True)

    num_scenes = len(nusc.scene)

    if split_select_mode == 'random':
        val_scenes_inds = random.sample(range(num_scenes),
                                        int(val_ratio * num_scenes))
        val_scenes = [nusc.scene[idx]['token'] for idx in val_scenes_inds]
        train_scenes = [nusc.scene[idx]['token'] for idx in range(num_scenes)
                        if idx not in val_scenes_inds]

    elif split_select_mode == 'rain_night':
        splits = create_splits_scenes()
        val_scenes = []
        train_scenes = []
        rain_night_cnt = 0
        scene_attributes = ["rain", "night"]

        for scene in nusc.scene:
            scene_description = scene['description'].lower()
            if any(x in scene_description for x in scene_attributes):
                rain_night_cnt += 1
                if scene['name'] in splits['val']:
                    val_scenes.append(scene['token'])
                elif scene['name'] in splits['train']:
                    train_scenes.append(scene['token'])
                else:
                    raise ValueError(f"Could not associate scene with name: {scene['name']}")

        print(f'Number of rain or night scenes: {rain_night_cnt}')

    elif split_select_mode == 'official':
        # Use official split
        splits = create_splits_scenes()

        val_scenes = [scene['token'] for scene in nusc.scene if
                      scene['name'] in splits['val']]
        train_scenes = [scene['token'] for scene in nusc.scene if
                        scene['name'] in splits['train']]

    else:
        raise NotImplementedError(f"Split {split_select_mode} is not recognized. \
             Choose either: official, random, or rain_night")

    val_samples = []
    train_samples = []

    print("Filtering samples")
    for count, sample in enumerate(nusc.sample, 0):
        # Get ground-truth bboxes in camera FOV
        boxes = get_bbox_anns_in_fov(nusc, sample, conf.cam_name,
                                     conf.img_size, categories)

        if conf.pointcloud_range is not None:
            boxes = [box for box in boxes if
                     coords_in_range(box.center, conf.pointcloud_range)]

        if len(boxes) >= min_objects:
            if sample['scene_token'] in val_scenes:
                val_samples.append(sample['token'])
            elif sample['scene_token'] in train_scenes:
                train_samples.append(sample['token'])

        if count % 2000 == 0:
            print(f"Processing sample {count}.")
    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='"official", "rain_night" or "random" '
                                            'split selection',
                                       dest='command')

    parser_nuscenes = subparsers.add_parser('official',
                                            help='Use the official nuScenes '
                                                 'split.')

    parser_rain_night = subparsers.add_parser('rain_night',
                                              help='Select only rain and nights scenes of \
                                             the official nuScenes split.')

    parser_random = subparsers.add_parser('random',
                                          help='Randomly select the scenes for'
                                               ' train and validation sets.')

    parser_random.add_argument("val_ratio", help="portion of dataset scenes to use for the \
         random validation set", type=float, default=None)

    parser.add_argument("--config", help="dataset configuration json file",
                        type=str)

    parser.add_argument("--min_objects",
                        help="minimum number of objects in each sample",
                        type=int, default=0)

    parser.add_argument("--output_dir",
                        help="directory where the output json files are "
                             "written", type=str, default='./')

    args = parser.parse_args()

    json_conf_path = args.config
    with open(json_conf_path) as file:
        conf_dict = json.load(file)

    cfg = PipelineConfig(conf_dict)
    cfg = cfg["dataset_config"]

    if args.command == "random":
        train, val = make_subset(cfg, args.min_objects, args.command, args.val_ratio)
    else:
        train, val = make_subset(cfg, args.min_objects, args.command)

    print("Training samples:", len(train))
    print("Validation samples:", len(val))

    print("Saving to file...")
    with open(args.output_dir + 'train_samples.json', 'w+') as file:
        json.dump(train, file)

    with open(args.output_dir + 'val_samples.json', 'w+') as file:
        json.dump(val, file)
    print("Done.")


if __name__ == '__main__':
    main()
