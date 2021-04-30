"""
Convenience script for creating separate files for the main different
 configurations based on the defaults.
"""

from rvfn.config.pipeline_config import PipelineConfig
from rvfn.config.training_config import TrainingConfig
from rvfn.config.eval_config import EvalConfig

import json
from pathlib import Path
import argparse


def write_confs(confs: dict, path: Path):
    """
    Write the confs in a dict to json files on disk
    """
    for conf in confs:
        file_path = path / (conf + '.json')
        with open(file_path, 'w') as f:
            json.dump(confs[conf].dict(), f, indent=4)


def make_defaults(path: str):
    """
    Make the most useful variants of the dataset and training configs and
     write them as json files to disk
    Args:
        path: Path to the base directory where the files will be saved
    """
    base_path = Path(path)
    pipeline_base_path = base_path / 'pipeline/'

    # Make default pipeline configs

    # LIDAR
    nusc_lidar = PipelineConfig()

    # LIDAR + RADAR
    nusc_lidar_radar = nusc_lidar.copy()
    nusc_lidar_radar.dataset_config.use_radar = True
    nusc_lidar_radar.model_config.svfeb_config.in_channels = 10

    # LIDAR + RGB
    nusc_lidar_rgb = nusc_lidar.copy()
    nusc_lidar_rgb.dataset_config.use_rgb = True
    nusc_lidar_rgb.model_config.svfeb_config.in_channels = 10

    # LIDAR + RGB-FILLED
    nusc_lidar_rgb_filled = nusc_lidar_rgb.copy()
    nusc_lidar_rgb_filled.dataset_config.fill_type = 'ipbasic'

    # LIDAR + RGB + RADAR
    nusc_lidar_rgb_radar = nusc_lidar_rgb.copy()
    nusc_lidar_rgb_radar.dataset_config.use_radar = True
    nusc_lidar_rgb_radar.model_config.svfeb_config.in_channels = 13

    # LIDAR + RGB-FILLED + RADAR
    nusc_lidar_rgb_filled_radar = nusc_lidar_rgb_radar.copy()
    nusc_lidar_rgb_filled_radar.dataset_config.fill_type = 'ipbasic'

    # RADAR
    nusc_radar = nusc_lidar_radar.copy()
    nusc_radar.dataset_config.use_lidar = False
    nusc_radar.model_config.svfeb_config.in_channels = 9

    # Mini lidar
    # for local usage
    nusc_mini_lidar = nusc_lidar.copy()
    nusc_mini_lidar.dataset_config.root_path = '~/data/sets/nuscenes'
    nusc_mini_lidar.dataset_config.version = 'v1.0-mini'

    pipeline_confs = {
        'mini-lidar': nusc_mini_lidar,
        'lidar': nusc_lidar,
        'lidar-radar': nusc_lidar_radar,
        'lidar-rgb': nusc_lidar_rgb,
        'lidar-rgb-filled': nusc_lidar_rgb_filled,
        'lidar-rgb-radar': nusc_lidar_rgb_radar,
        'lidar-rgb-filled-radar': nusc_lidar_rgb_filled_radar,
        'radar': nusc_mini_lidar
    }

    training_confs = {
        'train': TrainingConfig()
    }

    eval_confs = {
        'eval': EvalConfig()
    }

    write_confs(pipeline_confs, pipeline_base_path)
    write_confs(training_confs, base_path)
    write_confs(eval_confs, base_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Path to metrics output directory",
                        default='./')
    args = parser.parse_args()

    make_defaults(args.out)
