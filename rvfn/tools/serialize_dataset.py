"""
Saves NuScenes Fusion dataset to files on disk to be loaded later by the
 training script
"""

from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset, \
    NuscenesDatasetConfig

import pickle
from os import path, getpid
import argparse
import json
import logging

import multiprocessing as mp
from functools import partial
from copy import deepcopy


def write_sample(idx, dataset, base_path='./'):
    sample = dataset[idx]
    file_path = path.join(base_path, sample['sample_token'])
    with open(file_path + '.pickle', 'wb') as f:
        pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)


def create_dataset(config: NuscenesDatasetConfig, out_path: str):
    """
    Writes the fused samples to disk in parallel by spawning multiple processes
    Args:
        config: Dataset config
        out_path: Path to directory where pickle files will be saved
    """
    dataset = NuscenesFusionDataset(config)

    indices = range(len(dataset))

    num_processes = mp.cpu_count()

    with mp.Pool(num_processes) as pool:
        list(pool.imap_unordered(
            partial(write_sample, dataset=dataset, base_path=out_path),
            indices, chunksize=8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="dataset configuration json file",
                        type=str)

    parser.add_argument("--output_dir",
                        help="directory where the output json files are "
                             "written", type=str, default='./')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    json_conf_path = args.config
    with open(json_conf_path) as file:
        conf_dict = json.load(file)

    cfg = NuscenesDatasetConfig(conf_dict)

    create_dataset(cfg, args.output_dir)
