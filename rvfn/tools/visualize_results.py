"""
Given a predictions JSON file, this script visualizes them over the samples in
    bird's eye view
"""

import argparse
from pathlib import Path
import json

from progressbar import progressbar

from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset,\
    find_sample_idx
from rvfn.config.pipeline_config import PipelineConfig
from rvfn.utils.results_dict import get_prediction_from_result_dict


def main():
    parser = argparse.ArgumentParser()

    # We technically only need the dataset config here, but take the pipeline
    #  config for convenience
    parser.add_argument('pipeline_config',
                        help='Pipeline configuration json file')
    parser.add_argument('predictions', help='Predictions JSON file')
    parser.add_argument('out', help='Output path')
    parser.add_argument('--display_conf',
                        help='Display confidence for predicted bboxes',
                        action='store_true')

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = PipelineConfig.from_file(args.pipeline_config)

    dataset = NuscenesFusionDataset(pipeline_cfg.dataset_config, perturb=False)

    with open(args.predictions) as f:
        predictions = json.load(f)

    for token in progressbar(predictions):
        results = predictions[token]

        sample_idx = find_sample_idx(dataset, token)
        if sample_idx is None:
            print('Sample with token {} not found in dataset!'.format(token))
            return

        ego_pose = dataset.get_ego_pose(sample_idx)
        bboxes = []
        scores = []
        for result in results:
            bbox, score = get_prediction_from_result_dict(result, ego_pose,
                                                          dataset.offsets)
            bboxes.append(bbox)
            scores.append(score)

        if args.display_conf:
            fig = dataset.visualize_sample_bev(sample_idx, bboxes, scores)
        else:
            fig = dataset.visualize_sample_bev(sample_idx, bboxes)

        fig.savefig(
            out_path / (str(sample_idx).zfill(5) + "_" + token + '.svg'))


if __name__ == '__main__':
    main()
