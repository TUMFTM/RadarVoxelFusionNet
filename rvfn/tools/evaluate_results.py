"""
Evaluate on results saved in nuScenes results JSON format
"""

import argparse
import json
from pathlib import Path

from progressbar import progressbar
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate

from rvfn.config.pipeline_config import PipelineConfig
from rvfn.config.eval_config import EvalConfig
from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset, \
    find_sample_idx
from rvfn.utils.results_dict import get_prediction_from_result_dict
from rvfn.eval import Metrics, MetricsMean, make_detection_box, \
    get_pr_curves
from rvfn.datasets.common import VehicleBox


def main():
    parser = argparse.ArgumentParser()

    # We technically only need the dataset config here, but take the pipeline
    #  config for convenience
    parser.add_argument('pipeline_config',
                        help='Pipeline configuration json file')
    parser.add_argument("eval_config",
                        help="Evaluation configuration json file")

    parser.add_argument('predictions', help='Predictions JSON file')
    parser.add_argument('out', help='Output path')

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = PipelineConfig.from_file(args.pipeline_config)
    eval_cfg = EvalConfig.from_file(args.eval_config)

    dataset = NuscenesFusionDataset(pipeline_cfg.dataset_config, perturb=False)
    with open(args.predictions) as f:
        predictions = json.load(f)

    all_pred_bboxes = EvalBoxes()
    all_gt_bboxes = EvalBoxes()

    for token in progressbar(predictions):
        results = predictions[token]

        sample_idx = find_sample_idx(dataset, token)
        if sample_idx is None:
            print('Sample with token {} not found in dataset!'.format(token))
            return

        _, gt_bboxes, _ = dataset.get_sample(sample_idx)
        gt_bboxes = [VehicleBox.from_nuscenes_box(box).to_list() for box in
                     gt_bboxes]

        ego_pose = dataset.get_ego_pose(sample_idx)
        bboxes = []
        scores = []
        for result in results:
            bbox, score = get_prediction_from_result_dict(result, ego_pose,
                                                          dataset.offsets)
            bboxes.append(bbox)
            scores.append(score)

        # Convert ground-truth and predicted bboxes to nuScenes
        #  EvalBoxes
        sample_bboxes = [make_detection_box(bboxes[_], scores[_], token)
                         for _ in range(len(bboxes))]

        sample_gt_bboxes = [make_detection_box(gt_bboxes[_], -1.0, token)
                            for _ in range(len(gt_bboxes))]

        all_pred_bboxes.add_boxes(token, sample_bboxes)
        all_gt_bboxes.add_boxes(token, sample_gt_bboxes)

    metrics_list = []
    for dist_th in eval_cfg.distance_thresholds:
        dm = accumulate(all_gt_bboxes, all_pred_bboxes, 'car',
                        center_distance, dist_th)
        metrics_list.append(Metrics(dm))

    metrics_mean = MetricsMean(metrics_list, eval_cfg.distance_thresholds)

    metrics_file_path = out_path / 'metrics.json'
    with open(metrics_file_path, 'w+') as file:
        json.dump(metrics_mean.serialize(), file, indent=4)

    pr_file_path = out_path / 'pr.svg'
    fig = get_pr_curves(metrics_mean)
    fig.savefig(pr_file_path)

    metrics_mean.print()


if __name__ == '__main__':
    main()
