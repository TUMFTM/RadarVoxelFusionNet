import numpy as np
from typing import List
import json
import torch
import logging
import argparse
from copy import deepcopy

from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.data_classes import DetectionMetricData, \
    DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import calc_ap, calc_tp, accumulate
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

from rvfn.datasets.dataset import SerializedDataset
from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset
from rvfn.model.model import RadarVoxelFusionNet
from rvfn.model.target import LossTargetGenerator
from rvfn.config.pipeline_config import PipelineConfig
from rvfn.config.eval_config import EvalConfig
from rvfn.config.infer_config import InferConfig
from rvfn.utils.collate import collate
from rvfn.infer import run, post_process


class Metrics:
    """
    Calculate average precision, and translation, scale, and orientation
        error according to official Nuscenes evaluation:
        https://www.nuscenes.org/object-detection
    """

    def __init__(self, md: DetectionMetricData):
        self.precision = md.precision
        self.recall = md.recall

        self.ap = calc_ap(md, 0.1, 0.1)
        self.scale_err = calc_tp(md, 0.1, 'scale_err')
        self.trans_err = calc_tp(md, 0.1, 'trans_err')
        self.orient_err = calc_tp(md, 0.1, 'orient_err')

    def serialize(self) -> dict:
        return {
            'precision': list(self.precision),
            'recall': list(self.recall),
            'average_precision': self.ap,
            'scale_error': self.scale_err,
            'translation_error': self.trans_err,
            'orientation_error': self.orient_err
        }


class MetricsMean:
    """
    An accumulation of metrics at different thresholds
    """

    def __init__(self, metrics_list: List[Metrics],
                 distance_thresholds: List[float]):
        """
        Args:
            metrics_list: List of metrics at different distance thresholds
            distance_thresholds: List of distance thresholds used to create the
                metrics
        """
        self.metrics_list = metrics_list
        self.distance_thresholds = distance_thresholds
        num_metrics = len(self.metrics_list)

        self.metrics_dict = {}
        for idx in range(num_metrics):
            self.metrics_dict[distance_thresholds[idx]] = \
                metrics_list[idx].serialize()

        self.mAP = sum(
            [metric.ap for metric in metrics_list]) / num_metrics
        self.mATE = sum(
            [metric.trans_err for metric in metrics_list]) / num_metrics
        self.mASE = sum(
            [metric.scale_err for metric in metrics_list]) / num_metrics
        self.mAOE = sum(
            [metric.orient_err for metric in metrics_list]) / num_metrics

    def serialize(self) -> dict:
        return {
            'threshold_metrics': self.metrics_dict,
            'mAP': self.mAP,
            'mATE': self.mATE,
            'mASE': self.mASE,
            'mAOE': self.mAOE
        }

    def print(self):
        print('mAP: ', self.mAP)
        print('mATE: ', self.mATE)
        print('mASE: ', self.mASE)
        print('mAOE: ', self.mAOE)


def get_pr_curves(metrics_mean: MetricsMean):
    """
    Make Precision-Recall curves as matplotlib figures from Metrics
    """
    metrics_list = metrics_mean.metrics_list
    thresholds = metrics_mean.distance_thresholds

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(metrics_list[0].recall, metrics_list[0].precision)
    axs[0, 0].set_title('@' + str(thresholds[0]) + 'm')

    axs[0, 1].plot(metrics_list[1].recall, metrics_list[1].precision)
    axs[0, 1].set_title('@' + str(thresholds[1]) + 'm')

    axs[1, 0].plot(metrics_list[2].recall, metrics_list[2].precision)
    axs[1, 0].set_title('@' + str(thresholds[2]) + 'm')

    axs[1, 1].plot(metrics_list[3].recall, metrics_list[3].precision)
    axs[1, 1].set_title('@' + str(thresholds[3]) + 'm')

    for ax in axs.flat:
        ax.label_outer()

    return fig


def write_metrics(metrics: MetricsMean, writer: SummaryWriter, step,
                  prefix='val'):
    """
    Write metrics to Tensorboard
    Args:
        prefix: Adds this string to the name of each scalar in Tensorboard
    """

    # Make a precision-recall plot for the last metric using matplotlib
    fig = get_pr_curves(metrics)

    writer.add_figure(prefix + 'Precision-Recall', fig, step)

    writer.add_scalar(prefix + 'mAP', metrics.mAP, step)
    writer.add_scalar(prefix + 'mATE', metrics.mATE, step)
    writer.add_scalar(prefix + 'mASE', metrics.mASE, step)
    writer.add_scalar(prefix + 'mAOE', metrics.mAOE, step)


def make_detection_box(box: torch.Tensor, score: float, sample_token: str):
    translation = tuple(box[:3])
    size = tuple(box[3:6])

    rotation = tuple(Quaternion(axis=(0.0, 0.0, 1.0), radians=box[6]))

    # placeholder for velocity as we don't currently predict that
    velocity = (0, 0)

    return DetectionBox(sample_token, translation, size, rotation, velocity,
                        detection_name='car', detection_score=score)


def evaluate(net: RadarVoxelFusionNet,
             dataset,
             target_generator: LossTargetGenerator,
             score_criteria,
             reg_criteria,
             dir_criteria,
             device,
             dataset_fraction: float = 1.0,
             metrics_file_path=None,
             pr_curve_file_path=None,
             eval_config: EvalConfig = None,
             infer_config: InferConfig = None,
             calc_loss: bool = True) -> (List[Metrics], float, float):
    """
    Calculates precision and recall of a model on a dataset.
    Args:
        net: Model to be evaluated
        dataset: Dataset to run the evaluation on
        target_generator: Target generator that was used for training the
            network
        score_criteria: Bbox score loss function.
        reg_criteria: Regression loss function
        dir_criteria: Direction loss function
        device: Torch device to run the model on
        dataset_fraction: Fraction of dataset to use. If less than one, an
            evenly spaced subset of dataset will be used. If you want to
            evaluate on a random subset, shuffle the dataset before passing it
            to this function.
        metrics_file_path: Where to save the results
        pr_curve_file_path: Path to a png file where the PR curve of the last
            metrics threshold will be saved
        eval_config: Evaluation config options
        calc_loss: Whether to calculate loss

    Returns:
        - A MetricsMean object
        - Bbox score loss, None if calc_loss=False
        - Regression loss, None if calc_loss=False
        - Direction loss, None if calc_loss=False

    """
    eval_config = eval_config if eval_config is not None else EvalConfig()
    infer_config = infer_config if infer_config is not None else InferConfig()

    net.eval()

    with torch.no_grad():
        # Deep copy as we may want to change the dataset samples and settings
        dataset = deepcopy(dataset)
        dataset.perturb = False

        if dataset_fraction < 1.0:
            num_samples = int(len(dataset) * dataset_fraction)
            # A list of evenly spaced sample indices to evaluate on.
            # Will be equivalent to all indices in the dataset if
            #  dataset_fraction = 1.0
            sample_inds = np.round(
                np.linspace(0, len(dataset) - 1, num_samples)).astype(int)

            dataset.samples = [dataset.samples[idx] for idx in sample_inds]

        val_loader = data.DataLoader(dataset, collate_fn=collate,
                                     batch_size=eval_config.batch_size,
                                     num_workers=eval_config.data_loader_workers)

        all_pred_bboxes = EvalBoxes()
        all_gt_bboxes = EvalBoxes()

        mean_reg_loss = 0
        mean_score_loss = 0
        mean_dir_loss = 0

        for i, sample in enumerate(val_loader):
            if i % 100 == 0:
                logging.info('Running predictions for batch {}'.format(i))

            voxels, features, gt_bboxes, labels, tokens = sample

            scores, regs, dirs = run(net, voxels, features)

            bbox_targets, reg_targets, dir_targets, _, positive, negative, \
            ignore = target_generator.get_targets(gt_bboxes)

            if calc_loss:
                mean_score_loss += score_criteria(scores,
                                                  bbox_targets.to(device),
                                                  positive.to(device),
                                                  negative.to(device))
                mean_reg_loss += reg_criteria(regs, reg_targets.to(device),
                                              positive.to(device))

                if target_generator.config.sine_yaw_targets:
                    mean_dir_loss += dir_criteria(dirs, dir_targets.to(device),
                                                  positive.to(device))

            # Predicted bboxes and scores for each sample in batch
            batch_bboxes, batch_bbox_scores = post_process(target_generator,
                                                           scores, regs,
                                                           dirs,
                                                           infer_config)
            for idx in range(len(batch_bboxes)):
                sample_bboxes = batch_bboxes[idx]
                sample_bbox_scores = batch_bbox_scores[idx]

                # Convert ground-truth and predicted bboxes to nuScenes
                #  EvalBoxes
                sample_bboxes = \
                    [make_detection_box(sample_bboxes[_],
                                        sample_bbox_scores[_].item(),
                                        tokens[idx])
                     for _ in range(sample_bboxes.shape[0])]

                sample_gt_bboxes = \
                    [make_detection_box(gt_bboxes[idx][_],
                                        -1.0, tokens[idx])
                     for _ in range(gt_bboxes[idx].shape[0])]

                all_pred_bboxes.add_boxes(tokens[idx], sample_bboxes)
                all_gt_bboxes.add_boxes(tokens[idx], sample_gt_bboxes)

        if calc_loss:
            mean_score_loss /= len(val_loader)
            mean_reg_loss /= len(val_loader)
            mean_dir_loss /= len(val_loader)
        else:
            mean_score_loss = None
            mean_reg_loss = None
            mean_dir_loss = None

        metrics_list = []
        for dist_th in eval_config.distance_thresholds:
            logging.info(
                'calculating metrics for dist threshold {}'.format(dist_th))
            dm = accumulate(all_gt_bboxes, all_pred_bboxes, 'car',
                            center_distance, dist_th)
            metrics_list.append(Metrics(dm))
        metrics_mean = MetricsMean(metrics_list,
                                   eval_config.distance_thresholds)

        if metrics_file_path is not None:
            with open(metrics_file_path, 'w+') as file:
                json.dump(metrics_mean.serialize(), file)

        if pr_curve_file_path is not None:
            fig = get_pr_curves(metrics_mean)
            fig.savefig(pr_curve_file_path)

    return metrics_mean, mean_score_loss, mean_reg_loss, mean_dir_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("pipeline_config",
                        help="Pipeline configuration json file")

    parser.add_argument("eval_config",
                        help="Evaluation configuration json file")

    parser.add_argument("checkpoint", help="checkpoint file")

    parser.add_argument("--val_samples",
                        help="Path to json file containing validation samples'"
                             " tokens", default=None)

    parser.add_argument("--fraction", help="Fraction of dataset to use",
                        default=1.0, type=float)

    parser.add_argument("--output", help="Path to metrics output file",
                        default=None)
    parser.add_argument("--pr_output", help="Path to PR curve output file",
                        default=None)

    parser.add_argument("--serialized",
                        help="Load from a serialized dataset. "
                             "Path to the dataset files directory")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    checkpoint = torch.load(args.checkpoint)

    pipeline_cfg = PipelineConfig.from_file(args.pipeline_config)
    eval_cfg = EvalConfig.from_file(args.eval_config)

    if args.serialized is not None:
        dataset = SerializedDataset(args.serialized,
                                    args.val_samples)
    else:
        dataset = NuscenesFusionDataset(pipeline_cfg.dataset_config,
                                        args.val_samples,
                                        perturb=False)

    gpu_zero = eval_cfg.device_ids[0]
    device = torch.device(
        'cuda:' + str(gpu_zero) if gpu_zero is not None else "cpu")

    net = RadarVoxelFusionNet(pipeline_cfg.model_config)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    target_generator = LossTargetGenerator(pipeline_cfg.target_config, device)

    metrics, _, _, _ = evaluate(net, dataset, target_generator,
                                None, None, None,
                                device, args.fraction, args.output,
                                args.pr_output,
                                eval_cfg,
                                pipeline_cfg.infer_config,
                                calc_loss=False)

    metrics.print()


if __name__ == '__main__':
    main()
