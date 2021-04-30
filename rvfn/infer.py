import argparse
from pathlib import Path
import time
import json

import torch
from torchvision.ops import nms
from progressbar import progressbar

from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset
from rvfn.model.model import RadarVoxelFusionNet
from rvfn.model.target import LossTargetGenerator, upright_shadows
from rvfn.utils.collate import collate
from rvfn.config.infer_config import InferConfig
from rvfn.config.pipeline_config import PipelineConfig
from rvfn.utils.results_dict import make_result_dict


def run(model: RadarVoxelFusionNet,
        voxels: torch.Tensor,
        features: torch.Tensor):
    """
    Run model on a batch of data
    Args:
        model: model instance
        voxels: (num_anchors_in_batch, 4) where the first 3
                numbers are voxel coordinates, and the 4th indicates the
                sample id in batch.
        features: (num_voxels, num_features) sparse features tensor

    Returns:
        scores: (batch_size, num_anchors,) probability of objectness per anchor
        regs: (batch_size, num_anchors, 7) bbox regression values
        dirs: (batch_size, num_anchors,) direction prediction for each anchor

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scores, regs, dirs = model(voxels.to(device), features.to(device))

    return scores, regs, dirs


def post_process(target_generator: LossTargetGenerator,
                 scores, regs, dirs, config: InferConfig):
    """
    Post process the model outputs to create predicted bounding boxes
    Args:
        target_generator:
        scores: (batch_size, num_anchors)
        regs: (batch_size, num_anchors, 7)
        dirs: (batch_size, num_anchors)
        config:

    Returns:
        bboxes: list[(<num_boxes>, 7), len=batch_size]
        bbox_scores: list[(<num_boxes>,), len=batch_size]
    """
    batch_size = scores.shape[0]

    bboxes = []
    bbox_scores = []

    for idx in range(batch_size):
        sample_bboxes = target_generator.apply_regs(regs[idx], dirs[idx]).cpu()
        sample_scores = scores[idx].cpu()

        # Filter low confidence bboxes
        min_conf_inds = torch.where(sample_scores > config.min_confidence)[0]
        sample_bboxes = sample_bboxes[min_conf_inds]
        sample_scores = sample_scores[min_conf_inds]

        # Non Maximum Suppression
        upright_bboxes = torch.as_tensor(upright_shadows(
            sample_bboxes.numpy()), dtype=torch.float)
        nms_inds = nms(upright_bboxes, sample_scores, config.nms_threshold)

        sample_bboxes = sample_bboxes[nms_inds]
        sample_scores = sample_scores[nms_inds]

        bboxes.append(sample_bboxes)
        bbox_scores.append(sample_scores)

    return bboxes, bbox_scores


def infer(model: RadarVoxelFusionNet,
          target_generator: LossTargetGenerator,
          sample,
          config: InferConfig):
    """
    Run inference on a single sample

    Args:
        model: Model instance
        target_generator: Target generator that was used for training the model
        sample: A single sample from nuScenesFusionDataset
        config: Inference configurations

    Returns:
        bboxes: (<num_boxes>, 7)
        scores: (<num_boxes>,)

    """
    with torch.no_grad():
        model.eval()

        voxels, features, gt_bboxes, labels, tokens = collate([sample])

        scores, regs, dirs = run(model, voxels, features)

        bboxes, scores = post_process(target_generator, scores,
                                      regs, dirs, config)

        return bboxes[0], scores[0]


def make_results_dict(tokens, ego_poses, offsets, bboxes_list, scores_list,
                      detection_name='car'):
    """
    Creates a results dictionary in compliance with official nuScenes
        submission format:
        https://www.nuscenes.org/object-detection/
    Args:
        tokens: List of sample tokens
        ego_poses: nuScenes ego pose record for each sample
        offsets: Dataset offset
        bboxes_list: Predictions for each sample
        scores_list: Prediction scores for each sample
        detection_name: The class of the predictions

    Returns: results dict
    """
    ret = {}
    for sample_idx in range(len(tokens)):
        token = tokens[sample_idx]
        ego_pose = ego_poses[sample_idx]
        ret[token] = []

        for pred_idx in range(len(bboxes_list[sample_idx])):
            bbox = bboxes_list[sample_idx][pred_idx]

            sample_result = make_result_dict(token, bbox, offsets, ego_pose,
                                             scores_list[sample_idx][pred_idx],
                                             detection_name)
            ret[token].append(sample_result)

    return ret


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('pipeline_config',
                        help='Pipeline configuration json file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('out', help='Output path')
    parser.add_argument('--min_conf', type=float,
                        help='Override min confidence score in config.')
    parser.add_argument('--samples',
                        help='Path to json file containing samples tokens to'
                             ' infer on. Entire dataset will be used if not'
                             ' provided.', default=None)
    parser.add_argument('--display_conf',
                        help='Display confidence for predicted bboxes',
                        action='store_true')
    parser.add_argument('--write_preds',
                        help='Write predictions dict to disk',
                        action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipeline_cfg = PipelineConfig.from_file(args.pipeline_config)
    if args.min_conf is not None:
        pipeline_cfg.infer_config.min_confidence = args.min_conf

    model = RadarVoxelFusionNet(pipeline_cfg.model_config).to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = NuscenesFusionDataset(pipeline_cfg.dataset_config,
                                    args.samples, perturb=False)

    target_generator = LossTargetGenerator(pipeline_cfg.target_config, device)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    tokens = []
    ego_poses = []
    bboxes_list = []
    scores_list = []

    total_time_preprocess = 0
    total_time_infer = 0

    for idx in progressbar(range(len(dataset))):
        start = time.time()

        sample = dataset[idx]

        total_time_preprocess += time.time() - start
        start = time.time()

        bboxes, scores = infer(model, target_generator, sample,
                               pipeline_cfg.infer_config)

        total_time_infer += time.time() - start

        tokens.append(sample['sample_token'])
        ego_poses.append(dataset.get_ego_pose(idx))
        bboxes_list.append(bboxes.numpy().astype(float))
        scores_list.append(scores.numpy().astype(float))

        if args.display_conf:
            fig = dataset.visualize_sample_bev(idx, bboxes.numpy(),
                                               scores.numpy())
        else:
            fig = dataset.visualize_sample_bev(idx, bboxes.numpy())

        fig.savefig(out_path / (
                str(idx).zfill(5) + "_" + sample['sample_token'] + '.svg'))

    print('Average preprocessing time: {:.3f}s'.format(
        total_time_preprocess / len(dataset)))
    print('Average inference time: {:.3f}s'.format(
        total_time_infer / len(dataset)))

    if args.write_preds:
        print('Writing predictions to json...')
        with open(out_path / 'predictions.json', 'w') as f:
            predictions = make_results_dict(tokens, ego_poses, dataset.offsets,
                                            bboxes_list, scores_list)
            json.dump(predictions, f, indent=4)

    print('Done.')


if __name__ == '__main__':
    main()
