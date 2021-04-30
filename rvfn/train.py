import argparse
import logging
from pathlib import Path
import json

import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from rvfn.datasets.dataset import SerializedDataset
from rvfn.datasets.nuscenes_fusion import NuscenesFusionDataset
from rvfn.model.model import RadarVoxelFusionNet
from rvfn.model.loss import RegLoss, ScoreLoss, DirLoss
from rvfn.config.training_config import TrainingConfig
from rvfn.config.eval_config import EvalConfig
from rvfn.config.pipeline_config import PipelineConfig
from rvfn.model.target import LossTargetGenerator
from rvfn.eval import evaluate, write_metrics
from rvfn.utils.collate import collate


def train(pipeline_config: PipelineConfig,
          train_config: TrainingConfig,
          eval_config: EvalConfig,
          train_samples_filepath: str,
          val_samples_filepath: str,
          serialized_dataset_path: str = None,
          checkpoint=None):
    if serialized_dataset_path:
        train_dataset = SerializedDataset(serialized_dataset_path,
                                          train_samples_filepath)
        val_dataset = SerializedDataset(serialized_dataset_path,
                                        val_samples_filepath)
    else:
        train_dataset = NuscenesFusionDataset(pipeline_config.dataset_config,
                                              train_samples_filepath,
                                              perturb=train_config.augment)
        val_dataset = NuscenesFusionDataset(pipeline_config.dataset_config,
                                            val_samples_filepath,
                                            perturb=False)

    train_loader = data.DataLoader(train_dataset, collate_fn=collate,
                                   batch_size=train_config.batch_size,
                                   num_workers=train_config.data_loader_workers)

    num_devices = len(train_config.device_ids)
    gpu_zero = None
    if num_devices > 0:
        assert torch.cuda.is_available(), 'You specified GPU ids but cuda is' \
                                          ' not available. Use device_ids: ' \
                                          '[] to run on CPU.'
        gpu_zero = train_config.device_ids[0]

    device = torch.device(
        'cuda:' + str(gpu_zero) if gpu_zero is not None else "cpu")

    net = RadarVoxelFusionNet(pipeline_config.model_config)

    if num_devices > 1:
        if pipeline_config.model_config.middle_block == 'sparse':
            raise NotImplementedError(
                'SparseConvNet doesn\'t support multi-GPU '
                'operation. You can use a different middle '
                'block with multiple GPUs.')
        else:
            net = nn.DataParallel(net, device_ids=train_config.device_ids)

    net.to(device)

    if train_config.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=train_config.learning_rate)
    elif train_config.optimizer == 'AMSGrad':
        optimizer = optim.Adam(net.parameters(), lr=train_config.learning_rate,
                               eps=train_config.epsilon,
                               weight_decay=train_config.weight_decay,
                               amsgrad=True)
    else:
        raise NotImplementedError('Optimizer not implemented: ',
                                  train_config.optimizer)

    score_criteria = ScoreLoss(
        train_config.loss_config.score_loss_config)
    reg_criteria = RegLoss(train_config.loss_config.reg_loss_config)
    dir_criteria = DirLoss(train_config.loss_config.dir_loss_config)

    target_generator = LossTargetGenerator(pipeline_config.target_config,
                                           device)

    epoch = 0
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1

    # Create output directories if they don't exist already
    base_out_path = Path(train_config.out_path)
    checkpoint_path = base_out_path / 'checkpoints'
    logs_path = base_out_path / 'logs'
    config_save_path = base_out_path / 'configs'

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(exist_ok=True)
    config_save_path.mkdir(exist_ok=True)

    writer = SummaryWriter(logs_path)

    # Write pipeline, training, and evaluation configs used for the session
    #  to the output directory
    with open(config_save_path / 'pipeline_config.json', 'w') as f:
        json.dump(pipeline_config.dict(), f, indent=4)
    with open(config_save_path / 'train_config.json', 'w') as f:
        json.dump(train_config.dict(), f, indent=4)
    with open(config_save_path / 'eval_config.json', 'w') as f:
        json.dump(eval_config.dict(), f, indent=4)

    epochs = train_config.epochs
    while epoch < epochs:
        logging.info("Starting epoch {}".format(epoch))

        running_reg_loss = 0.0
        running_score_loss = 0.0
        running_dir_loss = 0.0

        running_total_loss = 0.0

        net.train()
        for i, sample in enumerate(train_loader):
            voxels, features, bboxes, labels, tokens = sample

            # In case the last sample in batch has no points, SparseConvNet
            #  may not deduce the batch size correctly. This is a workaround
            #  by checking if the voxels tensor is non-empty, and the sample-id
            #  of the last voxel matches the batch_size.
            if voxels.shape[0] == 0 or voxels[-1][-1] != len(bboxes) - 1:
                continue

            voxels = voxels.to(device)
            features = features.to(device)
            scores, regs, dirs = net(voxels, features)

            bbox_targets, reg_targets, dir_targets, _, positive, negative, \
                ignore = target_generator.get_targets(bboxes)

            score_loss = score_criteria(scores,
                                        bbox_targets.to(device),
                                        positive.to(device),
                                        negative.to(device))
            reg_loss = reg_criteria(regs, reg_targets.to(device),
                                    positive.to(device))

            running_score_loss += score_loss.item()
            running_reg_loss += reg_loss.item()

            score_loss *= train_config.loss_config.score_loss_weight
            reg_loss *= train_config.loss_config.reg_loss_weight

            loss = score_loss + reg_loss

            if pipeline_config.target_config.sine_yaw_targets:
                dir_loss = dir_criteria(dirs,
                                        dir_targets.to(device),
                                        positive.to(device))

                running_dir_loss += dir_loss.item()
                dir_loss *= train_config.loss_config.dir_loss_weight

                loss += dir_loss

            running_total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            interval = train_config.log_interval
            if i % interval == interval - 1:  # every interval mini-batches
                logging.info(
                    "RUNNING LOSS: {}".format(running_total_loss / interval))

                step = (epoch * len(
                    train_loader) + i) * train_config.batch_size

                writer.add_scalar("running_loss_total",
                                  running_total_loss / interval, step)
                writer.add_scalar("running_loss_reg",
                                  running_reg_loss / interval, step)
                writer.add_scalar("running_loss_score",
                                  running_score_loss / interval, step)
                writer.add_scalar("running_loss_dir",
                                  running_dir_loss / interval, step)

                running_score_loss = 0
                running_reg_loss = 0
                running_dir_loss = 0
                running_total_loss = 0

        if epoch % train_config.eval_interval == \
                train_config.eval_interval - 1:
            # Evaluate on the validation set
            val_metrics, val_score_loss, val_reg_loss, val_dir_loss = \
                evaluate(net, val_dataset, target_generator,
                         score_criteria, reg_criteria, dir_criteria, device,
                         dataset_fraction=1.0, eval_config=eval_config,
                         infer_config=pipeline_config.infer_config)

            writer.add_scalar("val_loss_bbox", val_score_loss, epoch)
            writer.add_scalar("val_loss_reg", val_reg_loss, epoch)
            writer.add_scalar("val_loss_dir", val_dir_loss, epoch)

            # Write the metrics to tensorboard
            write_metrics(val_metrics, writer, epoch, 'val_')

            if train_config.eval_on_train_fraction > 0:
                # Evaluate on a fraction of training set
                train_metrics, _, _, _ = \
                    evaluate(net, train_dataset,
                             target_generator,
                             score_criteria,
                             reg_criteria,
                             dir_criteria,
                             device,
                             dataset_fraction=train_config.eval_on_train_fraction,
                             eval_config=eval_config,
                             infer_config=pipeline_config.infer_config,
                             calc_loss=False)

                write_metrics(train_metrics, writer, epoch, 'train_')

            val_metrics.print()

        if epoch % train_config.checkpoint_interval == \
                train_config.checkpoint_interval - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path / 'checkpoint_{}'.format(epoch))
        epoch += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("pipeline_config",
                        help="Pipeline configuration json file")
    parser.add_argument("train_config",
                        help="Training configuration json file")
    parser.add_argument("eval_config",
                        help="Evaluation configuration json file")

    parser.add_argument("--train_samples",
                        help="Path to json file containing training samples' "
                             "tokens", default=None)
    parser.add_argument("--val_samples",
                        help="Path to json file containing validation samples'"
                             " tokens", default=None)

    parser.add_argument("--checkpoint", help="checkpoint file",
                        default=None)

    parser.add_argument("--serialized",
                        help="Load from a serialized dataset. "
                             "Path to the dataset files directory")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    pipeline_cfg = PipelineConfig.from_file(args.pipeline_config)
    train_cfg = TrainingConfig.from_file(args.train_config)
    eval_cfg = EvalConfig.from_file(args.eval_config)

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)

    train(pipeline_cfg, train_cfg, eval_cfg, args.train_samples,
          args.val_samples, args.serialized, checkpoint)


if __name__ == '__main__':
    main()
