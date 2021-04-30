from rvfn.config.target_config import AnchorGeneratorConfig, \
    LossTargetGeneratorConfig
from rvfn.datasets.common import upright_shadows

import numpy as np
import torch

from torchvision.ops import box_iou


class AnchorGenerator:
    """
    Creates anchors conforming to the given AnchorGeneratorConfig.
    Given a config with:
        feature_map_shape: (H(y), W(x))
        anchors_per_position: N (:= len(sizes) x len(rotations))
      the anchors will be of shape: H x W x N x 7
    """

    def __init__(self, config: AnchorGeneratorConfig = None):
        self.config = config if config is not None else AnchorGeneratorConfig()
        self.anchors = torch.as_tensor(self._generate_anchors(),
                                       dtype=torch.float)

    def _generate_anchors(self):
        H, W = self.config.feature_map_shape
        x_range, y_range = self.config.anchor_range

        anchors_per_pos = self.config.anchors_per_position

        x = np.linspace(x_range[0], x_range[1], W)
        y = np.linspace(y_range[0], y_range[1], H)
        cx, cy = np.meshgrid(x, y)

        cx = np.tile(cx[..., np.newaxis], anchors_per_pos)
        cy = np.tile(cy[..., np.newaxis], anchors_per_pos)
        cz = np.ones_like(cx) * self.config.center_z

        w = np.empty_like(cx)
        l = np.empty_like(cx)
        h = np.empty_like(cx)
        r = np.empty_like(cx)

        anchor_idx = 0
        for size in self.config.anchor_sizes:
            for rotation in self.config.anchor_rotations:
                w[..., anchor_idx] = size[0]
                l[..., anchor_idx] = size[1]
                h[..., anchor_idx] = size[2]

                r[..., anchor_idx] = rotation

                anchor_idx += 1

        return np.stack([cx, cy, cz, w, l, h, r], axis=-1)


class LossTargetGenerator:
    """
    Creates anchors, and produces loss targets based on ground-truth data.
    """

    def __init__(self, config: LossTargetGeneratorConfig = None,
                 device=None):
        self.config = config if config is not None else \
            LossTargetGeneratorConfig()

        self.device = device if device is not None else torch.device('cpu')

        self.anchor_generator = AnchorGenerator(self.config.anchor_config)
        self.anchors = self.anchor_generator.anchors.reshape(-1, 7). \
            to(self.device)
        self.num_anchors = self.anchors.shape[0]

        # anchors are in format [x, y, z, w, l, h, yaw]
        self.anchors_diag = torch.sqrt(
            self.anchors[:, 3] ** 2 + self.anchors[:, 4] ** 2).to(self.device)

        self.anchors4d = torch.as_tensor(
            upright_shadows(self.anchors.cpu().numpy()),
            dtype=torch.float).to(self.device)

    def _get_bbox_targets(self, positive_mask):
        """
        Creates targets for objectness scores
        Args:
            positive_mask: (num_anchors,) boolean array

        Returns:
            (num_anchors,) array of integers in which 1 indicates positive
             anchor, and 0 indicates negative

        """

        targets = torch.zeros(self.num_anchors).to(self.device)
        targets[positive_mask] = 1

        return targets

    def _get_reg_targets(self, gt_bboxes, gt_idx_per_anchor):
        """
        Creates targets for localization and direction values.
        References:
            This matches VoxelNet's regression targets with the exception of
                the yaw targets if config.sine_yaw_targets is True.
            https://arxiv.org/abs/1711.06396
        Args:
            gt_bboxes: (N, 7) array of ground-truth bounding boxes
            gt_idx_per_anchor: (num_anchors,)

        Returns:
            Regression targets: (num_anchors, 7)
            Direction targets: (num_anchors, 1) indicating weather the yaw is
                more than Pi. Will be all zeros if config.sine_yaw_targets is
                False as the direction is predicted directly with the yaw in
                that case.
        """

        targets = torch.empty((self.num_anchors, 7)).to(self.device)

        # For each anchor, the ground-truth bbox with which it has the highest
        #  iou (num_anchors, 7)
        assigned_gt_bboxes = gt_bboxes[gt_idx_per_anchor]

        # targets for x, y, and z
        for i in range(3):
            targets[:, i] = assigned_gt_bboxes[:, i] - self.anchors[:, i]
            targets[:, i] /= self.anchors_diag

        # targets for w, l, and h
        for i in range(3, 6):
            targets[:, i] = torch.log(
                assigned_gt_bboxes[:, i] / self.anchors[:, i])

        # targets for yaw and direction if applicable
        dir_targets = torch.zeros(self.num_anchors).to(self.device)

        gt_yaws = assigned_gt_bboxes[:, 6]
        anchor_yaws = self.anchors[:, 6]

        if self.config.sine_yaw_targets:
            # gt_yaws' yaws are already in [0, 2pi)
            targets[:, 6] = torch.sin(gt_yaws - anchor_yaws)

            # Direction target is 1 iff the target yaw is within pi/2 of the
            #  anchor.
            dir_targets[torch.abs(
                (gt_yaws - anchor_yaws + np.pi) % (2 * np.pi) - np.pi) >
                        (np.pi / 2)] = 1
        else:
            targets[:, 6] = gt_yaws - anchor_yaws

        return targets, dir_targets

    def apply_regs(self, regs: torch.Tensor,
                   dirs: torch.Tensor) -> torch.Tensor:
        """
        Applies the output localization values from the model to the anchors
         to get prediction bounding boxes.
        Args:
            regs: (num_anchors, 7) output from the model's regression head.
            dirs: (num_anchors,) output from the model's direction head. Can
                be None if self.config.sine_yaw_targets is False.

        Returns: (num_anchors, 7) predicted bboxes

        """
        bboxes = self.anchors.clone()

        # for x, y , and z
        for i in range(3):
            regs[:, i] *= self.anchors_diag
            bboxes[:, i] += regs[:, i]

        # for w, l, and h
        for i in range(3, 6):
            bboxes[:, i] *= torch.exp(regs[:, i])

        # for yaw
        if self.config.sine_yaw_targets:
            regs[:, 6] = torch.clamp(regs[:, 6], -1.0, 1.0)

            bboxes[:, 6] += torch.asin(regs[:, 6])
            bboxes[:, 6] %= 2 * np.pi

            flipped_mask = (dirs > 0.5)
            bboxes[:, 6][flipped_mask] = \
                2 * self.anchors[:, 6][flipped_mask] + np.pi - \
                bboxes[:, 6][flipped_mask]
        else:
            bboxes[:, 6] += regs[:, 6]

        return bboxes

    def get_targets(self, batch_gt_bboxes):
        """
        Creates targets for learning based on ground-truth bboxes
        Args:
            batch_gt_bboxes: list<(N, 7), len=batch_size> list of ground-truth
             bounding boxes

        Returns:
            (batch_size, num_anchors) bbox score targets,
            (batch_size, num_anchors, 7) localization targets,
            (batch_size, num_anchors) direction targets,
            (batch_size, feature_map_size) classification targets (TODO),
            (batch_size, num_anchors) mask of positive anchors
            (batch_size, num_anchors) mask of negative anchors
            (batch_size, num_anchors) mask of ignore anchors

            Anchor masks indicate which anchors should be included in loss
             calculation. Specifically, positive anchors are the ones that
             match a ground-truth box, negative anchors are ones that don't,
             and 'ignore' anchors are ones that are in-between and should be
             ignored.

        """
        with torch.no_grad():
            bbox_targets = []
            reg_targets = []
            dir_targets = []

            positive_masks = []
            negative_masks = []
            ignore_masks = []

            for gt_bboxes in batch_gt_bboxes:
                gt_bboxes = gt_bboxes.to(self.device)

                # gt_bboxes shape: (N, 7) where N is the number of ground-truth
                #  bounding boxes.
                if gt_bboxes.shape[0] == 0:
                    positive_masks.append(
                        torch.zeros(self.num_anchors,
                                    dtype=torch.bool).to(self.device))
                    negative_masks.append(
                        torch.ones(self.num_anchors,
                                   dtype=torch.bool).to(self.device))
                    ignore_masks.append(
                        torch.zeros(self.num_anchors,
                                    dtype=torch.bool).to(self.device))

                    bbox_targets.append(
                        torch.zeros(self.num_anchors).to(self.device))
                    reg_targets.append(
                        torch.zeros((self.num_anchors, 7)).to(self.device))
                    dir_targets.append(
                        torch.zeros(self.num_anchors).to(self.device))

                    continue

                # Convert yaws to interval [0, 2pi)
                gt_bboxes[:, 6] %= 2 * np.pi

                gt4d = torch.as_tensor(
                    upright_shadows(gt_bboxes.cpu().numpy()),
                    dtype=torch.float).to(self.device)

                ious = box_iou(self.anchors4d, gt4d)

                max_iou_per_anchor, gt_idx_per_anchor = torch.max(ious, dim=1)

                pos_iou_mask = max_iou_per_anchor > self.config.pos_threshold

                if self.config.pos_dist_threshold is not None:
                    gt_xys = gt_bboxes[gt_idx_per_anchor][:, :2]
                    dist_to_closest_gt = torch.norm((self.anchors[:, :2] -
                                                     gt_xys), dim=1)
                    pos_dist_mask = dist_to_closest_gt < \
                                    self.config.pos_dist_threshold

                    positive_mask = torch.logical_and(pos_iou_mask,
                                                      pos_dist_mask)
                else:
                    positive_mask = pos_iou_mask

                negative_mask = max_iou_per_anchor < self.config.neg_threshold
                ignore_mask = ~positive_mask & ~negative_mask

                positive_masks.append(positive_mask)
                negative_masks.append(negative_mask)
                ignore_masks.append(ignore_mask)

                bbox_targets.append(self._get_bbox_targets(positive_mask))

                regs, dirs = self._get_reg_targets(gt_bboxes,
                                                   gt_idx_per_anchor)
                reg_targets.append(regs)
                dir_targets.append(dirs)

            bbox_targets = torch.stack(bbox_targets)
            reg_targets = torch.stack(reg_targets)
            dir_targets = torch.stack(dir_targets)

            positive_masks = torch.stack(positive_masks)
            negative_masks = torch.stack(negative_masks)
            ignore_masks = torch.stack(ignore_masks)

        return bbox_targets, \
               reg_targets, \
               dir_targets, \
               None, \
               positive_masks, \
               negative_masks, \
               ignore_masks


def test():
    """Small example to test the anchor generator functionality"""
    ag = AnchorGenerator(AnchorGeneratorConfig({'feature_map_shape': (2, 3)}))
    print(ag.anchors)

    tg = LossTargetGenerator()
    print("ANCHORS READY")

    bboxes = torch.as_tensor([[[1, 1, 1, 3, 2, 2, 0]]], dtype=torch.float)
    bbox_targets, reg_targets, dir_targets, _, positive, negative, ignore = \
        tg.get_targets(bboxes)

    print("BBox_TARGETS", bbox_targets.shape)
    print("Reg_TARGETS", reg_targets.shape)
    print("Dir_TARGETS", dir_targets.shape)
    print("POSITIVE", positive.shape)
    print("NEGATIVE", negative.shape)
    print("IGNORE", ignore.shape)


if __name__ == '__main__':
    test()
