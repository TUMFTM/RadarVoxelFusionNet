import torch
import torch.nn as nn
import torch.nn.functional as F

from rvfn.config.loss_config import FocalLossConfig, \
    BinaryFocalLossConfig, ScoreLossConfig, RegLossConfig, DirLossConfig


class BinaryFocalLoss(nn.Module):
    """
    An implementation of focal loss: https://arxiv.org/abs/1708.02002
    Similar to FocalLoss() for two classes with the exception that this only
     takes one value per sample rather than two.
    """

    def __init__(self, config: BinaryFocalLossConfig = None):
        super(BinaryFocalLoss, self).__init__()
        self.config = config if config is not None else BinaryFocalLossConfig()

        self.eps = 1e-7

    def forward(self, prediction, target):
        """
        Args:
            prediction: (batch_size, positions) prediction values for class 0
            target: (batch_size, positions) target class indices (0 or 1)

        Returns:
            Single loss value if config.reduce == True,
            One loss value for each input (batch_size, positions) otherwise

        """
        p_t = prediction.clone()

        if self.config.logits:
            p_t = torch.sigmoid(p_t)
            p_t = p_t.clamp(self.eps, 1. - self.eps)

        target_zero = (target == 0)
        p_t[target_zero] = 1 - p_t[target_zero]

        alpha_t = torch.ones_like(p_t) * self.config.alpha
        alpha_t[target_zero] = 1 - alpha_t[target_zero]

        # binary cross entropy
        loss = -torch.log(p_t)
        # focal loss
        loss = alpha_t * ((1 - p_t) ** self.config.gamma) * loss

        if self.config.reduce:
            loss = loss.sum(-1)  # sum over anchors
            loss = loss.mean()  # mean over samples in minibatch

        return loss


class FocalLoss(nn.Module):
    """
    An implementation focal loss (https://arxiv.org/abs/1708.02002) for
     multiple classes.
    Based on:
     https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    """

    def __init__(self, config: FocalLossConfig = None):
        super(FocalLoss, self).__init__()
        self.config = config if config is not None else FocalLossConfig()

        self.eps = 1e-7

    def forward(self, prediction, target):
        """
        Args:
            prediction: (batch_size, positions, num_classes) prediction value
                for each class
            target: (batch_size, positions) target class indices

        Returns:
            Single loss value if config.reduce == True,
            One loss value for each input (batch_size, positions, num_classes)
             otherwise

        """
        num_classes = prediction.size(-1)
        assert num_classes == len(self.config.alphas)

        # batch_size x positions x num_classes
        y = F.one_hot(target, num_classes=num_classes)

        p_t = prediction.clone()
        if self.config.logits:
            p_t = F.softmax(p_t, dim=-1)
            p_t = p_t.clamp(self.eps, 1. - self.eps)

        alpha_t = torch.ones_like(p_t)
        for c in range(num_classes):
            alpha_t[:, :, c] *= self.config.alphas[c]

        loss = -y * torch.log(p_t)  # cross entropy
        # focal loss
        loss = alpha_t * ((1 - p_t) ** self.config.gamma) * loss

        if self.config.reduce:
            loss = loss.sum([-1, -2])  # sum over classes and anchors
            loss = loss.mean()  # mean over samples in minibatch

        return loss


class ScoreLoss(nn.Module):
    """
    Wrapper for calculation bbox (objectness) score loss.
    """

    def __init__(self, config: ScoreLossConfig = None):
        super(ScoreLoss, self).__init__()

        self.config = config if config is not None else ScoreLossConfig()
        if self.config.criteria == 'binary_focal':
            self.criteria = BinaryFocalLoss(self.config.criteria_config)
        else:
            raise NotImplementedError('Criteria not implemented: ',
                                      self.config.criteria)

        self.eps = 1e-7

    def forward(self, prediction, target, pos, neg):
        """
        Args:
            prediction: (batch_size, num_anchors) score for each anchor
            target: (batch_size, num_anchors) target of objectness
            pos: (batch_size, num_anchors) mask of positive anchors
            neg: (batch_size, num_anchors) mask of negative anchors

        Returns: Mean loss value over batch based on config.criteria

        """
        loss = self.criteria(prediction, target)

        # (batch_size x num_anchors)
        assert len(loss.shape) == 2

        num_pos_anchors = pos.sum(axis=1)
        num_neg_anchors = neg.sum(axis=1)

        pos_loss = loss.clone()
        pos_loss[~pos] = 0

        neg_loss = loss.clone()
        neg_loss[~neg] = 0

        # sum over anchors
        pos_loss = pos_loss.sum(-1)
        neg_loss = neg_loss.sum(-1)

        if self.config.normalize_by_type:
            # for each sample, divide total loss by the number of anchors that
            # contributed to the loss
            pos_loss /= num_pos_anchors + self.eps
            neg_loss /= num_neg_anchors + self.eps

            loss = pos_loss + neg_loss
        else:
            loss = pos_loss + neg_loss
            loss /= num_pos_anchors + num_neg_anchors + self.eps

        loss = loss.mean()  # mean over batch

        return loss


class RegLoss(nn.Module):
    """
    Wrapper for calculating regression (localization) loss
    """

    def __init__(self, config: RegLossConfig = None):
        super(RegLoss, self).__init__()

        self.config = config if config is not None else RegLossConfig()

        if self.config.criteria == 'smoothL1':
            self.criteria = nn.SmoothL1Loss(reduction='none')
        else:
            raise NotImplementedError('Criteria not implemented: ',
                                      self.config.criteria)

        self.eps = 1e-7

    def forward(self, prediction, target, positive):
        """
        Args:
            prediction: (batch_size, num_anchors, 7) predictions for bbox
                localization values
            target: (batch_size, num_anchors, 7) localization targets
            positive: (batch_size, num_anchors) Mask of anchors that should be
                included in loss calculation

        Returns: Mean loss value over batch based on config.criteria

        """
        loss = self.criteria(prediction, target)
        loss[~positive] = 0

        assert len(loss.shape) == 3

        # Number of non-ignored elements in each sample of batch
        num_valid_anchors = positive.sum(axis=1)

        # sum over parameters in each anchor and over anchors
        loss = loss.sum([-1, -2])
        # for each sample, divide total loss by the number of anchors that
        # contributed to the loss
        loss = loss / (num_valid_anchors + self.eps)

        loss = loss.mean()  # mean over batch

        return loss


class DirLoss(nn.Module):
    """
    Wrapper for calculating direction loss
    """

    def __init__(self, config: DirLossConfig = None):
        super(DirLoss, self).__init__()

        self.config = config if config is not None else DirLossConfig()

        if self.config.criteria == 'binary_cross_entropy':
            self.criteria = nn.BCELoss(reduction='none')
        else:
            raise NotImplementedError('Criteria not implemented: ',
                                      self.config.criteria)

        self.eps = 1e-7

    def forward(self, prediction, target, positive):
        """
        Args:
            prediction: (batch_size, num_anchors) direction predictions
            target: (batch_size, num_anchors) direction targets
            positive: (batch_size, num_anchors) Mask of anchors that should be
                included in loss calculation

        Returns: Mean loss value over batch based on config.criteria

        """
        loss = self.criteria(prediction, target)
        loss[~positive] = 0

        assert len(loss.shape) == 2

        # Number of non-ignored elements in each sample of batch
        num_valid_anchors = positive.sum(axis=1)

        # sum over each anchor
        loss = loss.sum(-1)
        # for each sample, divide total loss by the number of anchors that
        # contributed to the loss
        loss = loss / (num_valid_anchors + self.eps)

        loss = loss.mean()  # mean over batch

        return loss


def test():
    """
    Some examples for testing the loss functions
    """
    dir_loss = DirLoss()
    dir_pred = torch.as_tensor([[0.7, 0.8, 0.1], [0.3, 0.2, 0.1]], dtype=torch.float)
    dir_target = torch.as_tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float)
    pos = torch.as_tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.bool)

    print("DIR", dir_loss(dir_pred, dir_target, pos))

    fl_config = FocalLossConfig(
        {'gamma': 2.0, 'alphas': [0.25, 0.65, 0.1], 'logits': True})

    # Focal Loss
    fl_input = torch.as_tensor(
        [[[1, 5, 5], [12, 5, 0]], [[1, 5, 5], [12, 5, 0]]], dtype=torch.float)
    fl_target = torch.as_tensor([[1, 0], [0, 2]], dtype=torch.int64)

    fl = FocalLoss(fl_config)

    print("FL", fl(fl_input, fl_target))

    bfl_config = BinaryFocalLossConfig(
        {'gamma': 2.0, 'alpha': 0.25, 'logits': True})
    # Binary Focal Loss
    bfl_input = torch.as_tensor([[8, .1, -10], [2, 50, -7]],
                                dtype=torch.float)
    bfl_target = torch.as_tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.int64)

    bfl = BinaryFocalLoss(bfl_config)

    print("BFL", bfl(bfl_input, bfl_target))

    # BboxLoss
    bbox_ignore = torch.as_tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.bool)
    pos = torch.as_tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    neg = torch.as_tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.bool)

    bboxl = ScoreLoss(ScoreLossConfig({'criteria_config': bfl_config}))

    print("BBLoss", bboxl(bfl_input, bfl_target, pos, neg))

    # Regression Loss
    reg_input = torch.as_tensor([[[1, 2, 3, 4, 5, 6, 0],
                                  [10, 20, 33, 42, 10, 2, 0]],

                                 [[0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1]]], dtype=torch.float)
    reg_target = torch.as_tensor([[[1, 3, 3, 6, 4, 7, 0],
                                   [1, 3, 3, 6, 4, 7, 0]],

                                  [[-9, 0, 10, 0, 0, 5, 0],
                                   [2, 1, 21, 1, 13, 12, 2]]], dtype=torch.float)
    reg_pos = torch.as_tensor([[0, 1], [1, 1]], dtype=torch.bool)
    regloss = RegLoss()

    print("REGLOSS", regloss(reg_input, reg_target, reg_pos))


if __name__ == '__main__':
    test()
