"""
The main network module definitions. Most of this with the exception of the
 sparse convolutional block and a few other changes is written following the
 VoxelNet paper: https://arxiv.org/abs/1711.06396
Also some of this code is either directly taken from, or is re-written based on
 some unofficial implementations of VoxelNet here:
 https://github.com/tsinghua-rll/VoxelNet-tensorflow
 https://github.com/skyhehe123/VoxelNet-pytorch
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import sparseconvnet as scn
from rvfn.config.model_config import SvfebConfig, CobConfig, \
    RvfnConfig, HeadConfig, CmbConfig


class Conv2d(nn.Module):
    """
    Wrapper for nn.Conv2D with options for BN and activation
    Taken from:
        https://github.com/skyhehe123/VoxelNet-pytorch
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=True, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class FCN(nn.Module):
    """
    Wrapper for nn.Linear + nn.BatchNorm
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        voxels, points, _ = x.shape

        # Input to batch-norm layer will be (num_points, num_feature_channels)
        x = self.linear(x.view(voxels * points, -1))
        x = F.relu(self.bn(x))

        return x.view(voxels, points, -1)


class VFE(nn.Module):
    """
    Voxel Feature Encoding layer.
    As per: https://arxiv.org/abs/1711.06396
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0, 'Number of output channels must be even.'

        self.out_channels = out_channels
        self.half_out_channels = out_channels // 2

        self.fcn = FCN(in_channels, self.half_out_channels)

    def forward(self, x, mask: torch.tensor):
        """
        Args:
            mask: (num_voxels, max_points_per_voxel), Array indicating the
            points in the input that are valid.
        """
        num_voxels, max_points_per_voxel, num_features = x.shape

        # point-wise features
        # (num_voxels x max_points_per_voxel x self.half_out_channels)
        pwf = self.fcn(x)

        # locally aggregated features, element-wise max-pool
        # (num_voxels x 1 x self.half_out_channels)
        laf = torch.max(pwf, 1, keepdim=True)[0]

        # Repeat and concatenate with calculated features of each point
        # (num_voxels x max_points_per_voxel x self.half_out_channels)
        laf = laf.repeat(1, max_points_per_voxel, 1)

        # (num_voxels x max_points_per_voxel x self.out_channels)
        pwcf = torch.cat((pwf, laf), dim=2)

        # apply mask of valid points
        # (repeat for number of out_channels per voxel)
        mask = mask.unsqueeze(2).repeat(1, 1, self.out_channels)

        pwcf = pwcf * mask.float()

        return pwcf


class SVFEB(nn.Module):
    """
    Stacked Voxel Feature Encoding block.
    As per: https://arxiv.org/abs/1711.06396
    """

    def __init__(self, config: SvfebConfig = None):
        super().__init__()

        self.config = config if config is not None else SvfebConfig()

        in_channels = self.config.in_channels
        hidden_channels = self.config.hidden_channels
        out_channels = self.config.out_channels

        self.vfe_1 = VFE(in_channels, hidden_channels)
        self.vfe_2 = VFE(hidden_channels, out_channels)
        self.fcn = FCN(out_channels, out_channels)

    def forward(self, x):
        # mask of valid points in input
        # (num_voxels, max_points_per_voxel)
        mask = torch.ne(torch.max(x, 2)[0], 0)

        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)

        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


class SCMB(nn.Module):
    """
    Sparse Convolutional Middle Block. Consisting of several conventional
     convolutions interspersed by Submanifold Sparse Convolutions for increased
     performance.
    For the input of the Sparse input layer, the voxel coordinates must be
     appended by the batch index. See scn.InputLayer for more details.
    """

    def __init__(self, config: CmbConfig = None):
        nn.Module.__init__(self)

        self.config = config if config is not None else CmbConfig()

        self.input_layer = scn.InputLayer(3, self.config.input_spatial_size,
                                          mode=0)

        # SparseResNet creates a ResNet-like network where every convolution
        # that doesn't change the input spatial-size is replaced with an SSC.
        # The signature is <number of dimensions of the input space,
        #                   number of output channels (filters),
        #                   a list of layers, where in each element we have:
        #                           [<type of layer ('b':='basic')>,
        #                            <number of output channels>,
        #                            <repetition of this layer>,
        #                            <stride>]
        # The input spatial size must be such that it can neatly accommodate
        # the down-sampling operations (such as stride=2 convs) without residue
        self.sparse_net = scn.Sequential(
            scn.SparseResNet(3, self.config.in_channels, [['b', 128, 1, 1]]),

            # conventional convolution, with stride=2 on the z axis
            scn.Convolution(3, 128, 64, (3, 3, 3), (2, 1, 1), False),
            scn.BatchNormReLU(64),
            scn.Convolution(3, 64, self.config.out_channels, (3, 3, 3),
                            (1, 1, 1), False),
            scn.BatchNormReLU(self.config.out_channels)
        )

        expected_input_spatial_size = tuple(self.sparse_net.input_spatial_size(
            torch.LongTensor(self.config.expected_output_spatial_size)))

        assert expected_input_spatial_size == tuple(
            self.config.input_spatial_size), \
            'The spatial sizes don\'t match the expected.' + \
            ' Expected input: ' + str(expected_input_spatial_size) + \
            ' Got input: ' + str(self.config.input_spatial_size)

        self.sparse_to_dense = scn.SparseToDense(3, self.config.out_channels)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.sparse_net(x)
        x = self.sparse_to_dense(x)

        return x


class CMB(nn.Module):
    """
    Convolutional Middle Block. Equivalent to SCMB, but using conventional
     convolutions
    """

    def __init__(self, config: CmbConfig = None):
        super().__init__()

        self.config = config if config is not None else CmbConfig()

        self.conv1 = nn.Conv3d(self.config.in_channels, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 64, 3, (2, 1, 1), 0)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, self.config.out_channels, 3, 1, 0)
        self.bn4 = nn.BatchNorm3d(self.config.out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)

        return out


class COB(nn.Module):
    """
    Convolutional output block. Creates a more abstract feature map from the
     output of the middle block. This is identical to the RPN in VoxelNet,
     except the prediction heads are decoupled.
    Adapted from:
     https://github.com/skyhehe123/VoxelNet-pytorch
    """

    def __init__(self, config: CobConfig = None):
        super().__init__()

        self.config = config if config is not None else CobConfig()

        in_channels = self.config.in_channels
        hidden_channels = self.config.hidden_channels

        self.block_1 = [Conv2d(in_channels, in_channels, 3, 2, 1)]
        self.block_1 += [Conv2d(in_channels, in_channels, 3, 1, 1) for _ in
                         range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(in_channels, in_channels, 3, 2, 1)]
        self.block_2 += [Conv2d(in_channels, in_channels, 3, 1, 1) for _ in
                         range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(in_channels, hidden_channels, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
                         for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 4, 0),
            nn.BatchNorm2d(hidden_channels))
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, 2, 2, 0),
            nn.BatchNorm2d(hidden_channels))
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels))

    def forward(self, x):
        x = self.block_1(x)
        x_skip_1 = x

        x = self.block_2(x)
        x_skip_2 = x

        x = self.block_3(x)

        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)

        x = torch.cat((x_0, x_1, x_2), 1)

        return x


class ScoreHead(nn.Module):
    """
    Produces object-ness scores for each anchor
    """

    def __init__(self, config: HeadConfig = None):
        super().__init__()
        self.config = config if config is not None else HeadConfig()

        self.eps = 1e-7

        self.conv2d = Conv2d(self.config.in_channels,
                             self.config.anchors_per_position, 1,
                             1, 0, activation=False, batch_norm=False)

    def forward(self, x):
        scores = self.conv2d(x)

        # Rearrange the output
        # (batch_size, anchors_per_pos, H, W) ->
        # (batch_size, H x W x anchors_per_pos)
        scores = scores.permute(0, 2, 3, 1)
        scores = scores.contiguous().view(scores.shape[0],
                                          scores.shape[1] * scores.shape[2] *
                                          scores.shape[3])

        scores = torch.sigmoid(scores)
        scores = scores.clamp(self.eps, 1. - self.eps)

        return scores


class RegressionHead(nn.Module):
    """
    Produces delta values for bbox localization from for each anchor
    """

    def __init__(self, config: HeadConfig = None):
        super().__init__()
        self.config = config if config is not None else HeadConfig()

        self.conv2d = Conv2d(self.config.in_channels,
                             7 * self.config.anchors_per_position,
                             1, 1, 0, activation=False, batch_norm=False)

    def forward(self, x):
        regs = self.conv2d(x)

        # Rearrange the output
        # (batch_size, 7 x anchors_per_pos, H,  W) ->
        # (batch_size, H x W x anchors_per_pos, 7)
        regs = regs.permute(0, 2, 3, 1)
        regs = regs.contiguous().view(regs.shape[0],
                                      regs.shape[1] * regs.shape[2] *
                                      self.config.anchors_per_position,
                                      7)

        return regs


class DirectionHead(nn.Module):
    """
    Produces a direction within the same heading-mod-pi for each anchor
    """

    def __init__(self, config: HeadConfig = None):
        super().__init__()
        self.config = config if config is not None else HeadConfig()

        self.eps = 1e-7

        self.conv2d = Conv2d(self.config.in_channels,
                             self.config.anchors_per_position,
                             1, 1, 0, activation=False, batch_norm=False)

    def forward(self, x):
        dirs = self.conv2d(x)

        # Rearrange the output
        # (batch_size, anchors_per_pos, H, W) ->
        # (batch_size, H x W x anchors_per_pos)
        dirs = dirs.permute(0, 2, 3, 1)
        dirs = dirs.contiguous().view(dirs.shape[0],
                                      dirs.shape[1] * dirs.shape[2] *
                                      dirs.shape[3])

        dirs = torch.sigmoid(dirs)
        dirs = dirs.clamp(self.eps, 1. - self.eps)

        return dirs


class ClassHead(nn.Module):
    """
    Produces class scores for each position
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

        # self.conv2d = Conv2d(self.config.in_channels,
        #                      self.config.num_classes, 1,
        #                      1, 0, activation=False,
        #                      batch_norm=False)

    def forward(self, x):
        # TODO: Fix ordering
        raise NotImplementedError


class RadarVoxelFusionNet(nn.Module):
    """
    The main network definition
    """

    def __init__(self, config: RvfnConfig = None):
        super().__init__()
        self.config = config if config is not None else RvfnConfig()

        # network elements
        self.svfeb = SVFEB(self.config.svfeb_config)

        if self.config.middle_block == 'sparse':
            self.cmb = SCMB(self.config.cmb_config)
        elif self.config.middle_block == 'normal':
            self.cmb = CMB(self.config.cmb_config)
        else:
            raise NotImplementedError('Middle block: ' +
                                      self.config.middle_block)
        self.cob = COB(self.config.cob_config)

        self.bbox_head = ScoreHead(self.config.head_config)
        self.reg_head = RegressionHead(self.config.head_config)
        self.dir_head = DirectionHead(self.config.head_config)

    @classmethod
    def sparse_to_dense(cls, voxels, features, spatial_size):
        """
        Converts sparse output of SVFEB to a dense feature tensor
        Args:
            voxels: (num_anchors_in_batch, 4) where the first 3
                numbers are voxel coordinates, and the 4th indicates the
                sample id in batch.
            features: (num_voxels, num_features) sparse features tensor
            spatial_size: (3,) Array indicating the target spatial size

        Returns: (batch_size, num_features, *spatial_size) dense feature tensor
        """
        num_features = features.shape[1]
        batch_size = voxels[-1][-1] + 1  # last sample id in batch + 1

        # (batch_size, num_features, D, H, W)
        dense_features = torch.zeros((batch_size, num_features, *spatial_size))

        dense_features = dense_features.to(voxels.device)

        # Only long, byte or bool tensors can be used as indices
        voxels = voxels.long()
        dense_features[voxels[:, 3], :, voxels[:, 0],
                       voxels[:, 1], voxels[:, 2]] = features

        return dense_features

    def forward(self, voxels, features):
        """
        Args:
            voxels: (num_anchors_in_batch, 4) where the first 3
                numbers are voxel coordinates, and the 4th indicates the
                sample id in batch.
            features: (num_anchors_in_batch, max_points_per_voxel,
                       num_voxel_channels) point features tensor

        Returns:
            bbox_scores: (num_anchors, 1) probability of objectness per anchor
            regs: (num_anchors, 7) bbox regression values
            dirs: (num_anchors, 1) direction prediction for each anchor
        """
        out = self.svfeb(torch.as_tensor(features))

        voxels[:, [0, 2]] = voxels[:, [2, 0]]  # swap x and z axes

        if self.config.middle_block == 'sparse':
            out = self.cmb([torch.as_tensor(voxels), out])
        elif self.config.middle_block == 'normal':
            spatial_size = self.config.cmb_config.input_spatial_size
            dense_features = self.sparse_to_dense(voxels, out, spatial_size)
            out = self.cmb(dense_features)
        else:
            raise NotImplementedError('Middle block: ' +
                                      self.config.middle_block)

        cob_conf = self.config.cob_config

        feature_map = self.cob(
            out.view(-1, cob_conf.in_channels, *cob_conf.in_spatial_size))

        bbox_scores = self.bbox_head(feature_map)
        regs = self.reg_head(feature_map)
        dirs = self.dir_head(feature_map)

        return bbox_scores, regs, dirs
