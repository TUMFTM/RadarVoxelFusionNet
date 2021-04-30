from rvfn.config.config import Config


class SvfebConfig(Config):
    """
    Config parameters for the Stacked Voxel Feature Encoder module
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # Voxel-grid point channels
            # Each voxel has 3 global and 3 local coordinates for a total of 6.
            # Other features such as lidar intensity or radar RCS must be
            #  added to this number to yield the total in_channels.
            'in_channels': 7,
            # same as voxelnet
            'hidden_channels': 32,
            'out_channels': 128,
        }


class CmbConfig(Config):
    """
    Config parameters for the Convolutional Middle Block
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # The spatial sizes must be accurately set in case sparseConv is
            #  used for the middle block.
            # The z dimension is chosen such that it will be down-sampled to
            #  3 after the CMB module. The other two must be divisible by eight
            #  after the CMB and big enough to accommodate all input voxels.
            # The down-sampled z dimension's size multiplied by the output
            #  channels must equal the input channels of the RPN.
            #                      z    y    x
            'input_spatial_size': (11, 204, 252),
            # The expected spatial size after the SCB. This is only used for
            #  verification.
            'expected_output_spatial_size': (3, 200, 248),
            'in_channels': 128,
            'out_channels': 64
        }


class CobConfig(Config):
    """
    Config parameters for the fully convolutional output block
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # = ScbConfig.output_spatial_size[0] * ScbConfig.output_channels
            'in_channels': 192,

            'hidden_channels': 256,  # similar to VoxelNet
            # consequence of SCB input spatial size
            #                    y    x
            'in_spatial_size': (200, 248)
        }


class HeadConfig(Config):
    """
    Config for the score prediction head
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'num_classes': 2,
            'anchors_per_position': 2,
            'in_channels': 768  # = CobConfig.hidden_channels * 3
        }


class RvfnConfig(Config):
    """
    Network config parameters
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'svfeb_config': SvfebConfig(),
            'cmb_config': CmbConfig(),
            'cob_config': CobConfig(),
            'head_config': HeadConfig(),
            'middle_block': 'sparse',
        }

    def verify(self):
        # -- Verify spatial sizes and channels match correctly -- #

        cmb_out_size = self.cmb_config.expected_output_spatial_size
        cmb_out_channels = self.cmb_config.out_channels
        cob_in_size = self.cob_config.in_spatial_size
        cob_in_channels = self.cob_config.in_channels
        cob_hidden_channels = self.cob_config.hidden_channels
        head_in_channels = self.head_config.in_channels

        # SCB -> CB
        if cmb_out_size[1:] != cob_in_size:
            return False
        if cmb_out_size[0] * cmb_out_channels != cob_in_channels:
            return False

        # CB -> heads
        if head_in_channels != cob_hidden_channels * 3:
            return False

        return True
