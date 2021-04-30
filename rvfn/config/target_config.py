from rvfn.config.config import Config
import numpy as np


class AnchorGeneratorConfig(Config):
    """
    Config parameters for anchor generator
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # Feature map shape of the COB. We produce several anchors for
            #  each position in this shape.
            #                      y    x
            'feature_map_shape': (100, 124),
            # Size, and rotation of produced anchors. Anchor generator will
            #  create len(anchor_sizes) * len(anchor_rotations) anchors per
            #  position in the feature map
            #                  w    l    h
            'anchor_sizes': [(1.92, 4.62, 1.69)],
            # Rotations must be in range [0, pi)
            'anchor_rotations': [0, np.pi / 2],

            # The ranges in which the anchors will be spread
            'anchor_range': (
                (0, 50),  # x
                (0, 40),  # y
            ),
            # The center of anchors in z axis
            'center_z': 2.0  # meters
        }

    @property
    def anchors_per_position(self):
        return len(self.anchor_sizes) * len(self.anchor_rotations)


class LossTargetGeneratorConfig(Config):
    """
    Config parameters for creating loss targets from network output
    """
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            # IoU threshold above which an anchor is considered to match with
            #  a ground-truth bbox
            'pos_threshold': 0.35,
            # IoU threshold below which an anchor is considered to not match
            #  with a ground-truth bbox
            'neg_threshold': 0.3,
            # If not None, an anchor is considered a match if its center is
            #  within this distance of the ground_truth box.
            # This is applied after the IoU filter. An anchor will be positive
            #  if (IoU > pos_threshold) AND (dist < pos_dist_threshold),
            #  negative if (IoU < neg_threshold), and ignored otherwise.
            'pos_dist_threshold': 0.5,

            # Config for anchor generation
            'anchor_config': AnchorGeneratorConfig(),

            # Use sine of yaw difference as regression targets
            'sine_yaw_targets': True
        }
