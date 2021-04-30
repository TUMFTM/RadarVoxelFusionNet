from rvfn.config.config import Config


class FocalLossConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'gamma': 0.0,
            'alphas': [0.5, 0.5],  # Per-class alpha value
            'reduce': False,
            'logits': False  # whether the inputs are logits
        }


class BinaryFocalLossConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'gamma': 0.0,
            'alpha': 0.6,
            'reduce': False,
            'logits': False  # whether the inputs are logits
        }


class ScoreLossConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'criteria': 'binary_focal',
            'criteria_config': BinaryFocalLossConfig(),

            # Divide the loss for positive and negative anchors by their
            #  respective numbers before adding them together.
            'normalize_by_type': True
        }


class RegLossConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'criteria': 'smoothL1'
        }


class DirLossConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'criteria': 'binary_cross_entropy'
        }


class LossConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'score_loss_config': ScoreLossConfig(),
            'reg_loss_config': RegLossConfig(),
            'dir_loss_config': DirLossConfig(),
            'score_loss_weight': 2.0,
            'reg_loss_weight': 1.0,
            'dir_loss_weight': 0.2
        }
