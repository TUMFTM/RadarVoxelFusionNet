from rvfn.config.config import Config


class InferConfig(Config):
    """
    Configuration parameters for model evaluation
    """

    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'nms_threshold': 0.1,  # IoU threshold for NMS
            'min_confidence': 0.001,
        }
