from rvfn.config.config import Config


class EvalConfig(Config):
    """
    Configuration parameters for model evaluation
    """

    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'distance_thresholds': [0.5, 1, 2, 4],  # as per official Nuscenes

            'data_loader_workers': 6,
            'batch_size': 4,
            'device_ids': [0]  # IDs of the GPUs to use
        }
