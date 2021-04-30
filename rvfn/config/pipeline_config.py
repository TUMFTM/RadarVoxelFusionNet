"""
Contains the combination of configs for the prediction pipeline.
Including the dataset preprocessing, model, and target configs.
"""

from rvfn.config.config import Config
from rvfn.config.dataset_config import NuscenesDatasetConfig
from rvfn.config.model_config import RvfnConfig
from rvfn.config.target_config import LossTargetGeneratorConfig
from rvfn.config.infer_config import InferConfig


class PipelineConfig(Config):
    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'dataset_config': NuscenesDatasetConfig(),
            'model_config': RvfnConfig(),
            'target_config': LossTargetGeneratorConfig(),
            'infer_config': InferConfig()
        }

    def verify(self):
        cob_input_size = self.model_config.cob_config.in_spatial_size
        cob_output_size = [ax / 2 for ax in cob_input_size]

        anchor_feature_map_shape = \
            self.target_config.anchor_config.feature_map_shape

        if cob_output_size != list(anchor_feature_map_shape):
            return False

        return True
