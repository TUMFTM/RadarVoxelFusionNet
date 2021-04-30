from rvfn.config.config import Config
from rvfn.config.loss_config import LossConfig


class TrainingConfig(Config):
    """
    Configuration parameters for setting up dataset, network and training.
    """

    @classmethod
    def get_default_conf(cls) -> dict:
        return {
            'loss_config': LossConfig(),

            # Do on-the-fly augmentations to training set
            'augment': True,

            'batch_size': 4,
            'optimizer': 'AMSGrad',
            'learning_rate': 0.001,
            'epsilon': 1e-8,  # Epsilon parameter for optimizer
            'weight_decay': 0.0,
            'epochs': 1000,
            # Path to directory where the checkpoints and tensorboard logs
            #  are written to
            'out_path': '/persistent_storage/out/',
            'log_interval': 500,  # log the loss every this many batches
            'checkpoint_interval': 4,  # save every this many epochs
            'eval_interval': 4,
            # If larger than zero, evaluation will also be run on this fraction
            #  of the training set, in addition to the validation set.
            'eval_on_train_fraction': 0.1,

            'data_loader_workers': 6,
            'device_ids': [0]  # IDs of the GPUs to use
        }

    def verify(self):
        if not 0 <= self.eval_on_train_fraction <= 1:
            return False

        return True
