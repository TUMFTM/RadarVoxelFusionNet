"""
Common interface for loading datasets from files
"""

import pickle5 as pickle
import json
from os import path


class SerializedDataset:
    def __init__(self, dataset_path: str, samples_filepath: str):
        """
        Args:
            dataset_path: Dataset directory path
            samples_filepath: Path to a file containing a json array of sample
             tokens.
        """
        self.dataset_path = dataset_path

        with open(samples_filepath, 'r') as f:
            self.samples = json.load(f)

    def __getitem__(self, idx):
        file_path = path.join(self.dataset_path,
                              self.samples[idx]) + '.pickle'

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.samples)
