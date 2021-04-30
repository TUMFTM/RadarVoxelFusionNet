"""
Provides a function for combining multiple samples into a batch
"""

import torch


def collate(batch):
    """
    Combines multiple samples from the dataset into a batch.
    Args:
        batch: A list of outputs from NuscenesDataset

    Returns:
        voxels, features, ground-truth bboxes, labels, and sample tokens for
            the batch

    """
    voxels = [sample['voxels'] for sample in batch]
    for idx, sample_voxels in enumerate(voxels):
        sample_idx_vector = torch.full((sample_voxels.shape[0], 1), idx,
                                       dtype=torch.int)
        voxels[idx] = torch.cat((sample_voxels, sample_idx_vector), dim=1)
    voxels = torch.cat(voxels)

    features = [sample['features'] for sample in batch]
    features = torch.cat(features)

    bboxes = [sample['bboxes'] for sample in batch]
    labels = [sample['labels'] for sample in batch]
    tokens = [sample['sample_token'] for sample in batch]

    return voxels, features, bboxes, labels, tokens
