import torch.utils.data as data

import numpy as np

import torch

from typing import List

class ActiveLearningData:
    """Splits `dataset` into an active dataset and an available dataset."""

    dataset: data.Dataset
    training_dataset: data.Dataset
    pool_dataset: data.Dataset
    training_mask: np.ndarray
    pool_mask: np.ndarray

    def __init__(self, 
                dataset: data.Dataset):

        super().__init__()

        self.dataset = dataset
        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        self.training_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)

        self._update_indices()
        
    @property
    def train_size(self):
        return np.count_nonzero(self.training_mask)
        
    @property
    def pool_size(self):
        return np.count_nonzero(self.pool_mask)

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def get_dataset_indices(self, pool_indices: List[int]) -> List[int]:
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """Acquire elements from the pool dataset into the training dataset.

        Add them to training dataset & remove them from the pool dataset."""

        self.training_mask[pool_indices] = True
        self.pool_mask[pool_indices] = False
        self._update_indices()

    def remove_from_pool(self, pool_indices):
        indices = self.get_dataset_indices(pool_indices)

        self.pool_mask[indices] = False
        self._update_indices()

    def get_random_pool_indices(self, size:int) -> torch.LongTensor:
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = np.random.permutation(self.pool_dataset.indices)[:size]
        return pool_indices

    def extract_dataset_from_pool(self, size:int) -> data.Dataset:
        """Extract a dataset randomly from the pool dataset and make those indices unavailable.

        Useful for extracting a validation set."""
        return self.extract_dataset_from_pool_from_indices(self.get_random_pool_indices(size))

    def extract_dataset_from_pool_from_indices(self, pool_indices) -> data.Dataset:
        """Extract a dataset from the pool dataset and make those indices unavailable.
            Useful for extracting a validation set.
        """
        dataset_indices = self.get_dataset_indices(pool_indices)

        self.remove_from_pool(pool_indices)
        return data.Subset(self.dataset, dataset_indices)

        


