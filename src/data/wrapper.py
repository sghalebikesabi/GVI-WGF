import jax.numpy as jnp
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset
from typing import Mapping
import random


Batch = Mapping[str, np.ndarray]


class AbstractDataset(Dataset):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

        if self.x is not None:
            self.n_samples = len(self.x)
        else:
            self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.y[idx]
        # x_i = np.array(self.x[idx], dtype=jnp.float32)
        # y_i = np.array(self.y[idx], dtype=jnp.float32)
        return x_i, y_i


def _numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.array(torch.stack(batch), dtype=jnp.float32)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [_numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch, dtype=jnp.float32)


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        seed=1,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        g = torch.Generator()
        g.manual_seed(seed)

        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn if worker_init_fn else _seed_worker,
        )
