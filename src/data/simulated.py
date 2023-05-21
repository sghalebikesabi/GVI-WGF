import jax.numpy as jnp
import numpy as np
import torch
import jax

import math

from .wrapper import AbstractDataset


def toy_repulsive_paper(n_samples, low, high, std):
    x = torch.linspace(low, high, n_samples).unsqueeze(-1)
    y = x * np.sin(x)
    if std:
        y += np.random.normal(loc=0.0, scale=std, size=(n_samples, 1))
    return x, y


def linear_curve(n_samples, low, high, std=0.5):
    """From https://github.com/MrHuff/GWI/blob/main/simulate_data/unit_test_data.py."""
    x = torch.linspace(low, high, n_samples)
    y = 3 * x + torch.randn_like(x) * std
    return x.unsqueeze(-1), y.unsqueeze(-1)


def sim_sin_curve_1(n_samples, low, high, std=0.5):
    """From https://github.com/MrHuff/GWI/blob/main/simulate_data/unit_test_data.py."""
    x = torch.linspace(low, high, n_samples)
    y = torch.sin(x) + torch.randn_like(x) * std + 0.1 * x
    return x.unsqueeze(-1), y.unsqueeze(-1)


def sim_sin_curve_2(n_samples, low, high, std=0.5):
    """From https://github.com/MrHuff/GWI/blob/main/simulate_data/unit_test_data.py."""
    x = torch.linspace(low, high, n_samples)
    y = torch.sin(x) + torch.randn_like(x) * std + 0.1 * x**2
    return x.unsqueeze(-1), y.unsqueeze(-1)


def sim_sin_curve_3(n_samples, low, high, std=0.5):
    """From https://github.com/MrHuff/GWI/blob/main/simulate_data/unit_test_data.py."""
    x = torch.linspace(low, high, n_samples)
    y = (
        torch.sin(x * 3 * 3.14)
        + 0.3 * torch.cos(x * 9 * 3.14)
        + 0.5 * torch.sin(x * 7 * 3.14)
    )
    y = y + torch.randn_like(x) * std
    return x.unsqueeze(-1), y.unsqueeze(-1)


def gmm(n_samples=1000, num_clusters=2, mean_mult=3.0):
    y = np.random.choice(num_clusters, size=(n_samples,))
    x = (
        np.random.normal(
            size=(n_samples, 2),
        )
        + 1
        - y[:, None] * mean_mult
    )
    y = jax.nn.one_hot(y, num_clusters)
    return x, y


_SIM_DICT = {
    "linear": linear_curve,
    "sin1": sim_sin_curve_1,
    "sin2": sim_sin_curve_2,
    "sin3": sim_sin_curve_3,
    "gmm": gmm,
    "toy_repulsive_paper": toy_repulsive_paper,
}


class SimulatedDataset(AbstractDataset):
    def __init__(
        self,
        name,
        low,
        high,
        n_samples,
        drop_ranges=[],
        n_kernel_samples=0,
        kernel_distr=None,
        seed=None,
        split=None,
        error_split=None,
        **data_kwargs
    ):
        del seed
        del split
        del error_split

        keep_ratio = 1.0 - (
            sum([r[1] - r[0] for r in drop_ranges]) / (high - low)
        )

        self.x, self.y = _SIM_DICT[name](
            n_samples=math.floor(n_samples / keep_ratio),
            low=low,
            high=high,
            **data_kwargs
        )

        dropped_obs = []
        for range in drop_ranges:
            drop_idx = (self.x[:, 0] >= range[0]) & (self.x[:, 0] < range[1])
            dropped_obs.append((self.x[drop_idx], self.y[drop_idx]))
            self.x = self.x[~drop_idx]
            self.y = self.y[~drop_idx]

        if dropped_obs:
            self.dropped_x = np.concatenate([x for x, _y in dropped_obs], 0)
            self.dropped_y = np.concatenate([y for _x, y in dropped_obs], 0)

        if n_kernel_samples > 0:
            if kernel_distr == "train":
                n_kernel_samples = math.ceil(n_kernel_samples / keep_ratio)
                kernel_drop_ranges = drop_ranges

            elif kernel_distr == "test":
                n_kernel_samples = math.ceil(
                    n_kernel_samples / (1 - keep_ratio)
                )
                kernel_drop_ranges = []
                if low - drop_ranges[0][0] > 0:
                    kernel_drop_ranges.append((low, drop_ranges[0][0]))

                last_high = drop_ranges[0][1]
                for range in drop_ranges:
                    kernel_drop_ranges.append((last_high, range[0]))
                    last_high = range[1]

                if high - last_high > 0:
                    kernel_drop_ranges.append((last_high, high))

            elif kernel_distr == "valid":
                kernel_drop_ranges = []

            else:
                raise ValueError("Unknown kernel_drop_range_descr")

            self.kernel_samples = AbstractDataset()
            kernel_x, _ = _SIM_DICT[name](
                n_samples=n_kernel_samples, low=low, high=high, **data_kwargs
            )
            for range in kernel_drop_ranges:
                drop_idx = (kernel_x[:, 0] >= range[0]) & (
                    kernel_x[:, 0] < range[1]
                )
                kernel_x = kernel_x[~drop_idx]

            self.kernel_samples.x = np.array(kernel_x, dtype=jnp.float32)
        self.n_samples = len(self.x)
