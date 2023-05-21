"""UCI Datasets.

This file contains the code to load the UCI datasets. Mostly adapted from https://github.com/MrHuff/GWI/blob/7321cc7a3ae6566d9c1f145a65d7707a00d6384f/utils/regression_dataloaders.py.
"""
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import urllib.request
import zipfile
from sklearn.linear_model import LinearRegression

import utils
from .wrapper import AbstractDataset


class UCIDataset(AbstractDataset):
    """UCI Datasets.

    # TODO: Each dataset is split into 20 train-test folds, except for the protein dataset which uses 5 folds and the Year Prediction MSD dataset which uses a single train-test split.

    """

    def __init__(
        self,
        name,
        seed,
        split,
        n_kernel_samples=0,
        kernel_distr=None,
        error_split="random",
    ):
        self.name = name
        self.seed = seed

        self.datasets = {
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            "boston": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "naval": "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
            "KIN8NM": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
            "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        }
        self.data_path = utils.get_data_root() + "/"

        data = self._load_dataset()
        data = data[:, np.std(data, 0) > 0]

        self.in_dim = data.shape[1] - 1
        self.out_dim = 1

        self.x, self.y = self._get_split(data, split, error_split)
        self.n_samples = len(self.x)

        if n_kernel_samples > 0:
            self.kernel_samples = AbstractDataset()
            kernel_x, _ = self._get_split(data, kernel_distr)
            kernel_x = kernel_x[
                np.random.permutation(np.arange(len(kernel_x)))
            ][:n_kernel_samples]
            self.kernel_samples.x = np.array(kernel_x, dtype=jnp.float32)

    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not os.path.exists(self.data_path + "UCI"):
            os.mkdir(self.data_path + "UCI")

        url = self.datasets[self.name]
        file_name = url.split("/")[-1]

        if not os.path.exists(self.data_path + "UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path + "UCI/" + file_name
            )

        if self.name == "boston":
            data = pd.read_csv(
                self.data_path + "UCI/housing.data",
                header=None,
                delimiter="\s+",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "concrete":
            data = pd.read_excel(
                self.data_path + "UCI/Concrete_Data.xls", header=0
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "energy":
            data = pd.read_excel(
                self.data_path + "UCI/ENB2012_data.xlsx", header=0
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "power":
            zipfile.ZipFile(self.data_path + "UCI/CCPP.zip").extractall(
                self.data_path + "UCI/CCPP/"
            )
            data = pd.read_excel(
                self.data_path + "UCI/CCPP/CCPP/Folds5x2_pp.xlsx", header=0
            ).values
            np.random.shuffle(data)

        elif self.name == "KIN8NM":
            data = pd.read_csv(
                self.data_path + "UCI/dataset_2175_kin8nm.csv", header=0
            )
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "naval":
            zipfile.ZipFile(
                self.data_path + "UCI/UCI%20CBM%20Dataset.zip"
            ).extractall(self.data_path + "UCI/UCI CBM Dataset/")
            data = pd.read_csv(
                self.data_path + "UCI/UCI CBM Dataset/UCI CBM Dataset/data.txt",
                header=None,
                delimiter="\s+",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "protein":
            data = pd.read_csv(
                self.data_path + "UCI/CASP.csv", header=0, delimiter=","
            ).iloc[:, ::-1]
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "wine":
            data = pd.read_csv(
                self.data_path + "UCI/winequality-red.csv",
                header=0,
                delimiter=";",
            )
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            data = pd.read_csv(
                self.data_path + "UCI/yacht_hydrodynamics.data",
                header=None,
                delimiter="\s+",
            )
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        data = data.astype("float")

        return data

    def _get_split(self, data, split, error_split="random"):
        if error_split == "random":
            train_valid_idx, test_idx = train_test_split(
                np.arange(len(data)),
                test_size=0.1,
                random_state=self.seed,
                shuffle=True,
            )

        elif error_split == "pred_error":
            model = LinearRegression().fit(
                data[:, : self.in_dim], data[:, self.in_dim :]
            )
            err = model.predict(data[:, : self.in_dim]) - data[:, self.in_dim :]
            sort_idx = np.argsort(err.squeeze() ** 2)
            train_valid_idx = sort_idx[: int(0.9 * len(data))]
            test_idx = sort_idx[int(0.9 * len(data)) :]

        elif error_split == "norm_error":
            mean = data[:, : self.in_dim].mean(0)
            err = data[:, : self.in_dim] - mean
            sort_idx = np.argsort(np.linalg.norm(err, axis=1))
            train_valid_idx = sort_idx[: int(0.9 * len(data))]
            test_idx = sort_idx[int(0.9 * len(data)) :]

        x_train, y_train = (
            data[train_valid_idx, : self.in_dim],
            data[train_valid_idx, self.in_dim :],
        )

        x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
        y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5

        x_stds[x_stds == 0] = 1.0
        y_stds[y_stds == 0] = 1.0

        if split == "train_valid":
            x = (x_train - x_means) / x_stds
            y = (y_train - y_means) / y_stds

        elif split == "test":
            x_test, y_test = (
                data[test_idx, : self.in_dim],
                data[test_idx, self.in_dim :],
            )
            x = (x_test - x_means) / x_stds
            y = (y_test - y_means) / y_stds

        elif split == "train":
            x = (x_train - x_means) / x_stds
            y_train = (y_train - y_means) / y_stds
            x, _, y, _ = train_test_split(x, y_train, test_size=0.1)

        elif split == "valid":
            x = (x_train - x_means) / x_stds
            y_train = (y_train - y_means) / y_stds
            _, x, _, y = train_test_split(x, y_train, test_size=0.1)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y
