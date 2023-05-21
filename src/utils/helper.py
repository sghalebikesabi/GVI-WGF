import random
import numpy as np
import jax
import torch


def make_deterministic(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_maybe_compile(framework):
    if framework == "haiku":
        return jax.jit
    # elif framework == "torch":
    #     return torch.compile
    else:
        return lambda f: f


class EarlyStopping:
    def __init__(
        self,
        patience=10,
        miniter=0,
        verbose=True,
        disable=False,
    ):
        self.patience = patience
        self.miniter = miniter
        self.verbose = verbose
        self.disable = disable

        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.best_params = None
        self.iter = 0

    def callback(self, params, score):
        if self.disable:
            return False

        self.iter += 1
        if self.iter <= self.miniter:
            return False

        self.last_score = score
        if score > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(params)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        self.best_params = model.detach().clone()
