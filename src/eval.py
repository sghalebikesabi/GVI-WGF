import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import KernelDensity
import torch
from typing import Mapping, Sequence

Scalars = Mapping[str, jnp.ndarray]

# TODO: Compute loss (Eq. 18) at evaluation time
# TODO: Think of evaluation metric that is better at tracking better performance for uncertainty quantification
# TOD0: confidence threshold vs accuracy plot


def accuracy(preds, targets):
    predicted_label = jnp.argmax(preds, axis=-1)
    target_label = jnp.argmax(targets, axis=-1)
    correct = jnp.sum(jnp.equal(predicted_label, target_label))
    return correct.astype(jnp.float32)


def mse_loss(preds, targets):
    # sum instead of mean to be consistent with accuracy (in contrast to utils.mse_loss)
    return ((preds - targets) ** 2).sum() / preds.shape[0]


def gaussian_nll(preds, targets):
    var_values = preds.var(axis=0)
    y_diff_2 = (targets - preds.mean(axis=0)) ** 2
    return torch.sum(
        0.5 * torch.log(var_values)
        + 0.5 * y_diff_2 / var_values
        + 0.5 * np.log(2 * np.pi)
    )


def kde_nll(preds, targets):
    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    sum_nll = 0
    for i in range(preds.shape[1]):
        kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(preds[:, i])
        sum_nll -= kde.score(targets[i : i + 1])
    return sum_nll


def brier_score(preds, targets):
    return ((preds - targets) ** 2).sum() / preds.shape[0]


def softmax_cross_entropy(preds, targets):
    return -torch.mean(
        torch.sum(torch.nn.functional.log_softmax(preds) * targets, axis=-1)
    )


def predictive_variance(preds, targets):
    return preds.var(axis=0).mean()


_EVAL_METRICS = {
    "accuracy": accuracy,
    "mse": mse_loss,
    "predictive_variance": predictive_variance,
    "mse_of_mean": lambda preds, targets: mse_loss(
        preds.mean(axis=0)[None], targets
    ),
    "gaussian_nll": gaussian_nll,
    "kde_nll": kde_nll,
    "categorical_nll": softmax_cross_entropy,
}

_EVAL_METRICS_EARLY_STOP = {
    "accuracy": lambda x: -x,
    "mse": lambda x: x,
    "predictive_variance": lambda x: -x,
    "mse_of_mean": lambda x: x,
    "brier": lambda x: x,
    "gaussian_nll": lambda x: x,
    "kde_nll": lambda x: x,
    "categorical_nll": lambda x: x,
    "None": lambda _: "not interesting",
}


def evaluate(
    eval_dataloader,
    pred_fn,
    metrics: Sequence[str],
    early_stopping_metric: str,
    maybe_compile: Mapping,
) -> Scalars:
    """Evaluates the model at the given params/state.

    Args:
        eval_dataloader: Dataloader for evaluation.
        pred_fn: Function that takes in inputs and returns predictions (one column per network in ensemble).
        eval_metric: Metric to evaluate.
        enable_jit: Whether to enable JIT compilation.
    """

    @maybe_compile
    def eval_batch(inputs, targets, eval_metric) -> jnp.ndarray:
        """Evaluates a batch."""
        ensemble_preds = pred_fn(inputs=inputs)
        eval_value = _EVAL_METRICS[eval_metric](ensemble_preds, targets)
        return eval_value

    # Params/state are sharded per-device during training. We just need the copy
    # from the first device (since we do not pmap evaluation at the moment).
    # params, state = jax.tree_util.tree_map(lambda x: x[0], (params, state))
    correct = {metric: 0 for metric in metrics}
    total = 0
    for inputs, targets in eval_dataloader:
        for metric in metrics:
            correct[metric] += eval_batch(inputs, targets, metric)
        total += targets.shape[0]
    eval_metric_vals = {k: v / total for k, v in correct.items()}
    eval_metric_vals["None"] = None

    eval_metric_vals.update(
        {
            "r" + k: torch.sqrt(v)
            for k, v in eval_metric_vals.items()
            if k[:3] == "mse"
        }
    )
    return (
        eval_metric_vals,
        _EVAL_METRICS_EARLY_STOP[early_stopping_metric](
            eval_metric_vals[early_stopping_metric]
        ),
    )
