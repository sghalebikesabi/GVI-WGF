from typing import Sequence, Mapping, Any

import haiku as hk
import jax
import jax.numpy as jnp

import utils


class DefaultUpdateRule:
    def __init__(
        self,
        loss_fn_grad,
        init_params_var,
        keys,
        lr,
        langevin_reg_param,
        kernel_reg_param,
        forward,
    ):
        del init_params_var
        del langevin_reg_param
        del kernel_reg_param
        del keys
        del forward

        self.lr = lr
        self.loss_fn_grad = loss_fn_grad

    def single_param_step(self, param, update):
        return param - self.lr * update

    def step(self, ensemble_train_state, inputs, targets):
        grads, (loss, state) = self.loss_fn_grad(
            ensemble_train_state["params"],
            ensemble_train_state["state"],
            inputs,
            targets,
        )

        updated_params = jax.tree_util.tree_map(
            self.single_param_step, ensemble_train_state["params"], grads
        )

        return {"params": updated_params, "state": state}, {"loss": loss}


class LangevinUpdateRule(DefaultUpdateRule):
    def __init__(
        self,
        loss_fn_grad,
        init_params_var,
        keys,
        lr,
        langevin_reg_param,
        kernel_reg_param,
        forward,
    ):

        super().__init__(
            loss_fn_grad,
            init_params_var,
            keys,
            lr,
            langevin_reg_param,
            kernel_reg_param,
            forward,
        )
        self.keys = keys
        self.init_params_var = init_params_var

        self.lr_langevin = lr * langevin_reg_param
        self.sqrt_2_lr_langevin = jnp.sqrt(2 * self.lr_langevin)

    def single_default_update_rule(self, param, update):
        return param - self.lr * update

    def single_param_step(self, param, update, init_param_var, key):
        reg_term = self.lr_langevin * param / init_param_var
        noise_term = self.sqrt_2_lr_langevin * jax.random.normal(key, shape=param.shape)
        return param - self.lr * update + reg_term + noise_term

    def step(self, ensemble_train_state, inputs, targets):
        self.keys, training_key = jax.random.split(self.keys)
        training_key = utils.random_split_like_tree(
            training_key, ensemble_train_state["params"]
        )

        grads, (loss, state) = self.loss_fn_grad(
            ensemble_train_state["params"],
            ensemble_train_state["state"],
            inputs,
            targets,
        )

        updated_params = jax.tree_util.tree_map(
            self.single_param_step,
            ensemble_train_state["params"],
            grads,
            self.init_params_var,
            training_key,
        )

        return {"params": updated_params, "state": state}, {"loss": loss}


def repulsive_loss_fn(
    params: hk.Params,
    state: hk.State,
    other_train_states: Sequence[Mapping[str, Any]],
    inputs,
    forward,
):
    raise NotImplementedError(
        "This is not implemented for ensemble_train_state when not used in combinaion with ensemble loop"
    )
    # TODO: check if vectorisation saves compute
    # TODO: do this once for all models
    loss = 0
    preds, state = forward.apply(params, state, None, inputs=inputs, is_training=True)
    for other_train_state in other_train_states:
        _other_pred, state = forward.apply(
            other_train_state["params"],
            other_train_state["state"],
            None,
            inputs=inputs,
            is_training=False,
        )
        loss += utils.rbf_kernel(preds, _other_pred)
    loss /= len(other_train_states)

    return loss, (loss, state)


def repulsive_loss_fn_given_other_preds(
    params: hk.Params,
    state: hk.State,
    other_preds,
    inputs,
    forward,
):
    # TODO: check if vectorisation saves compute
    loss = 0
    preds, state = forward.apply(params, state, None, inputs=inputs, is_training=True)
    for other_pred in other_preds:
        loss += utils.rbf_kernel(preds, other_pred)
    loss /= len(other_preds)

    return loss, (loss, state)


def repulsive_langevin_update_rule(
    loss_fn_grad,
    ensemble_train_state,
    model_idx,
    init_params_var,
    keys,
    lr,
    langevin_reg_param,
    kernel_reg_param,
    inputs,
    targets,
    forward,
    prior_kernel_preds,
    kernel_samples,
):
    raise NotImplementedError(
        "This is not implemented for ensemble_train_state when not used in combinaion with ensemble loop"
    )

    grads, (loss, state) = loss_fn_grad(
        ensemble_train_state[model_idx]["params"],
        ensemble_train_state[model_idx]["state"],
        inputs,
        targets,
    )

    kernel_grads, (kernel_loss, _) = jax.grad(repulsive_loss_fn, has_aux=True)(
        ensemble_train_state[model_idx]["params"],
        ensemble_train_state[model_idx]["state"],
        [
            train_state
            for j, train_state in enumerate(ensemble_train_state)
            if j != model_idx
        ],
        kernel_samples.x,
        forward,
    )
    prior_kernel_grads, (prior_kernel_loss, _) = jax.grad(
        repulsive_loss_fn_given_other_preds, has_aux=True
    )(
        ensemble_train_state[model_idx]["params"],
        ensemble_train_state[model_idx]["state"],
        prior_kernel_preds,
        kernel_samples.x,
        forward,
    )

    def single_repulsive_langevin_update_rule(
        param, update, kernel_grad, prior_kernel_grad, init_param_var, key
    ):
        reg_term = lr * langevin_reg_param * param / init_param_var
        noise_term = jnp.sqrt(2 * lr * langevin_reg_param) * jax.random.normal(
            key, shape=param.shape
        )
        return (
            param
            - lr * update
            + lr * kernel_reg_param * prior_kernel_grad
            - lr * kernel_reg_param * kernel_grad
            + reg_term
            + noise_term
        )

    updated_params = jax.tree_util.tree_map(
        single_repulsive_langevin_update_rule,
        ensemble_train_state[model_idx]["params"],
        grads,
        kernel_grads,
        prior_kernel_grads,
        init_params_var,
        keys,
    )

    loss_dict = {
        "total_loss": loss + kernel_loss + prior_kernel_loss,
        "loss": loss,
        "kernel_loss": kernel_loss,
        "prior_kernel_loss": prior_kernel_loss,
    }

    return {"params": updated_params, "state": state}, {"loss": loss}


_UPDATE_RULES = {
    "standard": DefaultUpdateRule,
    "langevin": LangevinUpdateRule,
    "repulsive": repulsive_langevin_update_rule,
}


def create_update_rule(
    method,
    loss_fn,
    init_params_var,
    keys,
    lr,
    langevin_reg_param,
    kernel_reg_param,
    optimizer,
    forward,
):
    """
    Utils for the creation of the SVGD method
    """
    del optimizer

    loss_fn_grad = jax.grad(loss_fn, has_aux=True)

    method = _UPDATE_RULES[method](
        loss_fn_grad,
        init_params_var,
        keys,
        lr,
        langevin_reg_param,
        kernel_reg_param,
        forward,
    )

    return method


def maybe_create_optimizer(ensemble_train_state, lr, optimizer):
    return None
