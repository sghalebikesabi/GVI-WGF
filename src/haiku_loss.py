import haiku as hk
import jax
import jax.numpy as jnp


def softmax_cross_entropy(preds, targets):
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(preds) * targets, axis=-1))


def mse_loss(preds, targets):
    return jnp.sum(jnp.mean(jnp.sum(jnp.square(preds - targets), axis=-1), -1))


LOSS_DICT = {
    "softmax_cross_entropy": softmax_cross_entropy,
    "mse": mse_loss,
}


def create_loss_fn(loss_fn_name, forward):
    def loss_fn(params: hk.Params, state: hk.State, features, targets):
        preds, state = forward.apply(
            params, state, None, inputs=features, is_training=True
        )
        loss = LOSS_DICT[loss_fn_name](preds, targets)
        return loss, (loss, state)

    return loss_fn
