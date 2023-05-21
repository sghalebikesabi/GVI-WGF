import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections

from typing import Optional


class Mlp(hk.Module):
    def __init__(
        self,
        n_output=1,
        name: Optional[str] = None,
        init: ml_collections.ConfigDict = None,
        hidden_nodes=[200],
    ):
        super().__init__(name=name)

        self.weights_init = hk.initializers.VarianceScaling(
            init.weight_init_var, "fan_in", "normal"
        )
        # self.bias_init = hk.initializers.RandomNormal(jnp.sqrt(init.bias_init_var))
        self.bias_init = hk.initializers.VarianceScaling(
            init.bias_init_var, "fan_in", "normal"
        )

        layers = []
        for h in hidden_nodes:
            layers += [
                hk.Linear(
                    h,
                    w_init=self.weights_init,
                    b_init=self.bias_init,
                ),
                jax.nn.relu,
            ]
        layers += [
            hk.Linear(
                n_output,
                w_init=self.weights_init,
                b_init=self.bias_init,
            ),
        ]
        self.net = hk.Sequential(layers)

    def __call__(self, inputs, is_training):
        del is_training
        return self.net(inputs)


def init_value_mapping(key, param, init):
    if isinstance(param, dict):
        return {k: init_value_mapping(k, v, init) for k, v in param.items()}
    elif key == "w":
        return init.weight_init_var / param.shape[0]
    elif key == "b":
        return init.bias_init_var
    else:
        raise NotImplementedError("Unknown key: {}".format(key))


_MODELS_DICT = {
    "mlp": Mlp,
}


def model_fn(model_type, inputs, is_training, model_args):
    return _MODELS_DICT[model_type](**model_args)(inputs, is_training)


class AllEnsembleForward:
    def __init__(self, model_type, n_models, model_args):
        self.model_type = model_type
        self.n_models = n_models
        self.model_args = model_args

        self.transformed_forward = hk.transform_with_state(self.forward)

    def forward(self, inputs, is_training):
        model_fn_lst = [
            _MODELS_DICT[self.model_type](name=f"net_{i}", **self.model_args)
            for i in range(self.n_models)
        ]
        return jnp.stack(
            [model_fn(inputs, is_training) for model_fn in model_fn_lst],
        )

    def apply(self, *args, **kwargs):
        return jax.jit(self.transformed_forward.apply)(*args, **kwargs)

    def init(self, *args, **kwargs):
        params, state = self.transformed_forward.init(*args, **kwargs)
        return {"params": params, "state": state}

    def init_value_mapping(self, ensemble_train_state, init):
        return init_value_mapping("", ensemble_train_state["params"], init)


class MeanEnsembleForward(AllEnsembleForward):
    def __init__(self, model_type, n_models, model_args):
        super().__init__(model_type, n_models, model_args)

        self.transformed_forward = hk.transform_with_state(self.forward)

    def forward(self, inputs, is_training):
        all_preds = super().forward(inputs, is_training)
        return jnp.mean(all_preds, axis=0)


def mean_ensemble_predict(ensemble_train_state, forward, inputs):
    ensemble_preds = 0
    for train_state in ensemble_train_state:
        preds, _ = forward.apply(
            train_state["params"],
            train_state["state"],
            None,
            inputs=inputs,
            is_training=False,
        )
        ensemble_preds += preds
    ensemble_preds /= len(ensemble_train_state)
    return ensemble_preds


def predict(forward, inputs, ensemble_train_state):
    return forward.apply(
        inputs=inputs,
        params=ensemble_train_state["params"],
        state=ensemble_train_state["state"],
        rng=None,
        is_training=False,
    )[0]
