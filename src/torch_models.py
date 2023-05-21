import functools
import ml_collections

from typing import Optional

import math
import numpy as np
import jax.numpy as jnp
import torch.nn as nn
import torch
import torch.nn.functional as F


def dnorm2(X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return dnorm2


def ema(model_parameter, averaged_model_parameter, alpha):
    return (1 - alpha) * averaged_model_parameter + alpha * model_parameter


class Mlp(nn.Module):
    """
    Implementation of Fully connected neural network from https://github.com/ratschlab/repulsive_ensembles

    Args:
        layer_sizes(list): list containing the layer sizes
        classification(bool): if the net is used for a classification task
        act: activation function in the hidden layers
        out_act: activation function in the output layer, if None then linear
        bias(Bool): whether or not the net has biases
    """

    def __init__(
        self,
        n_output=1,
        n_input=1,
        name: Optional[str] = None,
        init: ml_collections.ConfigDict = None,
        hidden_nodes=[200],
        act=F.relu,
        out_act=None,
        bias=True,
        d_logits=False,
    ):
        super(Mlp, self).__init__()
        del name

        self.bias = bias
        self.d_logits = d_logits
        self.ac = act
        self.out_act = out_act

        self.config_init = init

        self.layer_sizes = [n_input] + hidden_nodes + [n_output]
        self.layer_list = []
        for l in range(len(self.layer_sizes[:-1])):
            layer_l = nn.Linear(
                self.layer_sizes[l], self.layer_sizes[l + 1], bias=self.bias
            )
            self.add_module("layer_" + str(l), layer_l)

        self.num_params = sum(p.numel() for p in self.parameters())

        self.param_shapes = [list(i.shape) for i in self.parameters()]

        self.first_layer_params = np.prod(self.param_shapes[0])
        if self.bias:
            self.first_layer_params += np.prod(self.param_shapes[1])

        self.init_param_std = []

    def init(
        self,
    ):
        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._params = nn.ParameterList()

        for i, dims in enumerate(self.param_shapes):
            self._params.append(
                nn.Parameter(torch.Tensor(*dims), requires_grad=True)
            )

        if self.bias:
            for i in range(0, len(self._params), 2):
                self.init_params(self._params[i], self._params[i + 1])
        else:
            for i in range(0, len(self._params), 1):
                self.init_params(self._params[i])

        return self._params

    def init_params(self, weights, bias=None):
        """Initialize the weights and biases of a linear or (transpose) conv layer.

        Note, the implementation is based on the method "reset_parameters()",
        that defines the original PyTorch initialization for a linear or
        convolutional layer, resp. The implementations can be found here:

            https://git.io/fhnxV
        Args:
            weights: The weight tensor to be initialized.
            bias (optional): The bias tensor to be initialized.
        """
        # nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if self.config_init.method == "kaiming":
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weights)
            std_weights = math.sqrt(
                self.config_init.weight_init_var
            ) / math.sqrt(fan_out)
            std_bias = math.sqrt(self.config_init.bias_init_var) / math.sqrt(
                fan_out
            )
            weight_init_fn = functools.partial(
                nn.init.normal_, mean=0, std=std_weights
            )
            bias_init_fn = functools.partial(
                nn.init.normal_, mean=0, std=std_bias
            )

        elif self.config_init.method == "normal":
            std_weights = math.sqrt(self.config_init.weight_init_var)
            std_bias = math.sqrt(self.config_init.bias_init_var)
            weight_init_fn = functools.partial(
                nn.init.normal_, mean=0, std=std_weights
            )
            bias_init_fn = functools.partial(
                nn.init.normal_, mean=0, std=std_bias
            )

        elif self.config_init.method == "uniform":
            std_weights = (
                self.config_init.weight_max - self.config_init.weight_min
            )
            # std_bias = self.config_init.bias_max - self.config_init.bias_min
            weight_init_fn = functools.partial(
                nn.init.uniform_,
                a=self.config_init.weight_min,
                b=self.config_init.weight_max,
            )
            bias_init_fn = functools.partial(
                nn.init.uniform_,
                a=self.config_init.bias_min,
                b=self.config_init.bias_max,
            )

        else:
            raise NotImplementedError(
                f"Init method {self.config_init.method} not implemented."
            )

        weight_init_fn(weights)
        self.init_param_std.append(torch.zeros_like(weights) + std_weights)
        if bias is not None:
            # nn.init.uniform_(bias, -bound, bound)
            bias_init_fn(bias)
            self.init_param_std.append(torch.zeros_like(bias) + std_weights)

    def forward(self, inputs, params):
        """Can be used to make the forward step and make predictions.

        Args:
            x(torch tensor): The input batch to feed the network.
            weights(list): A reshaped particle
        Returns:
            (tuple): Tuple containing:

            - **y**: The output of the network
            - **hidden** (optional): if out_act is not None also the linear output before activation is returned
        """

        shapes = self.param_shapes
        assert len(params) == len(shapes)
        for i, s in enumerate(shapes):
            assert np.all(np.equal(s, list(params[i].shape)))

        hidden = inputs

        if self.bias:
            num_layers = len(params) // 2
            step_size = 2
        else:
            num_layers = len(params)
            step_size = 1

        for l in range(0, len(params), step_size):
            W = params[l]
            if self.bias:
                b = params[l + 1]
            else:
                b = None

            if l == len(params) - 2 and self.d_logits:
                pre_out = hidden
                distance_logits = dnorm2(pre_out, W)

            hidden = F.linear(hidden, W, bias=b)

            # Only for hidden layers.
            if l / step_size + 1 < num_layers:
                if self.ac is not None:
                    hidden = self.ac(hidden)

        if self.d_logits:
            hidden = -distance_logits
        if self.out_act is not None:
            return (
                self.out_act(hidden),
                hidden,
            )  # needed so that i can use second output for training first for predict
        else:
            return hidden

    def __call__(self, inputs, is_training):
        del is_training
        return self.net(inputs)


_MODELS_DICT = {
    "mlp": Mlp,
}


def model_fn(model_type, inputs, is_training, model_args):
    return _MODELS_DICT[model_type](**model_args)(inputs, is_training)


def mean_ensemble_fn(model_type, n_models, inputs, is_training, model_args):
    model_fn_lst = [
        _MODELS_DICT[model_type](name=f"net_{i}", **model_args)
        for i in range(n_models)
    ]
    return jnp.mean(
        jnp.stack(
            [model_fn(inputs, is_training) for model_fn in model_fn_lst],
        ),
        axis=0,
    )


class AllEnsembleForward:
    """Implementation of an ensemble of models

    This is a simple class to manage and make predictions using an ensemble with or without particles
    Args:
        device: Torch device (cpu or gpu).
        net: pytorch model to create the ensemble
        particles(Tensor): Tensor (n_particles, n_params) containing squeezed parameter value of the specified model,
            if None, particles will be sample from a gaussian N(0,1)
        n_particles(int): if no particles are provided the ensemble is initialized and the number of members is required

    """

    def __init__(self, model_type, n_models, model_args):
        self.model_type = model_type
        self.model_args = model_args
        self.n_models = n_models

        self.net = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def init(self, key, inputs, is_training):
        del key
        del is_training

        n_input = inputs.shape[-1]

        self.net = _MODELS_DICT[self.model_type](
            n_input=n_input, **self.model_args
        ).to(self.device)

        l = []
        for _ in range(self.n_models):
            l.append(torch.cat([p.flatten() for p in self.net.init()]).detach())
        self.particles = torch.stack(l).to(self.device)

        self.weighs_split = [np.prod(w) for w in self.net.param_shapes]

        return self.particles

    def reshape_particles(self, z):
        reshaped_weights = []
        z_splitted = torch.split(z, self.weighs_split, 1)
        for j in range(z.shape[0]):
            l = []
            for i, shape in enumerate(self.net.param_shapes):
                l.append(z_splitted[i][j].reshape(shape))
            reshaped_weights.append(l)
        return reshaped_weights

    def apply(self, inputs, ensemble_train_state):
        models = self.reshape_particles(ensemble_train_state)
        pred = [self.net.forward(inputs, w) for w in models]
        return torch.stack(pred), ()

    def init_value_mapping(self, params, init):
        del params
        del init
        return (
            torch.cat([p.flatten() for p in self.net.init_param_std])
            .detach()
            .to(self.device)
            .reshape((self.n_models, -1))
            ** 2
        )


class MeanEnsembleForward(AllEnsembleForward):
    def __init__(self, **args):
        super(MeanEnsembleForward, self).__init__(**args)

    def apply(self, inputs, ensemble_train_state):
        all_preds, _ = super().apply(inputs, ensemble_train_state)
        return all_preds.mean(0), ()


def predict(inputs, ensemble_train_state, forward):
    forward.net.eval()
    with torch.no_grad():
        preds, _ = forward.apply(
            inputs,
            ensemble_train_state,
        )
    return preds


def numpy_predict(inputs, ensemble_train_state, forward):
    inputs = torch.from_numpy(inputs).float().to(forward.device)
    return predict(inputs, ensemble_train_state, forward).cpu().numpy()
