""" 
In this file a lot of different SVGD implementations are collected, the basic structure of the class is the same of the 
standard SVGD.
"""

import math
import torch

import torch_models
import utils.torch_utils as torch_utils


class DefaultUpdateRule:
    def __init__(
        self,
        loss_fn,
        optimizer,
        mask,
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mask = mask

    def grad(self, ensemble_train_state, inputs, targets):
        loss = self.loss_fn(inputs, targets, ensemble_train_state)
        loss_dict = {"loss": loss}

        grad = torch.autograd.grad(loss, ensemble_train_state)[0] * self.mask

        return grad, loss_dict

    def step(self, ensemble_train_state, inputs, targets):
        self.optimizer.zero_grad()

        detached_state = ensemble_train_state.detach().requires_grad_(True)
        ensemble_train_state.grad, loss_dict = self.grad(
            detached_state, inputs, targets
        )
        loss_dict = {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}

        self.optimizer.step()

        return ensemble_train_state, loss_dict


class LangevinUpdateRule(DefaultUpdateRule):
    def __init__(
        self,
        loss_fn,
        init_params_var,
        keys,
        lr,
        langevin_reg_param,
        optimizer,
        mask,
        noise_param=1.0,
        likl_param=1.0,
    ):
        super().__init__(
            loss_fn,
            optimizer,
            mask,
        )
        self.keys = keys
        self.init_params_var = init_params_var

        self.langevin_reg_param = likl_param * langevin_reg_param
        self.sqrt_2_langevin__lr = noise_param * math.sqrt(
            2 * langevin_reg_param / lr
        )

    def get_langevin_terms(self, ensemble_train_state):
        reg_term = (
            self.langevin_reg_param
            * ensemble_train_state
            / self.init_params_var
        )
        noise_term = self.sqrt_2_langevin__lr * torch.randn(
            ensemble_train_state.shape, device=self.device
        )
        return reg_term, noise_term

    def grad(self, ensemble_train_state, inputs, targets):
        loss = self.loss_fn(inputs, targets, ensemble_train_state)
        loss_dict = {"loss": loss}

        grad = torch.autograd.grad(loss, ensemble_train_state)[0]
        reg_term, noise_term = self.get_langevin_terms(ensemble_train_state)

        return (grad + reg_term - noise_term) * self.mask, loss_dict


class RepulsiveLangevinUpdateRule(LangevinUpdateRule):
    def __init__(
        self,
        loss_fn,
        init_params_var,
        keys,
        lr,
        langevin_reg_param,
        kernel_reg_param,
        optimizer,
        forward,
        kernel_args,
        mask,
        noise_param=1.0,
        likl_param=1.0,
        kernel_param=1.0,
        prior_kernel_param=1.0,
    ):
        super().__init__(
            loss_fn,
            init_params_var,
            keys,
            lr,
            langevin_reg_param,
            optimizer,
            mask,
            noise_param,
            likl_param,
        )
        self.forward = forward
        self.kernel_reg_param = kernel_reg_param
        self.kernel_param = kernel_param
        self.prior_kernel_param = prior_kernel_param

        self.prior_kernel_preds = None
        self.prior_train_state = None
        self.kernel_samples = None

        self.prior_kernel = torch_utils.RBF(**kernel_args)
        self.post_kernel = torch_utils.RBF(**kernel_args)

    def set_prior_kernel_preds(self, prior_val_preds):
        self.prior_kernel_preds = prior_val_preds

    def set_kernel_samples(self, kernel_samples):
        self.kernel_samples = (
            torch.from_numpy(kernel_samples).float().to(self.device)
        )

    def set_prior_train_state(self, prior_train_state):
        self.prior_train_state = prior_train_state.detach()

    def grad(self, ensemble_train_state, inputs, targets):
        loss = self.loss_fn(inputs, targets, ensemble_train_state)

        reg_term, noise_term = self.get_langevin_terms(ensemble_train_state)

        val_preds, _ = self.forward.apply(
            self.kernel_samples, ensemble_train_state
        )
        val_preds_wo_grad = torch_models.predict(
            self.kernel_samples, ensemble_train_state, self.forward
        )

        # TODO: maybe add lengthscales
        # TODO: plot kernel matrix
        prior_kernel_loss = self.repulsive_loss_fn(
            preds_w_grad=val_preds,
            preds_wo_grad=self.prior_kernel_preds,
            ensemble_train_state=ensemble_train_state,
            ensemble_train_state_wo_grad=self.prior_train_state,
            kernel=self.prior_kernel,
        )

        kernel_loss = self.repulsive_loss_fn(
            preds_w_grad=val_preds,
            preds_wo_grad=val_preds_wo_grad,
            ensemble_train_state=ensemble_train_state,
            ensemble_train_state_wo_grad=ensemble_train_state.detach(),
            kernel=self.post_kernel,
        )

        grad = torch.autograd.grad(
            loss
            + self.kernel_reg_param
            * (
                self.kernel_param * kernel_loss
                - self.prior_kernel_param * prior_kernel_loss
            ),
            ensemble_train_state,
        )[0]

        full_grad = grad + reg_term - noise_term

        loss_dict = {
            "loss": loss,
            "total_loss": loss + kernel_loss - prior_kernel_loss,
            "kernel_loss": kernel_loss,
            "prior_kernel_loss": prior_kernel_loss,
        }

        return full_grad * self.mask, loss_dict


def create_update_rule(
    method,
    loss_fn,
    init_params_var,
    keys,
    lr,
    optimizer,
    forward,
    mask,
    langevin_reg_param=None,
    kernel_reg_param=None,
    kernel_args=None,
    noise_param=1.0,
    likl_param=1.0,
    kernel_param=1.0,
    prior_kernel_param=1.0,
):
    """
    Utils for the creation of the SVGD method
    """
    if method == "standard":
        method = DefaultUpdateRule(loss_fn, optimizer, mask)
    elif method == "langevin":
        method = LangevinUpdateRule(
            loss_fn,
            init_params_var,
            keys,
            lr,
            langevin_reg_param,
            optimizer,
            mask,
            noise_param,
            likl_param,
        )
    elif method == "repulsive":
        method = RepulsiveLangevinUpdateRule(
            loss_fn,
            init_params_var,
            keys,
            lr,
            langevin_reg_param,
            kernel_reg_param,
            optimizer,
            forward,
            kernel_args,
            mask,
            noise_param,
            likl_param,
            kernel_param,
            prior_kernel_param,
        )
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    return method


def maybe_create_optimizer(ensemble_train_state, lr, optimizer="adam"):
    if optimizer == "sgd":
        return torch.optim.SGD([ensemble_train_state], lr)
    elif optimizer == "adam":
        return torch.optim.Adam(
            [ensemble_train_state],
            lr,
            weight_decay=0,
            betas=[0.9, 0.999],
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")
