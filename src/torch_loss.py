import torch
import torch.distributions as D


def softmax_cross_entropy(preds, targets):
    return -torch.mean(
        torch.sum(torch.nn.functional.log_softmax(preds) * targets, axis=-1)
    )


def mse_loss(preds, targets):
    return torch.sum(
        torch.mean(torch.sum(torch.square(preds - targets), axis=-1), axis=-1)
    )


def toy1_loss(ensemble_train_state):
    return (
        3
        * (
            1 / 4 * ensemble_train_state**4
            + 1 / 3 * ensemble_train_state**3
            - ensemble_train_state**2
        )
        / 2
        - 3 / 8
    )


def toy1_loss_sum(ensemble_train_state):
    return toy1_loss(ensemble_train_state).sum()


def toy2_loss(ensemble_train_state):
    return -torch.sum(
        ensemble_train_state * torch.abs(torch.sin(ensemble_train_state))
    )


def toy3_loss(x):
    return ((x - 1) * x * (x**4 + 2 * x**3 - 12 * x**2 - 2 * x + 6)).sum()


def toy4_loss(x):
    return ((x + 3) * x**7 * (x - 3)).sum()


def sin_loss(x):
    return -torch.abs(torch.sin(x)).sum()


def gmm_distr():
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _GMM_MEANS = (
        torch.tensor(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32
        ).to(_DEVICE)
        * 3
    )
    mix = D.Categorical(
        torch.ones(
            4,
        ).to(_DEVICE)
    )
    comp = D.MultivariateNormal(
        _GMM_MEANS,
        torch.eye(
            2,
        )
        .repeat((4, 1, 1))
        .to(_DEVICE),
    )
    gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
    return gmm


def gmm_loss(ensemble_train_state):
    # _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _GMM_MEANS = torch.tensor([[-1, -1], [-1, 1]]).to(_DEVICE)
    # return -torch.log(
    #     torch.exp(
    #         -((ensemble_train_state[:, None] - _GMM_MEANS)**2).sum(-1) / 2
    #     ).sum(-1)
    # ).sum()
    return -gmm_distr().log_prob(ensemble_train_state).sum()


LOSS_DICT = {
    "softmax_cross_entropy": softmax_cross_entropy,
    "mse": mse_loss,
}

PARAMS_LOSS_DICT = {
    "toy1": toy1_loss_sum,
    "toy2": toy2_loss,
    "toy3": toy3_loss,
    "toy4": toy4_loss,
    "gmm": gmm_loss,
    "sin": sin_loss,
}


def create_loss_fn(loss_fn_name, forward):
    if loss_fn_name in LOSS_DICT:

        def loss_fn(inputs, targets, ensemble_train_state):
            preds, _ = forward.apply(inputs, ensemble_train_state)
            loss = LOSS_DICT[loss_fn_name](preds, targets)
            return loss

    elif loss_fn_name in PARAMS_LOSS_DICT:

        def loss_fn(inputs, targets, ensemble_train_state):
            loss = PARAMS_LOSS_DICT[loss_fn_name](ensemble_train_state)
            return loss

    else:
        raise ValueError(f"Loss function {loss_fn_name} not found")

    return loss_fn
