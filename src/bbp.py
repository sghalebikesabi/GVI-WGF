# %%
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import wandb


# %%
def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


# %%
def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
    exponent = -0.5 * (target - output) ** 2 / sigma**2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    if sum_reduce:
        return -(log_coeff + exponent).sum()
    else:
        return -(log_coeff + exponent)


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik * (varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.sigma**2
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))

        return (exponent + log_coeff).sum()


# %%
class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior

        scale = (2 / self.input_dim) ** 0.5
        rho_init = np.log(np.exp((2 / self.input_dim) ** 0.5) - 1)
        self.weight_mus = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01)
        )
        self.weight_rhos = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3)
        )

        self.bias_mus = nn.Parameter(
            torch.Tensor(self.output_dim).uniform_(-0.01, 0.01)
        )
        self.bias_rhos = nn.Parameter(
            torch.Tensor(self.output_dim).uniform_(-4, -3)
        )

    def forward(self, x, sample=True):
        if sample:
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(
                self.weight_mus.data.new(self.weight_mus.size()).normal_()
            )
            bias_epsilons = Variable(
                self.bias_mus.data.new(self.bias_mus.size()).normal_()
            )

            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))

            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons * weight_stds
            bias_sample = self.bias_mus + bias_epsilons * bias_stds

            output = torch.mm(x, weight_sample) + bias_sample

            # computing the KL loss term
            prior_cov, varpost_cov = self.prior.sigma**2, weight_stds**2
            KL_loss = (
                0.5 * (torch.log(prior_cov / varpost_cov)).sum()
                - 0.5 * weight_stds.numel()
            )
            KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
            KL_loss = (
                KL_loss
                + 0.5
                * ((self.weight_mus - self.prior.mu) ** 2 / prior_cov).sum()
            )

            prior_cov, varpost_cov = self.prior.sigma**2, bias_stds**2
            KL_loss = (
                KL_loss
                + 0.5 * (torch.log(prior_cov / varpost_cov)).sum()
                - 0.5 * bias_stds.numel()
            )
            KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
            KL_loss = (
                KL_loss
                + 0.5 * ((self.bias_mus - self.prior.mu) ** 2 / prior_cov).sum()
            )

            return output, KL_loss

        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, KL_loss

    def sample_layer(self, no_samples):
        all_samples = []
        for i in range(no_samples):
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(
                self.weight_mus.data.new(self.weight_mus.size()).normal_()
            )

            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))

            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons * weight_stds

            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()

        return all_samples


# %%
class BBP_Heteroscedastic_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(BBP_Heteroscedastic_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(
            num_units, 2 * output_dim, gaussian(0, 1)
        )

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)

        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss

        return x, KL_loss_total


# %%
class BBP_Heteroscedastic_Model_Wrapper:
    def __init__(self, network, learn_rate, batch_size, no_batches):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches

        self.network = network
        if torch.cuda.is_available():
            self.network.cuda()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learn_rate
        )
        self.loss_func = log_gaussian_loss

    def fit(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=torch.cuda.is_available())

        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0

        for i in range(no_samples):
            output, KL_loss_total = self.network(x)

            # calculate fit loss based on mean and standard deviation of output
            fit_loss = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1)
            fit_loss_total = fit_loss_total + fit_loss

        KL_loss_total = KL_loss_total / self.no_batches
        total_loss = (fit_loss_total + KL_loss_total) / (
            no_samples * x.shape[0]
        )
        total_loss.backward()
        self.optimizer.step()

        return fit_loss_total / no_samples, KL_loss_total

    def get_loss_and_rmse(self, x, y, no_samples, y_means=0, y_stds=1):
        x, y = to_variable(var=(x, y), cuda=torch.cuda.is_available())

        means, stds = [], []
        with torch.no_grad():
            for i in range(no_samples):
                output, KL_loss_total = self.network(x)
                output[:, :1] = output[:, :1] * y_stds + y_means
                output[:, 1:] = output[:, 1:].exp() * y_stds
                means.append(output[:, :1, None])
                stds.append(output[:, 1:, None].exp())

        means, stds = torch.cat(means, 2), torch.cat(stds, 2)
        mean = means.mean(dim=2)

        # calculate fit loss based on mean and standard deviation of output
        logliks = self.loss_func(
            output[:, :1], y, output[:, 1:], 1, sum_reduce=False
        )
        rmse = float((((mean - y) ** 2).mean() ** 0.5).cpu().data)

        return logliks, rmse


# %% [markdown]
# # UCI dataset fitting


# %%
class BBP_Heteroscedastic_Model_UCI(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(BBP_Heteroscedastic_Model_UCI, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(
            num_units, 2 * output_dim, gaussian(0, 1)
        )

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)

        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss

        return x, KL_loss_total


def train_BBP(
    train_ds, eval_ds, num_epochs, num_units, batch_size, lr, log_every
):
    num_epochs = num_epochs * len(train_ds.x) // batch_size
    train_logliks, test_logliks = [], []
    train_rmses, test_rmses = [], []

    x_train, y_train = train_ds.x, train_ds.y
    x_test, y_test = eval_ds.x, eval_ds.y

    y_means, y_stds = (
        y_train.mean(axis=0).numpy(),
        y_train.var(axis=0).numpy() ** 0.5,
    )
    # y_stds = np.ones_like(y_stds)
    # y_means = np.ones_like(y_means)
    y_train = (y_train - y_means) / y_stds
    # y_test = (y_test - y_means) / y_stds

    # batch_size, nb_train = len(x_train), len(x_train)

    net = BBP_Heteroscedastic_Model_Wrapper(
        network=BBP_Heteroscedastic_Model_UCI(
            input_dim=x_test.shape[-1], output_dim=1, num_units=num_units
        ),
        learn_rate=lr,
        batch_size=batch_size,
        no_batches=len(x_train) // batch_size,
    )

    fit_loss_train = np.zeros(num_epochs)
    KL_loss_train = np.zeros(num_epochs)
    total_loss = np.zeros(num_epochs)

    best_net, best_loss = None, float("inf")

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            x_train,
            y_train,
        ),
        shuffle=False,
        batch_size=batch_size,
        drop_last=True,
    )

    for i in range(num_epochs):
        x_train, y_train = next(iter(train_dataloader))
        fit_loss, KL_loss = net.fit(x_train, y_train, no_samples=20)
        fit_loss_train[i] += fit_loss.cpu().data.numpy()
        KL_loss_train[i] += KL_loss.cpu().data.numpy()

        total_loss[i] = fit_loss_train[i] + KL_loss_train[i]

        if fit_loss < best_loss:
            best_loss = fit_loss
            best_net = copy.deepcopy(net.network)

        if i % log_every == 0 or i == num_epochs - 1:
            train_losses, train_rmse = net.get_loss_and_rmse(
                x_train, y_train, 20
            )
            test_losses, test_rmse = net.get_loss_and_rmse(
                x_test, y_test, 20, y_means, y_stds
            )

            print(
                "Epoch: %s/%d, Train loglik = %.3f, Test loglik = %.3f, Train RMSE = %.3f, Test RMSE = %.3f"
                % (
                    str(i + 1).zfill(3),
                    num_epochs,
                    -train_losses.mean(),
                    -test_losses.mean(),
                    train_rmse,
                    test_rmse,
                )
            )
            eval_scalars = {
                "train_gaussian_nll": train_losses.mean(),
                "eval_gaussian_nll": test_losses.mean(),
                "train_rmse_of_mean": train_rmse,
                "eval_rmse_of_mean": test_rmse,
            }
            wandb.log(eval_scalars)

        train_losses, train_rmse = net.get_loss_and_rmse(x_train, y_train, 20)
        test_losses, test_rmse = net.get_loss_and_rmse(
            x_test, y_test, 20, y_means, y_stds
        )

        train_logliks.append((train_losses.cpu().data.numpy().mean()))
        test_logliks.append((test_losses.cpu().data.numpy().mean()))

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    print(
        "Train log. lik. = %6.3f +/- %6.3f"
        % (
            -np.array(train_logliks).mean(),
            np.array(train_logliks).var() ** 0.5,
        )
    )
    print(
        "Test  log. lik. = %6.3f +/- %6.3f"
        % (-np.array(test_logliks).mean(), np.array(test_logliks).var() ** 0.5)
    )
    print(
        "Train RMSE      = %6.3f +/- %6.3f"
        % (np.array(train_rmses).mean(), np.array(train_rmses).var() ** 0.5)
    )
    print(
        "Test  RMSE      = %6.3f +/- %6.3f"
        % (np.array(test_rmses).mean(), np.array(test_rmses).var() ** 0.5)
    )

    net = BBP_Heteroscedastic_Model_Wrapper(
        network=best_net, learn_rate=1e-2, batch_size=batch_size, no_batches=1
    )

    return


def run_bbp(
    train_dataset,
    eval_dataset,
    lr=1e-2,
    num_epochs=100,
    num_units=100,
    log_every=10,
):
    eval_scalars = train_BBP(
        train_dataset,
        eval_dataset,
        num_epochs=num_epochs,
        num_units=num_units,
        lr=lr,
        log_every=log_every,
        batch_size=len(train_dataset.x),
    )
    return eval_scalars


# %%
# %%
if __name__ == "__main__":
    import data

    np.random.seed(0)
    train_dataset = data.DATASET_DICT["uci"](
        seed=0,
        name="boston",
        split="train",
    )
    eval_dataset = data.DATASET_DICT["uci"](
        seed=0,
        name="boston",
        split="valid",
    )
    run_bbp(
        train_dataset,
        eval_dataset,
        lr=1e-2,
        num_epochs=100,
        num_units=100,
        log_every=10,
        batch_size=len(
            train_dataset.x
        ),  #! smaller batch size causes error in NLL calculation
    )

# %%
