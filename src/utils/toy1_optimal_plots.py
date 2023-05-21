import numpy as np


def get_optimal_q(config, prior_samples=None):

    import scipy.integrate as integrate

    lower = -3
    upper = 3
    x = np.linspace(lower, upper, 1_000)

    if config.training.update_rule.args.method == "langevin":

        def loss(x):
            return 3 / 2 * (x**4 / 4 + x**3 / 3 - x**2) - 3 / 8

        def unnorm_q_toy1_langevin(ensemble_train_state):
            unnorm = (
                -loss(ensemble_train_state)
                + config.training.update_rule.args.langevin_reg_param
                * (-(ensemble_train_state**2))
                / 2
            )
            return unnorm

        all_unnorm = unnorm_q_toy1_langevin(x.squeeze())
        max_unnorm = all_unnorm.max()

        def q_toy1_langevin(x): return np.exp(
            unnorm_q_toy1_langevin(x) - max_unnorm)
        integral, _ = integrate.quad(q_toy1_langevin, -np.inf, np.inf)
        y = q_toy1_langevin(x.squeeze()) / integral

    # if True:
    elif config.training.update_rule.args.method == "repulsive":

        reg_2 = config.training.update_rule.args.langevin_reg_param
        reg_1 = config.training.update_rule.args.kernel_reg_param

        if config.training.update_rule.args.method == "langevin":
            reg_1 = 0
            prior_samples = np.random.normal(size=200)
        else:
            prior_samples = prior_samples.cpu().numpy().squeeze()

        from idesolver import IDESolver

        def dloss(x):
            return 3 / 2 * (x + 2) * x * (x - 1)

        def dlog_prior(x):
            return -x

        def dkernel(x, x_p, lengthscale=1):
            return (
                -(x - x_p) / lengthscale *
                np.exp(-((x - x_p) ** 2) / (2 * lengthscale))
            )

        def dkme_prior(x):
            return np.mean([dkernel(x, x_p) for x_p in prior_samples])

        def dloss_adj(x, reg1=1, reg2=1):
            return dloss(x) - reg2 * dlog_prior(x) - reg1 * dkme_prior(x)

        solver = IDESolver(
            x=x,
            y_0=-10,
            c=lambda theta, u: -1
            / reg_2
            * dloss_adj(
                theta,
                reg_1,
                reg_2,
            ),
            d=lambda theta: -reg_1 / reg_2,
            k=lambda theta, theta_p: dkernel(theta, theta_p),
            f=lambda u: np.exp(u),
            ode_atol=1e-6,
            lower_bound=lambda x: lower,
            upper_bound=lambda x: upper,
        )
        solver.solve()
        x, y = solver.x, np.exp(solver.y)

        integral = integrate.simpson(y, x)
        y = y / integral

    return x, y
