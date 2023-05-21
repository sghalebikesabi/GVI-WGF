import functools
import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf
import time
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import wandb

import data
import eval
import utils


# TODO: try constant init with langevin
# TODO: mode collapse less likely?


@hydra.main(
    version_base=None,
    config_path=utils.get_project_root() + "/configs",
    config_name="config",
)
def main(config: DictConfig) -> None:
    # region ---------------------------------------------- init experiment
    print(OmegaConf.to_yaml(config))
    if config.data.train_args.name == "protein" and (
        config.seed > 5 and config.seed != 42
    ):
        return

    run_name = utils.make_run_name(config)
    wandb.init(
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
        config=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        ),
        name=run_name,
        **config.wandb_args,
    )

    params_path = utils.create_date_model_folder(
        "results/params", run_name, None
    )
    wandb.log({"params_path": params_path})
    if (
        config.logging.rolling_predictions_plot
        or config.logging.final_predictions_plot
        or config.logging.rolling_params_plot
        or config.logging.final_params_plot
    ):
        plots_path = utils.create_date_model_folder(
            "results/plots", run_name, config
        )

        wandb.log({"plots_path": plots_path})

    # random seed
    if config.make_deterministic:
        utils.make_deterministic(config.seed)

    if type(config.model.model_args.hidden_nodes) == int:
        config.model.model_args.hidden_nodes = [
            config.model.model_args.hidden_nodes
        ]
    # endregion

    # region ---------------------------------------------- create data
    train_dataset = data.DATASET_DICT[config.data.group](
        seed=config.seed,
        **config.data.train_args,
        **config.data.args,
        **config.training.update_rule.get("data", {}),
    )

    if config.data.eval_args.get("name") is None:
        config.data.eval_args.name = config.data.train_args.name
    eval_dataset = data.DATASET_DICT[config.data.group](
        seed=config.seed, **config.data.args, **config.data.eval_args
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = functools.partial(
        torch.utils.data.DataLoader,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )
    train_dataloader = data_loader(
        train_dataset,
        shuffle=True,
        **config.data.loader_args.train,
    )
    eval_dataloader = data_loader(
        eval_dataset,
        shuffle=False,
        **config.data.loader_args.eval,
    )
    if config.training.get("max_epoch") is not None:
        config.training.max_iter = config.training.max_epoch * len(
            train_dataloader
        )
    # endregion

    maybe_compile = utils.create_maybe_compile(config.framework)

    if config.run_bbp:
        import bbp

        bbp.run_bbp(
            train_dataset,
            eval_dataset,
            lr=config.bbp.lr,
            num_epochs=config.bbp.num_epochs,
            num_units=config.bbp.num_units,
            log_every=config.bbp.log_every,
        )

        wandb.finish()

        return

    if config.framework == "sklearn":
        from sklearn.neural_network import MLPRegressor

        if (
            config.model.model_type == "mlp"
            and config.training.loss_fn == "mse"
        ):
            regr_lst = []
            for model_idx in tqdm(range(config.model.n_models)):
                regr = MLPRegressor(
                    random_state=config.seed + model_idx,
                    max_iter=config.training.get(
                        "max_epoch",
                        config.training.max_iter // len(train_dataloader),
                    ),
                    batch_size=config.data.loader_args.train.batch_size,
                    hidden_layer_sizes=config.model.model_args.hidden_nodes,
                    learning_rate_init=config.training.update_rule.args.lr,
                    solver=config.training.optimizer,
                    **config.model.model_args.get("sklearn_args", {}),
                ).fit(train_dataset.x, train_dataset.y)
                regr_lst.append(regr)

            ensemble_train_state = regr_lst
            ema_avg = regr_lst

            def pred_fn(inputs, ensemble_train_state):
                predictions = []
                for regr in ensemble_train_state:
                    predictions.append(regr.predict(inputs)[..., None])
                return torch.from_numpy(np.stack(predictions))

        else:
            raise NotImplementedError(
                "Only sklearn Mlp Regressor is implemented"
            )

    elif config.framework == "haiku":
        import haiku_loss as loss
        import haiku_models as models
        import haiku_updates as updates

        training_key = jax.random.PRNGKey(config.seed)
        training_key, init_key = jax.random.split(training_key)

    elif config.framework == "torch":
        import torch_loss as loss
        import torch_models as models
        import torch_updates as updates

        init_key = None
        training_key = None

        if getattr(train_dataset, "dropped_x", None) is not None:
            train_dataset.dropped_x = torch.tensor(train_dataset.dropped_x).to(
                device
            )
            train_dataset.x = train_dataset.x.to(device)

    else:
        raise NotImplementedError(
            f"Model type {config.model.model_type} not implemented."
        )

    if config.framework == "haiku" or config.framework == "torch":
        # region ---------------------------------------------- init model and loss
        all_forward = models.AllEnsembleForward(
            n_models=config.model.n_models,
            model_type=config.model.model_type,
            model_args=config.model.model_args,
        )
        loss_fn = loss.create_loss_fn(
            config.training.loss_fn, forward=all_forward
        )
        # endregion

        # region ---------------------------------------------- init networks
        # samples from init Q(theta)
        dummy_inputs, _ = next(iter(train_dataloader))
        ensemble_train_state = all_forward.init(
            init_key, inputs=dummy_inputs, is_training=True
        )

        init_params_var = all_forward.init_value_mapping(
            ensemble_train_state, init=config.model.model_args.init
        )
        # print(hk.experimental.tabulate(all_forward.transformed_forward)(dummy_inputs, is_training=False))
        # endregion

        mask = torch.ones_like(ensemble_train_state)
        if config.model.get("load_from_day"):
            config.model.load_from_path = (
                f"results/models/{config.model.load_from_day}/{run_name}"
            )
        if config.model.get("load_from_path"):
            print("Old checkpoint recovered...")
            ensemble_train_state_old = ensemble_train_state
            try:
                ensemble_train_state = torch.tensor(
                    np.load(config.model.load_from_path)["arr_0"]
                ).to(device)
            except:
                ensemble_train_state = torch.tensor(
                    np.load(config.model.load_from_path)["params"]
                ).to(device)
            if config.model.reset_first_layer:
                ensemble_train_state[
                    :, : all_forward.net.first_layer_params
                ] = ensemble_train_state_old[
                    :, : all_forward.net.first_layer_params
                ]
                mask[:, all_forward.net.first_layer_params :] = 0

        # region ---------------------------------------------- init update rules
        optimizer = updates.maybe_create_optimizer(
            ensemble_train_state,
            config.training.update_rule.args.lr,
            config.training.get("optimizer"),
        )
        # some of the passed arguments may be filtered dependent on the update rule
        update_rule = updates.create_update_rule(
            loss_fn=loss_fn,
            keys=training_key,
            init_params_var=init_params_var,
            optimizer=optimizer,
            forward=all_forward,
            mask=mask,
            **config.training.update_rule.args,
        )
        prior_train_state = None
        if config.training.update_rule.args.method == "repulsive":
            # samples from prior P(theta), used for langevin term
            prior_forward = models.AllEnsembleForward(
                model_type=config.model.model_type,
                model_args=config.model.model_args,
                **config.training.update_rule.prior_model,
            )
            prior_train_state = prior_forward.init(
                init_key, inputs=dummy_inputs, is_training=True
            )

            update_rule.set_kernel_samples(train_dataset.kernel_samples.x)

            prior_kernel_preds = models.predict(
                update_rule.kernel_samples, prior_train_state, prior_forward
            )
            update_rule.set_prior_kernel_preds(prior_kernel_preds)

            if config.training.update_rule.distance_on == "preds":
                update_rule.repulsive_loss_fn = lambda preds_w_grad, preds_wo_grad, ensemble_train_state, ensemble_train_state_wo_grad, kernel: kernel.forward(
                    preds_w_grad[:, None], preds_wo_grad[None]
                )

            elif config.training.update_rule.distance_on == "params":
                update_rule.set_prior_train_state(prior_train_state)

                update_rule.repulsive_loss_fn = lambda preds_w_grad, preds_wo_grad, ensemble_train_state, ensemble_train_state_wo_grad, kernel: kernel.forward(
                    ensemble_train_state[:, None].unsqueeze(-1),
                    ensemble_train_state_wo_grad[None].unsqueeze(-1),
                )

            elif (
                config.training.update_rule.distance_on == "first_layer_params"
            ):
                update_rule.set_prior_train_state(prior_train_state)

                update_rule.repulsive_loss_fn = lambda preds_w_grad, preds_wo_grad, ensemble_train_state, ensemble_train_state_wo_grad, kernel: kernel.forward(
                    ensemble_train_state[:, None].unsqueeze(-1)[
                        :, :, : all_forward.net.first_layer_params
                    ],
                    ensemble_train_state_wo_grad[None].unsqueeze(-1)[
                        :, :, : all_forward.net.first_layer_params
                    ],
                )

            else:
                raise KeyError(
                    f"Distance on {config.training.update_rule.distance_on} not implemented"
                )

        if (
            config.training.update_rule.args.method in ["repulsive", "langevin"]
            and config.training.loss_fn == "toy1"
        ):
            from utils.toy1_optimal_plots import get_optimal_q

            params_plot_x2, params_plot_y2 = get_optimal_q(
                config, prior_samples=prior_train_state
            )
        else:
            params_plot_x2, params_plot_y2 = None, None

        pred_fn = functools.partial(models.predict, forward=all_forward)

        update = maybe_compile(update_rule.step)
        # endregion

        # region ---------------------------------------------- training
        start_time = time.time()
        early_stopping = utils.EarlyStopping(
            **config.training.early_stopping_args
        )
        ema_avg = ensemble_train_state.detach().clone()
        for iter_idx in tqdm(range(config.training.max_iter)):
            inputs, targets = next(iter(train_dataloader))

            # TODO: LR scheduler
            ensemble_train_state, loss_dict = update(
                ensemble_train_state=ensemble_train_state,
                inputs=inputs,
                targets=targets,
            )
            ema_avg = models.ema(
                ensemble_train_state.detach().clone(),
                ema_avg,
                config.training.ema.param,
            )

            wandb.log({"train_" + k: v for k, v in loss_dict.items()})

            if loss_dict["loss"].item() != loss_dict["loss"].item():
                raise ValueError("Loss is NAN.")

            if iter_idx % config.logging.checkpoint == 0:
                np.savez(
                    f"{params_path}/final_params",
                    params=ensemble_train_state.detach().cpu().numpy(),
                    iter_idx=iter_idx,
                )
            if iter_idx % config.logging.interval == 0:
                pred_fn_ = functools.partial(
                    pred_fn, ensemble_train_state=ensemble_train_state
                )
                eval_scalars, early_stopping_metric = eval.evaluate(
                    iter(eval_dataloader),
                    pred_fn_,
                    maybe_compile=maybe_compile,
                    **config.eval,
                )
                wandb.log({"eval_" + k: v for k, v in eval_scalars.items()})
                ema_pred_fn_ = functools.partial(
                    pred_fn, ensemble_train_state=ema_avg
                )
                eval_scalars, early_stopping_metric = eval.evaluate(
                    iter(eval_dataloader),
                    ema_pred_fn_,
                    maybe_compile=maybe_compile,
                    **config.eval,
                )
                wandb.log({"ema_eval_" + k: v for k, v in eval_scalars.items()})
                if early_stopping_metric is None:
                    raise ValueError("Eval metric is NAN.")
                # early_stop = early_stopping.callback(
                #     ensemble_train_state, early_stopping_metric
                # )
                # if early_stop:
                #     break
            if (
                config.logging.rolling_predictions_plot
                and iter_idx % config.logging.rolling_predictions_plot == 0
            ):
                pred_fn_ = functools.partial(
                    pred_fn, ensemble_train_state=ensemble_train_state
                )
                # try:
                utils.plot_predictions(
                    train_dataset,
                    pred_fn_,
                    f"{plots_path}/preds/rolling_predictions_{iter_idx}",
                    config,
                    f"{run_name}\niteration {iter_idx}"
                    + f" - Eval {config.eval.metrics[0]}: {eval_scalars[config.eval.metrics[0]]:.3f}"
                    if config.eval.metrics
                    else "",
                    **config.logging.plot_args,
                )
                # except AttributeError:
                #     config.logging.rolling_predictions_plot = False
                #     config.logging.final_predictions_plot = False
            if (
                config.logging.rolling_params_plot
                and iter_idx % config.logging.rolling_params_plot == 0
            ):
                utils.plot_kde(
                    ensemble_train_state,
                    f"{plots_path}/params/rolling_{iter_idx}",
                    f"{run_name}\niteration {iter_idx} - l(theta): {loss_dict['loss'].item():.3f}, num nan: {(ensemble_train_state !=ensemble_train_state).sum()}",
                    config=config,
                    x2=params_plot_x2,
                    y2=params_plot_y2,
                    # f"iteration {iter_idx} - Eval {config.eval.metrics[0]}: {eval_scalars[config.eval.metrics[0]]:.3f}",
                    **config.logging.plot_args,
                )
        # if early_stopping.early_stop:
        #     ensemble_train_state = early_stopping.best_params

        wandb.config.update({"training.n_iter": iter_idx})

        end_time = time.time()

        wandb.log({"training time": end_time - start_time})
        # endregion

    # region ---------------------------------------------- final eval
    pred_fn_ = functools.partial(
        pred_fn, ensemble_train_state=ensemble_train_state
    )
    eval_scalars, _early_stopping_metric = eval.evaluate(
        iter(eval_dataloader),
        pred_fn_,
        maybe_compile=maybe_compile,
        **config.eval,
    )
    wandb.log({"eval_" + k: v for k, v in eval_scalars.items()})
    ema_pred_fn_ = functools.partial(pred_fn, ensemble_train_state=ema_avg)
    eval_scalars, early_stopping_metric = eval.evaluate(
        iter(eval_dataloader),
        ema_pred_fn_,
        maybe_compile=maybe_compile,
        **config.eval,
    )
    wandb.log({"ema_eval_" + k: v for k, v in eval_scalars.items()})

    if config.logging.final_predictions_plot:
        final_preds_plot_path = f"{plots_path}/preds/final_predictions"
        utils.plot_predictions(
            train_dataset,
            pred_fn,
            final_preds_plot_path,
            config,
            f"{run_name}\niteration {iter_idx}"
            + f" - Eval {config.eval.metrics[0]}: {eval_scalars[config.eval.metrics[0]]:.3f}"
            if config.eval.metrics
            else "",
            **config.logging.plot_args,
        )
        wandb.log(
            {
                "final-preds-plot": wandb.Image(
                    f"{final_preds_plot_path}.{config.logging.plot_args.format}"
                )
            }
        )
    if config.logging.final_params_plot:
        final_params_plot_path = f"{plots_path}/params/final_params"
        utils.plot_kde(
            ensemble_train_state,
            final_params_plot_path,
            f"{run_name}\niteration {iter_idx} - l(theta): {loss_dict['loss'].item():.3f}",
            config=config,
            x2=params_plot_x2,
            y2=params_plot_y2,
            # f"iteration {iter_idx} - Eval {config.eval.metrics[0]}: {eval_scalars[config.eval.metrics[0]]:.3f}",
            **config.logging.plot_args,
        )
        wandb.log(
            {
                "final-params-plot": wandb.Image(
                    f"{final_params_plot_path}.{config.logging.plot_args.format}"
                )
            }
        )
    if config.logging.rolling_predictions_plot:
        utils.make_gif(
            plots_path + "/preds", **config.logging.get("gif_args", {})
        )
    if config.logging.rolling_params_plot:
        utils.make_gif(
            plots_path + "/params", **config.logging.get("gif_args", {})
        )
    if config.logging.final_predictions_save:
        preds_path = utils.create_date_model_folder(
            "results/predictions", run_name, config
        )
        wandb.log({"preds_path": preds_path})
        utils.save_predictions(
            train_dataset,
            pred_fn,
            f"{preds_path}/final_predictions",
        )

    if config.framework == "torch":
        np.savez(
            f"{params_path}/final_params",
            params=ensemble_train_state.detach().cpu().numpy(),
            iter_idx=iter_idx,
        )
    # endregion

    wandb.finish()


if __name__ == "__main__":
    main()
