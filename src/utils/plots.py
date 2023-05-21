import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import torch
import os
from matplotlib.pyplot import cm


def plot_data(inputs, targets, name):
    plt.plot(inputs, targets, "o")
    plt.savefig(f"{name}.png")
    plt.close()


def make_gif(
    frame_folder,
    duration=500,
):
    if os.path.exists(f"{frame_folder}/movie.gif"):
        os.remove(f"{frame_folder}/movie.gif")

    image_files = [
        os.path.join(frame_folder, image) for image in os.listdir(frame_folder)
    ]
    image_files.sort(key=lambda x: os.path.getmtime(x))

    if len(image_files) > 200:
        image_files = image_files[:: (len(image_files) // 200 + 1)]

    frames = [
        Image.open(image)
        for image in image_files
        if image.endswith(".png") or image.endswith(".jpg")
    ]
    frame_one = frames[0]
    frame_one.save(
        f"{frame_folder}/movie.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        loop=0,
    )


def save_predictions(dataset, pred_fn, name):
    dropped_x = np.concatenate([x for x, _y in dataset.dropped_obs], 0)
    dropped_y = np.concatenate([y for _x, y in dataset.dropped_obs], 0)
    all_x = np.sort(np.concatenate((dataset.x, dropped_x), 0), 0)
    ensemble_preds = pred_fn(inputs=all_x)
    np.savez(
        f"{name}.npz",
        dropped_x=dropped_x,
        dropped_y=dropped_y,
        dataset_x=dataset.x,
        dataset_y=dataset.y,
        all_x=all_x,
        ensemble_preds=ensemble_preds,
    )


def plot_predictions(dataset, pred_fn, name, config, title="", format="jpg"):
    if hasattr(dataset, "dropped_x"):
        dropped_x = dataset.dropped_x
        dropped_y = dataset.dropped_y
        all_x = torch.sort(torch.concatenate((dataset.x, dropped_x), 0), 0)[0]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_x = torch.sort(dataset.x, 0)[0].to(device)
    ensemble_preds = pred_fn(inputs=all_x).cpu().numpy()
    preds_mean = ensemble_preds.mean(axis=0)[:, 0]
    preds_std = ensemble_preds.std(axis=0)[:, 0]

    quantiles = {"10", "20", "30", "70", "80", "90"}
    preds_quantiles = {}
    for q in quantiles:
        preds_quantiles[q] = np.quantile(ensemble_preds, int(q) / 100, axis=0)[
            :, 0
        ]

        # Vincent's paper instead does following:
        # preds_mean + 2.0 * preds_std,
        # preds_mean - 2.0 * preds_std,

    # data = pd.DataFrame(
    #     {
    #         "x": np.tile(all_x, (ensemble_preds.shape[0], 1)).squeeze(),
    #         "y": ensemble_preds.reshape(-1, all_x.shape[-1]).squeeze(),
    #     }
    # )

    # region set plot style
    sns.set(style="white")
    ts, lw, ms = 20, 5, 4

    colors = [
        "#56641a",
        "#e6a176",
        "#00678a",
        "#b51d14",
        "#5eccab",
        "#3e474c",
        "#00783e",
    ]

    # The default matplotlib setting is usually too high for most plots.
    plt.locator_params(axis="y", nbins=2)
    plt.locator_params(axis="x", nbins=6)

    plt.figure(figsize=(10, 6))

    # endregion

    # region plot data
    plt.plot(
        dataset.x.cpu().numpy(),
        dataset.y,
        "o",
        label="observed",
        alpha=0.4,
        markersize=3,
    )
    if hasattr(dataset, "dropped_x"):
        plt.plot(
            dropped_x.cpu().numpy(),
            dropped_y,
            "o",
            label="dropped",
            alpha=0.4,
            markersize=3,
        )
    # endregion

    # region plot predictions
    all_x = all_x[:, 0].cpu().numpy()
    plt.plot(
        all_x,
        preds_mean,
        color=colors[2],
        label=f"{config.training.update_rule.args.method}",
        lw=lw / 3.0,
    )
    plt.fill_between(
        all_x,
        preds_quantiles["10"],
        preds_quantiles["90"],
        color=colors[2],
        alpha=0.3,
    )
    plt.fill_between(
        all_x,
        preds_quantiles["20"],
        preds_quantiles["80"],
        color=colors[2],
        alpha=0.2,
    )
    plt.fill_between(
        all_x,
        preds_quantiles["30"],
        preds_quantiles["70"],
        color=colors[2],
        alpha=0.1,
    )
    # endregion

    # region edit plot
    plt.title(title, fontsize=ts, pad=ts)

    leg = plt.legend()
    # plt.xlabel('$x$', fontsize=ts)
    # plt.ylabel('$y$', fontsize=ts)

    # plt.axis("off")
    # endregion

    plt.savefig(f"{name}.{format}", bbox_inches="tight", format=format)

    plt.close()


def plot_kde(data, name, title="", config=None, format="jpg", x2=None, y2=None):
    _fig, ax1 = plt.subplots()
    df = pd.DataFrame(data.to("cpu").numpy().squeeze())

    if df.shape[1] == 1:
        sns.kdeplot(df, ax=ax1)
        if x2 is not None and y2 is not None:
            ax2 = ax1  # .twinx()

            ax2.plot(
                x2,
                y2,
                c="k",
                linestyle="dashed",
            )
            # ax2.set_yticklabels([])

            plt.xlim(left=-4, right=3)
            plt.ylim(bottom=0.0, top=0.8)

            color = cm.viridis(np.linspace(0, 1, 5))
            for i, c in zip(range(5), color):
                plt.plot(
                    df.loc[i],
                    0.01,
                    marker="o",
                    markersize=10,
                    c=c,
                )
    elif df.shape[1] == 2:
        sns.kdeplot(df, ax=ax1, x=0, y=1)
        plt.scatter(df.loc[:, 0], df.loc[:, 1], alpha=0.5)
        color = cm.viridis(np.linspace(0, 1, 5))
        for i, c in zip(range(5), color):
            plt.plot(
                df.loc[i, 0],
                df.loc[i, 1],
                marker="o",
                markersize=20,
                markerfacecolor="green",
                c=c,
            )
    else:
        raise NotImplementedError

    plt.title(title)
    plt.savefig(f"{name}.{format}", bbox_inches="tight", format=format)

    plt.close()
