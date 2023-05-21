import os
import pickle as pkl
import wandb


def print_table(
    table, caption="Caption", label="my_label", rows=None, cols=None
):
    if rows is None:
        rows = table.index
    if cols is None:
        cols = table.columns
    table_str = "\\begin{table}[]\n"
    table_str += "\\centering\n"
    table_str += table.loc[rows, cols].to_latex(escape=False)
    table_str += "\\caption{" + caption + "}\n"
    table_str += "\\label{tab:" + label + "}\n"
    table_str += "\\end{table}\n"
    print(table_str)


def make_table_from_metric(
    metric,
    runs_df,
    val_metric=None,
    pm_metric="sem",
    data_name="dataset",
    method_name="method",
    latex=False,
    bold=True,
    show_group=False,
    minimise=True,
):
    if val_metric is None:
        val_metric = metric

    runs_df[metric] = runs_df[metric].astype(float)
    results = (
        runs_df.groupby(by=[method_name, data_name])
        .agg(
            {
                metric: ["mean", pm_metric],
                val_metric: ["mean", "std", "sem"],
            }
        )
        .reset_index()
    )
    # group_max_idx = (
    #     results.groupby(by=["method", data_name]).transform(max)[val_metric]["mean"]
    #     == results[val_metric]["mean"]
    # )
    table = results  # [group_max_idx]

    # table = table[table[data_name].isin(["Earthquake", "Fire", "Flood", "Volcano"])]

    if latex:

        def format_result(row):
            return f"{{{row[metric]['mean']:0.2f}_{{\pm {row[metric][pm_metric]:0.1f}}}}}"

        def bold_result(row):
            return (
                "\\bm" + row["result"] if row["bold"].any() else row["result"]
            )

    else:

        def format_result(row):
            return f"{row[metric]['mean']:0.2f} ± {row[metric][pm_metric]:0.2f}"

        def bold_result(row):
            return "* " + row["result"] if row["bold"].any() else row["result"]

    table["bold"] = (
        table.groupby(by=[data_name]).transform(min if minimise else max)[
            metric
        ]["mean"]
        == table[metric]["mean"]
    )

    table["result"] = table.apply(format_result, axis=1)
    if bold:
        table["result"] = table.apply(bold_result, axis=1)

    if latex:
        table["result"] = table.apply(
            lambda row: "$" + row["result"] + "$", axis=1
        )

    cols = (
        [method_name, data_name, "group"]
        if show_group
        else [method_name, data_name, "result"]
    )

    table_flat = table[cols].pivot(index=method_name, columns=data_name)

    table_flat = table_flat.droplevel(level=0, axis=1)
    table_flat = table_flat.droplevel(level=0, axis=1)
    table_flat.columns.name = None
    table_flat.index.name = None

    return table_flat


def rename_cols_for_pd_wide_to_long(col_names):
    new_column_names = [c for c in col_names]
    new_column_names = [
        c
        if " classification AUC" not in c
        else c.replace(" classification AUC", "_classification_AUC")
        for c in new_column_names
    ]
    new_column_names = [
        c if " test " not in c else c.replace(" test ", " test_")
        for c in new_column_names
    ]
    new_column_names = [
        c if "test" not in c else ";".join(c.split(" ")[::-1])
        for c in new_column_names
    ]
    new_column_names = [
        c if "_classification_AUC" not in c else ";".join(c.split(" ")[::-1])
        for c in new_column_names
    ]
    new_column_names = [
        c if "time" not in c else ";".join(c.split(" ")[::-1])
        for c in new_column_names
    ]
    return new_column_names


def print_df_duplicates(df, columns):
    return df[df.duplicated(subset=columns, keep=False)]


def slice_df(df, value_dict):
    for k, v in value_dict.items():
        df = df[df[k].isin(v)]
    return df


def load_or_run(path, fun, args):
    if not wandb.config.base["reload"] and os.path.exists(path):
        with open(path, "rb") as f:
            res = pkl.load(f)

    else:
        res = fun(*args)

        with open(path, "wb") as f:
            pkl.dump(res, f)

    return res
