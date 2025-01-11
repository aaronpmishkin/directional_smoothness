from functools import partial
from itertools import product
from collections import defaultdict
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from experiment_utils.plotting.defaults import DEFAULT_SETTINGS
from experiment_utils import configs

from stepback.models.main import get_model
from stepback.datasets.main import get_dataset, infer_shapes

from exp_configs import EXPERIMENTS  # type: ignore
from experiment_utils import (
    configs,
    utils,
    files,
    command_line,
)
from experiment_utils.plotting import plot_grid, plot_cell, defaults

from matplotlib import pyplot as plt  # type: ignore

plt.rcParams.update({"text.usetex": True})

marker_spacing = 0.1
marker_size = 16
line_width = 8
max_x = 1e3
min_x = 1e-2

settings = DEFAULT_SETTINGS
settings["titles_fs"] = 26
settings["axis_labels_fs"] = 20
settings["legend_fs"] = 18
settings["tick_fs"] = 18
settings["wspace"] = 0.18

marker_size = 8
line_width = 3


line_kwargs = {
    "gd_actual": {
        "c": defaults.line_colors[1],
        "label": "$1/M(x_{k+1}, x_k)$",
        "linewidth": line_width,
        "marker": defaults.marker_styles[1],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "gd_directional_smoothness": {
        "c": defaults.line_colors[1],
        "label": "Bound ($1/M(x_{k+1}, x_k)$)",
        "linewidth": line_width,
        "marker": None,
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
    "traditional": {
        "c": defaults.line_colors[2],
        "label": f"Bound ($L$-Smooth)",
        "linewidth": line_width,
        "marker": None,
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
    "sps_actual": {
        "c": defaults.line_colors[3],
        "label": f"Polyak",
        "linewidth": line_width,
        "marker": defaults.marker_styles[3],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "sps_directional_smoothness": {
        "c": defaults.line_colors[3],
        "label": f"Bound (Polyak)",
        "linewidth": line_width,
        "marker": None,
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
}

row_key = "dataset"

metrics = [
    "train_loss",
    "train_loss_avg",
    "train_loss_min",
    "val_score",
    "val_score_avg",
    "val_score_max",
    "grad_norm",
    "step_size",
    "directional_smoothness",
    "optimal_smoothness",
]


def line_key(exp_config):
    return exp_config["opt"]["name"]


repeat_key = "repeat"

variation_key = ("opt", "lr")


def filter_fn(exp_config):
    """Remove experiments corresponding to null models."""

    return True


def get_metric_name(line, train=True):
    if train:
        metric_key = "train_loss"
    else:
        metric_key = "val_score"

    if "sps" in line:
        metric_key += "_avg"

    if "normalized" in line:
        if train:
            metric_key += "_min"
        else:
            metric_key += "_max"

    return metric_key


dataset_kwargs = {
    "test_prop": 0.2,
    "valid_prop": 0.2,
    "use_valid": False,
}
data_dir = "data"


def load_data(dataset_name, seed=None):
    data_config = {"dataset": dataset_name, "dataset_kwargs": dataset_kwargs}

    train_set = get_dataset(
        config=data_config,
        split="train",
        seed=seed,
        path=data_dir,
    )

    X_train, y_train = train_set.dataset.tensors

    return train_set, X_train, y_train


results_dir = "results/uci_logreg"
exp_list = EXPERIMENTS["uci_logreg"]

with open("./scripts/exp_configs/uci_constants.pkl", "rb") as f:
    uci_constants = pkl.load(f)

with open("scripts/exp_configs/initial_distances.pkl", "rb") as f:
    initial_distances = pkl.load(f)


def compute_standard_rate(metrics, dataset_name, method_name):
    L = uci_constants[(dataset_name, 0)]["L"]
    smoothness = metrics["directional_smoothness"]["center"]
    # print(dataset_name, smoothness[smoothness/2 > L], L)
    K = np.arange(len(smoothness)) + 1
    distance = initial_distances[(dataset_name, 0)]

    if (
        method_name == "sps"
        or method_name == "sgd-m"
        or method_name == "implicit_ss"
    ):
        bound = 2 * L * distance / K
    elif method_name == "normalized_sgd":
        pass

    bound = np.concatenate([bound[0:1], bound])
    return bound


def compute_smoothness_rate(metrics, dataset_name, method_name):
    smoothness = metrics["directional_smoothness"]["center"]
    K = np.arange(len(smoothness)) + 1
    sum_of_smoothness = np.cumsum(smoothness)
    avg_smoothness = sum_of_smoothness / K
    inv_smoothness = 1 / smoothness
    sum_inv_smoothness = np.cumsum(inv_smoothness)

    opt_smoothness = metrics["optimal_smoothness"]["center"]
    inv_opt_smoothness = 1 / metrics["optimal_smoothness"]["center"]
    sum_inv_opt_smoothness = np.cumsum(inv_opt_smoothness)
    sum_of_opt_smoothness = np.cumsum(opt_smoothness)
    avg_opt_smoothness = sum_of_opt_smoothness / K

    distance = initial_distances[(dataset_name, 0)]

    f0 = metrics["train_loss"]["center"][0]
    f_star = uci_constants[(dataset_name, 0)]["f_star"]
    opt_gap = f0 - f_star

    if method_name == "sps":
        bound = 2 * distance / sum_inv_opt_smoothness
    elif method_name == "implicit_ss":
        bound = distance / (2 * sum_inv_smoothness)
    elif method_name == "normalized-sgd":
        bound = opt_gap / K + distance * avg_smoothness / K

    bound = np.concatenate([bound[0:1], bound])

    return bound


metric_grid = files.load_and_clean_experiments(
    exp_list,
    results_dir,
    metrics=metrics,
    row_key=row_key,
    line_key=line_key,
    repeat_key=repeat_key,
    variation_key=variation_key,
    target_metric="train_loss",
    maximize_target=False,
    metric_fn=utils.quantile_metrics,
    keep=[(("opt", "use_two"), [True])],
    remove=[],
    filter_fn=filter_fn,
    transform_fn=None,
    processing_fns=[],
    x_key=None,
    x_vals=None,
    silent_fail=False,
)


def get_quantiles_dict(metrics):
    return {"lower": metrics, "center": metrics, "upper": metrics}


def subtract_opt(metrics, f_star):
    metrics["upper"] = metrics["upper"] - f_star
    metrics["center"] = metrics["center"] - f_star
    metrics["lower"] = metrics["lower"] - f_star

    return metrics


# flip result dictionary
flipped_dict = deepcopy(metric_grid)
for dataset_name, run_metrics in metric_grid.items():
    for metric_name, lines in run_metrics.items():
        for line_name, value in lines.items():
            flipped_dict[dataset_name][line_name][metric_name] = value


datasets_to_plot = []
for dataset in datasets_to_plot:
    fig = plt.figure(figsize=(6, 4))
    ax0 = plt.gca()

    run_metrics = flipped_dict[dataset]

    lines = {}
    f_star = uci_constants[(dataset, 0)]["f_star"]

    lines["gd_actual"] = subtract_opt(
        run_metrics["implicit_ss"]["train_loss_min"], f_star
    )
    lines["sps_actual"] = subtract_opt(
        run_metrics["sps"]["train_loss_min"], f_star
    )
    lines["traditional"] = get_quantiles_dict(
        compute_standard_rate(run_metrics["sgd-m"], dataset, "implicit_ss")
    )
    lines["gd_directional_smoothness"] = get_quantiles_dict(
        compute_smoothness_rate(
            run_metrics["implicit_ss"], dataset, "implicit_ss"
        )
    )
    lines["sps_directional_smoothness"] = get_quantiles_dict(
        compute_smoothness_rate(run_metrics["sps"], dataset, "sps")
    )

    # sps_lines["directional_smoothness_2"] = get_quantiles_dict(
    #     compute_smoothness_rate(run_metrics["sps"], dataset, "sps_2")
    # )

    plot_cell.make_convergence_plot(ax0, lines, line_kwargs, settings)

    # ax0.set_title("", fontsize=settings["subtitle_fs"])
    ax0.set_xlabel("Iterations", fontsize=settings["axis_labels_fs"])
    ax0.tick_params(labelsize=settings["tick_fs"])
    ax0.set_ylabel("Optimality Gap", fontsize=settings["axis_labels_fs"])

    # skip initialization iteration since we don't have abound on this value.
    ax0.set_xlim([1, len(lines["sps_directional_smoothness"]["center"])])

    ax0.set_yscale("log")

    handles, labels = ax0.get_legend_handles_labels()

    handles_to_plot, labels_to_plot = [], []
    for i, label in enumerate(labels):
        if label not in labels_to_plot:
            handles_to_plot.append(handles[i])
            labels_to_plot.append(label)

    legend = fig.legend(
        handles=handles_to_plot,
        labels=labels_to_plot,
        loc="lower center",
        borderaxespad=0.1,
        fancybox=False,
        shadow=False,
        ncol=3,
        fontsize=settings["legend_fs"],
        frameon=False,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=settings["wspace"],
        hspace=settings["vspace"],
        bottom=0.30,
    )
    figure_name = f"{dataset}"

    plt.savefig(f"figures/theoretical_comp_single/{figure_name}.pdf")

    plt.close()
    plt.clf()


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

for dataset, ax in [("ionosphere", ax0), ("mammographic", ax1)]:
    run_metrics = flipped_dict[dataset]

    lines = {}
    f_star = uci_constants[(dataset, 0)]["f_star"]

    lines["gd_actual"] = subtract_opt(
        run_metrics["implicit_ss"]["train_loss_min"], f_star
    )
    lines["sps_actual"] = subtract_opt(
        run_metrics["sps"]["train_loss_min"], f_star
    )
    lines["traditional"] = get_quantiles_dict(
        compute_standard_rate(run_metrics["sgd-m"], dataset, "implicit_ss")
    )
    lines["gd_directional_smoothness"] = get_quantiles_dict(
        compute_smoothness_rate(
            run_metrics["implicit_ss"], dataset, "implicit_ss"
        )
    )
    lines["sps_directional_smoothness"] = get_quantiles_dict(
        compute_smoothness_rate(run_metrics["sps"], dataset, "sps")
    )

    plot_cell.make_convergence_plot(ax, lines, line_kwargs, settings)

    ax.set_title(dataset, fontsize=settings["subtitle_fs"])
    ax.set_xlabel("Iterations", fontsize=settings["axis_labels_fs"])
    ax.tick_params(labelsize=settings["tick_fs"])

    # skip initialization iteration since we don't have abound on this value.
    ax.set_xlim([1, len(lines["sps_directional_smoothness"]["center"])])

    ax.set_yscale("log")

ax0.set_ylabel("Optimality Gap", fontsize=settings["axis_labels_fs"])
handles, labels = ax0.get_legend_handles_labels()

handles_to_plot, labels_to_plot = [], []
for i, label in enumerate(labels):
    if label not in labels_to_plot:
        handles_to_plot.append(handles[i])
        labels_to_plot.append(label)

legend = fig.legend(
    handles=handles_to_plot,
    labels=labels_to_plot,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    ncol=5,
    fontsize=settings["legend_fs"],
    frameon=False,
    handletextpad=0.4,
    columnspacing=0.8,
    handlelength=1.5,
)

plt.tight_layout()
fig.subplots_adjust(
    wspace=settings["wspace"],
    hspace=settings["vspace"],
    bottom=0.30,
)
figure_name = f"{dataset}"

plt.savefig(f"figures/figure_1.pdf")

plt.close()
plt.clf()
