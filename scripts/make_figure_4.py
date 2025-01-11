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

datasets_to_plot = [
    "ionosphere",
    "horse-colic",
    "ozone",
]


marker_spacing = 0.1
marker_size = 16
line_width = 5
max_x = 1e3
min_x = 1e-2

settings = DEFAULT_SETTINGS
settings["titles_fs"] = 26
settings["axis_labels_fs"] = 24
settings["legend_fs"] = 18
settings["ticks_fs"] = 20
settings["wspace"] = 0.18

marker_size = 8
line_width = 3

line_kwargs = {
    "normalized-sgd": {
        "c": defaults.line_colors[6],
        "label": f"Norm. GD",
        "linewidth": line_width,
        "marker": defaults.marker_styles[6],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "bb": {
        "c": defaults.line_colors[2],
        "label": f"BB",
        "linewidth": line_width,
        "marker": defaults.marker_styles[2],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "sps": {
        "c": defaults.line_colors[3],
        "label": f"Polyak",
        "linewidth": line_width,
        "marker": defaults.marker_styles[3],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "sgd-m": {
        "c": defaults.line_colors[2],
        "label": f"GD ($1/L$)",
        "linewidth": line_width,
        "marker": defaults.marker_styles[2],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "dwod": {
        "c": defaults.line_colors[5],
        "label": f"AdGD",
        "linewidth": line_width,
        "marker": defaults.marker_styles[5],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "implicit_ss": {
        "c": defaults.line_colors[1],
        "label": f"GD ($1/D_k$)",
        "linewidth": line_width,
        "marker": defaults.marker_styles[1],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
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


results_dir = "results/uci_logreg"
exp_list = EXPERIMENTS["uci_logreg"]

with open("./scripts/exp_configs/uci_constants.pkl", "rb") as f:
    uci_constants = pkl.load(f)

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


def subtract_opt(metrics, f_star):
    metrics["upper"] = metrics["upper"] - f_star
    metrics["center"] = metrics["center"] - f_star
    metrics["lower"] = metrics["lower"] - f_star

    return metrics


fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))

for dataset, ax in zip(datasets_to_plot, axes):
    run_metrics = metric_grid[dataset]

    train_lines = {}
    test_lines = {}
    for key, line in run_metrics["train_loss"].items():
        train_metric = get_metric_name(key, train=True)
        f_star = uci_constants[(dataset, 0)]["f_star"]
        train_lines[key] = subtract_opt(
            run_metrics["train_loss_min"][key], f_star
        )

    plot_cell.make_convergence_plot(ax, train_lines, line_kwargs, settings)

    ax.set_title(dataset, fontsize=settings["subtitle_fs"])
    ax.set_xlabel("Iterations", fontsize=settings["axis_labels_fs"])
    ax.tick_params(labelsize=settings["tick_fs"])
    ax.set_yscale("log")

axes[0].set_ylabel("Optimality Gap", fontsize=settings["axis_labels_fs"])

handles, labels = ax.get_legend_handles_labels()

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
)

plt.tight_layout()
fig.subplots_adjust(
    wspace=settings["wspace"],
    hspace=settings["vspace"],
    bottom=0.26,
)

plt.savefig(f"figures/figure_4.pdf")

plt.close()
plt.clf()
