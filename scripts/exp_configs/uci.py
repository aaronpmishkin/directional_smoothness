from copy import deepcopy
import pickle as pkl
from itertools import product
import math
import numpy as np

with open("./scripts/exp_configs/uci_constants.pkl", "rb") as f:
    uci_constants = pkl.load(f)

step_sizes = np.logspace(-8, 1, 20).tolist()

repeats = list(range(1))

datasets = [
    "horse-colic", 
    "ionosphere", 
    "mammographic", 
    "ozone",
]

uci_logreg = {
    "dataset": None,
    "dataset_kwargs": {
        "test_prop": 0.2,
        "valid_prop": 0.2,
        "use_valid": False,
    },
    "model": "mlp",
    "model_kwargs": {
        "output_size": 1,
        "hidden_sizes": [[]],
        "bias": True,
    },
    "loss_func": "logistic",
    "reg": [0],
    "score_func": "logistic_accuracy",
    "opt": [
        {
            "name": "sgd-m",
            "lr": None,
            "weight_decay": 0,
            "momentum": 0,
            "dampening": 0,
            "lr_schedule": "constant",
        },
        {
            "name": "implicit_ss",
            "lr": 1,
            "weight_decay": 0,
            "use_two": True,
        },
        {
            "name": "sps",
            "lr": 1000,  # don't upper-bound step-size
            "lb": None,  # set manually.
            "weight_decay": 0,
            "lr_schedule": "constant",
            "prox": False,
        },
        {
            "name": "normalized-sgd",
            "lr": step_sizes,
            "weight_decay": 0,
            "lr_decay": True,
        },
        {
            "name": "dwod",
            "lr": 0.001,
            "weight_decay": 0,
            "smooth": True,
        },
    ],
    "batch_size": "full_batch",
    "max_epoch": 250,
    "eval_freq": 1,
    "repeat": None,
}

config_list = []

for name, repeat in product(datasets, repeats):
    const = uci_constants[(name, repeat)]
    exp_config = deepcopy(uci_logreg)
    exp_config["dataset"] = name
    exp_config["repeat"] = repeat
    # set 1/L step-size for GD
    exp_config["opt"][0]["lr"] = 1 / const["L"]

    # set exact f* for polyak step-size.
    exp_config["opt"][2]["lb"] = const["f_star"]

    config_list.append(exp_config)


EXPERIMENTS: dict[str, list] = {
    "uci_logreg": config_list,
}
