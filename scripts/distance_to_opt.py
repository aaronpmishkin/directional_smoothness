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
]

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
expanded_list = configs.expand_config_list(exp_list)
sorted_exps = defaultdict(list)

for exp_config in expanded_list:
    dataset_name = exp_config["dataset"]
    sorted_exps[dataset_name].append(exp_config)

initial_distances = {}
init_suboptimality = {}

with open("./scripts/exp_configs/uci_constants.pkl", "rb") as f:
    uci_constants = pkl.load(f)

repeats = list(range(3))

for key, repeat in product(sorted_exps.keys(), repeats):
    exp_config = sorted_exps[key][0]

    seed = 1234567 + repeat
    run_seed = 456789
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  # Reseed to have same initialization
    torch.cuda.manual_seed_all(seed)

    train_set, X, y = load_data(key, seed)
    exp_config["_input_dim"], exp_config["_output_dim"] = infer_shapes(
        train_set
    )

    model = get_model(exp_config)
    w_0 = parameters_to_vector(model.parameters()).detach().numpy()

    gap = uci_constants[(key, repeat)]["w_opt"] - w_0

    # save optimality gap
    initial_distances[(key, repeat)] = np.sum(gap**2)

with open("scripts/exp_configs/initial_distances.pkl", "wb") as f:
    pkl.dump(initial_distances, f)
