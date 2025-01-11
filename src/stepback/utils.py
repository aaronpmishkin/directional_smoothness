"""
@author: Fabian Schaipp
"""
from typing import Iterable

import numpy as np
import torch
import itertools
import copy
import os
import math

from sklearn.linear_model import Ridge, LogisticRegression
from torch.nn.utils import parameters_to_vector

from .log import load_json, save_json

# %%
"""
Utility functions for managing output files.
"""


def get_output_filenames(exp_id, output_dir="output/"):
    """Given exp_id, it returns list of output filenames.
    The output filename should either be identical to exp_id or have the form <exp_id>-1, <exp_id>-2 etc
    """
    all_files = os.listdir(output_dir)
    exp_files = [
        f for f in all_files if (f == exp_id + ".json") or (exp_id + "-" in f)
    ]

    # remove .json from string
    exp_files = [f.split(".json")[0] for f in exp_files]

    return exp_files


def filter_output_file(
    exp_id,
    exp_filter=dict(),
    opt_filter=dict(),
    fname=None,
    output_dir="output/",
):
    """Filter ouput file. Deletes all results that are specified by exp_filter and opt_filter.
    Example:
        opt_filter = {'name': 'adam'} # deletes all adam runs
        exp_filter = {'dataset':'mnist} # deletes all runs where mnist was the dataset

    CAUTION: This will overwrite the file if not fname is specified. Otherwise, it will write the filtered output in fname.
    """
    print(f"Reading from {output_dir+exp_id}.")
    data = load_json(exp_id, output_dir)
    print(f"Original file has {len(data)} entries.")
    new_data = list()

    for d in data:
        DROP = False
        conf = copy.deepcopy(d["config"])

        for k, v in exp_filter.items():
            if conf.get(k) == v:
                DROP = True

        for k, v in opt_filter.items():
            if conf["opt"].get(k) == v:
                DROP = True

        if not DROP:
            new_data.append(d)
        else:
            print("Dropping the config:")
            print(conf)

    NEW_FNAME = exp_id if fname is None else fname
    save_json(NEW_FNAME, new_data, output_dir)
    print(f"New file has {len(new_data)} entries.")

    return


# %%
"""
Utility functions for Experiments.
"""


def prepare_config(exp_config: dict) -> dict:
    """
    Given an experiment config, we do the following preparations:

        * Convert n_runs to a list of run_id (integer values)
        * Convert each element of opt to a list of opt configs.
    """
    c = copy.deepcopy(exp_config)

    c["run_id"] = list(range(c["n_runs"]))
    del c["n_runs"]

    assert isinstance(
        c["opt"], list
    ), f"The value of 'opt' needs to be a list, but is given as {c['opt']}."

    all_opt = list()
    for this_opt in c["opt"]:
        # make every value a list
        for k in this_opt.keys():
            if not isinstance(this_opt[k], list):
                this_opt[k] = [this_opt[k]]

        # cartesian product
        all_opt += [
            dict(zip(this_opt.keys(), v))
            for v in itertools.product(*this_opt.values())
        ]

    c["opt"] = all_opt

    return c


def create_exp_list(exp_config: dict):
    """
    This function was adapted from: https://github.com/haven-ai/haven-ai/blob/master/haven/haven_utils/exp_utils.py

    Creates a cartesian product of a experiment config.

    Each value of exp_config should be a single entry or a list.
    For list values, every entry of the list defines a single realization.

    Parameters
    ----------
    exp_config : dict

    Returns
    -------
    exp_list: list
        A list of configs, each defining a single run.
    """
    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (
        dict(zip(exp_config_copy.keys(), v))
        for v in itertools.product(*exp_config_copy.values())
    )

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]

    return exp_list


# %%
"""
Utility functions for Pytorch models.
"""


def reset_model_params(model: torch.nn.Module):
    """
    resets all parameters of a Pytorch model
    from:
        https://discuss.pytorch.org/t/how-to-reset-model-weights-to-effectively-implement-crossvalidation/53859/2
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return


@torch.no_grad()
def l2_norm(model: torch.nn.Module):
    """
    compute l2 norm of a Pytorch model.
    """
    w = 0.0
    for p in model.parameters():
        w += (p**2).sum()
    return torch.sqrt(w).item()


@torch.no_grad()
def grad_norm(model: torch.nn.Module):
    """
    Computes norm of (stochastic) gradient.
    """
    device = next(model.parameters()).device
    g = torch.tensor([0.0], device=device)
    for p in model.parameters():
        if p.grad is not None:
            g += torch.sum(torch.mul(p.grad, p.grad))

    return torch.sqrt(g).item()


def update_momentum(params, m=None, beta=0.9):
    if m is None:
        m = []
        for p in params:
            m.append(p.grad)

    for i, p in enumerate(params):
        m[i] = m[i] * beta + p.grad * (1 - beta)

    return m


def parameter_norm(params):
    w = 0.0

    for p in params:
        w += (p**2).sum()

    return w.item()


# %%
"""
Compute optimal value for convex problems.
"""


def ridge_opt_value(X, y, lmbda, fit_intercept=False):
    n_samples = len(y)
    sk = Ridge(
        alpha=n_samples * lmbda / 2,
        fit_intercept=fit_intercept,
        tol=1e-10,
        solver="auto",
        random_state=1234,
    )

    sk.fit(X, y)
    sol = sk.coef_

    if fit_intercept:
        t2 = ((X @ sol + sk.intercept_ - y) ** 2).mean()
    else:
        t2 = ((X @ sol - y) ** 2).mean()

    t1 = lmbda / 2 * np.linalg.norm(sol) ** 2

    return t1 + t2


def logreg_opt_value(X, y, lmbda, fit_intercept=False):
    n_samples = len(y)

    if lmbda > 0:
        sk = LogisticRegression(
            C=1 / (n_samples * lmbda),
            penalty="l2",
            fit_intercept=False,
            tol=1e-10,
            random_state=1234,
        )
    else:
        sk = LogisticRegression(
            penalty="none", fit_intercept=False, tol=1e-10, random_state=1234
        )

    sk.fit(X, y)
    sol = sk.coef_.squeeze()
    print(sol)

    if fit_intercept:
        t2 = np.log(1 + np.exp(-y * (X @ sol + sk.intercept_))).mean()
    else:
        t2 = np.log(1 + np.exp(-y * (X @ sol))).mean()

    t1 = lmbda / 2 * np.linalg.norm(sol) ** 2
    return t1 + t2


def vector_to_parameters_no_grad(
    vec: torch.Tensor, parameters: Iterable[torch.Tensor]
) -> None:
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def vector_to_parameters(
    vec: torch.Tensor, parameters: Iterable[torch.Tensor]
) -> None:
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param[:] = vec[pointer : pointer + num_param].view_as(param)

        # Increment the pointer
        pointer += num_param


def directional_smoothness(theta, theta_prime, grad, grad_prime):
    num = torch.sum((grad_prime - grad) ** 2)
    denom = torch.sum((theta_prime - theta) ** 2)
    smoothness = 2 * torch.sqrt(num / denom).item()

    if denom == 0 or num == 0:
        smoothness = None

    return smoothness


def optimal_smoothness(f, f_prime, theta, theta_prime, grad, grad_prime):
    num = 2 * torch.abs(f_prime - f - torch.sum(grad * (theta_prime - theta)))
    denom = torch.sum((theta_prime - theta) ** 2)
    smoothness = (num / denom).item()

    if denom == 0 or num == 0:
        smoothness = None

    return smoothness


def get_grad(params):
    curr_grad = []
    for i, p in enumerate(params):
        # save gradient and iterate
        curr_grad.append(p.grad.clone())

    return parameters_to_vector(curr_grad)


def get_iter(params):
    curr_iter = []
    for i, p in enumerate(params):
        # save gradient and iterate
        curr_iter.append(p.data.clone())

    return parameters_to_vector(curr_iter)


def update_averages(param_vec, average, alpha, denom):
    if average is None:
        return param_vec.clone(), alpha
    else:
        denom_plus = denom + alpha
        average_plus = ((average * denom) + param_vec * alpha) / denom_plus

    return average_plus, denom_plus
