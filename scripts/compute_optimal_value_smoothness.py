import os
import sys
from itertools import product
from functools import partial
import math

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.sparse.linalg import eigs
from scipy.special import expit
from scipy.optimize import check_grad, minimize
from sklearn.preprocessing import StandardScaler
import pickle as pkl

from stepback.datasets.main import get_dataset


def softplus(x, limit=10):
    result = np.copy(x)
    indices = result <= limit
    result[indices] = np.log1p(np.exp(x[indices]))
    return result


def logistic_loss(w, X, y):
    return np.sum(softplus(np.multiply(-y, X @ w))) / len(y)


def logistic_grad(w, X, y):
    return X.T @ np.multiply(expit(-np.multiply(y, X @ w)), -y) / len(y)


def logistic_hessian(w, X, y):
    sigmoid = expit(-np.multiply(y, X @ w))
    potentials = np.diag(sigmoid - sigmoid**2)
    H = X.T @ potentials @ X
    return H


def ls_cond(f_plus, f, g, dir, eta):
    return f_plus <= f - eta * np.dot(g, dir) / 2


def newton_method(w0, obj, grad, hess, grad_tol, max_iters):
    w = w0

    for i in range(max_iters):
        eta = 1
        f, g, H = obj(w), grad(w), hess(w)

        gg = np.dot(g, g)
        if gg <= grad_tol:
            print("Gradient tolerance reached: ", gg)
            return w

        dir = np.linalg.solve(H, g)

        w_plus = w - eta * dir
        f_plus = obj(w_plus)

        while not ls_cond(f_plus, f, g, dir, eta):
            eta = eta * 0.8
            w_plus = w - eta * dir
            f_plus = obj(w_plus)

        w = w_plus

        if i % 10 == 0:
            print(f"Iter {i+1}/max_iters, Grad Norm: {gg}, Obj: {f}, Step-size: {eta}")

    return w


def smoothness_cond(f_plus, f, g, iter_diff, eta):
    return f_plus <= f + g @ iter_diff + (iter_diff @ iter_diff) / (2 * eta)


def agd(w0, obj, grad, grad_tol, max_iters):
    w = w0
    v = np.copy(w0)
    alpha = 1
    eta = 1

    for i in range(max_iters):
        f, g = obj(v), grad(v)

        gg = np.dot(g, g)
        if gg <= grad_tol:
            print("Gradient tolerance reached: ", gg)
            return w

        w_plus = v - eta * g
        f_plus = obj(w_plus)

        while not ls_cond(f_plus, f, g, g, eta):
            eta = eta * 0.8
            w_plus = v - eta * g
            f_plus = obj(w_plus)

        alpha_plus = (1 + math.sqrt(1 + 4 * alpha**2)) / 2
        beta = (alpha - 1) / alpha_plus
        v_plus = w_plus + beta * (w_plus - w)

        if (v_plus - v) @ g >= 0:
            print("Restarting!")
            v = w
            alpha = 1
        else:
            alpha = alpha_plus
            v = v_plus

        w = w_plus
        eta = eta * 1.25

        if i % 10 == 0:
            print(f"Iter {i+1}/max_iters, Grad Norm: {gg}, Obj: {f}, Step-size: {eta}")

    return w


# Configure datasets.
dataset_kwargs = {
    "test_prop": 0.2,
    "valid_prop": 0.2,
    "use_valid": False,
}

datasets = [
    "horse-colic",
    "ionosphere",
    "mammographic",
    "ozone",
]

repeats = list(range(3))
data_dir = "./data"

tol = 1e-16
max_iter = 10000
verbose = False


results = {}

for name, repeat in product(datasets, repeats):
    print("Dataset:", name, "Repeat:", repeat)
    seed = 1234567 + repeat
    data_config = {"dataset": name, "dataset_kwargs": dataset_kwargs}

    train_set = get_dataset(
        config=data_config,
        split="train",
        seed=seed,
        path=data_dir,
    )
    test_set = get_dataset(
        config=data_config,
        split="val",
        seed=seed,
        path=data_dir,
    )

    X_train, y_train = train_set.dataset.tensors
    X_test, y_test = test_set.dataset.tensors

    X_train, y_train = X_train.numpy(), y_train.numpy().squeeze()
    X_test, y_test = X_test.numpy(), y_test.numpy().squeeze()

    n, _ = X_train.shape

    # manually add a bias (so it get's regularized).
    X_train = np.concatenate([X_train, np.ones((n, 1))], axis=-1)

    # compute smoothness constant: L <= lambda_max(X'X) / 4n
    XX = X_train.T @ X_train
    lam, _ = eigs(XX, k=1, which="LM")
    lam_prime = np.max(np.linalg.eigvals(XX))
    L = np.real(lam[0]) / (4 * n)

    d = X_train.shape[1]

    obj = partial(logistic_loss, X=X_train, y=y_train)
    grad = partial(logistic_grad, X=X_train, y=y_train)

    w_0 = np.zeros((d))

    result = minimize(
        fun=obj,
        x0=w_0,
        method="BFGS",
        jac=grad,
        tol=1e-12,
        options={"maxiter": max_iter, "gtol": 1e-12, "xrtol": 1e-12},
    )
    w_opt = result.x
    loss = obj(w_opt)

    results[(name, repeat)] = {"L": L, "f_star": loss, "w_opt": w_opt}


with open("./scripts/exp_configs/uci_constants.pkl", "wb") as f:
    pkl.dump(results, f)
