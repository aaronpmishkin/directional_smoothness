import os
import torch
import numpy as np
import urllib
from sklearn.preprocessing import OneHotEncoder  # type: ignore


def train_test_split(X, y, valid_prop=0.2, split_seed=1995):
    """ """
    n = y.shape[0]
    split_rng = np.random.default_rng(seed=split_seed)
    num_test = int(np.floor(n * valid_prop))
    indices = np.arange(n)
    split_rng.shuffle(indices)
    test_indices = indices[:num_test].tolist()
    train_indices = indices[num_test:].tolist()

    # subset the dataset
    X_train = X[train_indices, :]
    y_train = y[train_indices]

    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return (X_train, y_train), (X_test, y_test)


def load_uci_dataset(
    split: str,
    name: str,
    path: str = "data/uci/datasets",
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    use_valid: bool = True,
    split_seed: int = 1995,
):
    data_dict = {}
    for k, v in map(
        lambda x: x.split(),
        open(os.path.join(path, name, name + ".txt"), "r").readlines(),
    ):
        data_dict[k] = v

    # load data
    f = open(os.path.join(path, name, data_dict["fich1="]), "r").readlines()[
        1:
    ]
    full_X = np.asarray(
        list(map(lambda x: list(map(float, x.split()[1:-1])), f))
    )
    full_y = np.asarray(list(map(lambda x: int(x.split()[-1]), f))).squeeze()

    classes = np.unique(full_y)
    if len(classes) == 2:
        full_y[full_y == classes[0]] = -1
        full_y[full_y == classes[1]] = 1
        full_y = np.expand_dims(full_y, 1)
    else:
        # use one-hot encoding for multi-class problems.
        full_y = np.expand_dims(full_y, 1)
        encoder = OneHotEncoder()
        encoder.fit(full_y)
        full_y = encoder.transform(full_y).toarray()

    # for vector-outputs

    # split dataset
    train_set, test_set = train_test_split(
        full_X, full_y, test_prop, split_seed=split_seed
    )

    if use_valid:
        train_set, test_set = train_test_split(
            train_set[0], train_set[1], valid_prop, split_seed=split_seed
        )

    X_train, y_train = torch.DoubleTensor(train_set[0]), torch.tensor(
        train_set[1], dtype=torch.double
    )
    X_test, y_test = torch.DoubleTensor(test_set[0]), torch.tensor(
        test_set[1], dtype=torch.double
    )

    if split == "train":
        return torch.utils.data.TensorDataset(
            X_train,
            y_train,
        )
    else:
        return torch.utils.data.TensorDataset(
            X_test,
            y_test,
        )
