import torch
import torchvision
from torchvision import transforms


def get_mnist(split, path, normalize=True, as_image=True):
    transform_list = [torchvision.transforms.ToTensor()]

    if normalize:
        # from Pytorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py
        norm = transforms.Normalize((0.1307,), (0.3081,))
        transform_list.append(norm)

    # input shape (28,28) or (784,)
    if not as_image:
        view = torchvision.transforms.Lambda(lambda x: x.view(-1).view(784))
        transform_list.append(view)

    ds = torchvision.datasets.MNIST(
        root=path,
        train=(split == "train"),
        download=True,
        transform=torchvision.transforms.Compose(transform_list),
    )

    return ds


def get_fashion_mnist(split, path, normalize=True, as_image=True):
    transform_list = [torchvision.transforms.ToTensor()]

    if normalize:
        # from Pytorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py
        norm = transforms.Normalize((0.5,), (0.5,))
        transform_list.append(norm)

    # input shape (28,28) or (784,)
    if not as_image:
        view = torchvision.transforms.Lambda(lambda x: x.view(-1).view(784))
        transform_list.append(view)

    ds = torchvision.datasets.FashionMNIST(
        root=path,
        train=(split == "train"),
        download=True,
        transform=torchvision.transforms.Compose(transform_list),
    )

    return ds


def get_binary_mnist(
    split,
    path,
    normalize=True,
    as_image=True,
    cls_a=0,
    cls_b=1,
):
    full_mnist = get_mnist(split, path, normalize, as_image)
    dataset = torch.utils.data.DataLoader(
        full_mnist, batch_size=len(full_mnist)
    )

    # reduce to binary problem.
    X, y = dataset._get_iterator().__next__()
    a_indices = y == cls_a
    b_indices = y == cls_b
    y[a_indices] = +1
    y[b_indices] = -1
    all_indices = torch.logical_or(a_indices, b_indices)
    X, y = X[all_indices], y[all_indices]

    return torch.utils.data.TensorDataset(X, y)
