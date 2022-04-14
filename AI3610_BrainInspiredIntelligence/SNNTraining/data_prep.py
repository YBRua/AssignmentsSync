from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_cifar(path: str):
    """
    Loads the CIFAR dataset from the given path.
    :param path: The path to the CIFAR dataset.
    :return: The CIFAR dataset.
    """
    # define the transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    # load the CIFAR dataset
    cifar_train = datasets.CIFAR10(
        path, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(
        path, train=False, download=True, transform=transform)

    return cifar_train, cifar_test


def load_mnist(path: str):
    """
    Loads the MNIST dataset from the given path.
    :param path: The path to the MNIST dataset.
    :return: The MNIST dataset.
    """
    # define the transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    # load the MNIST dataset
    mnist_train = datasets.MNIST(
        path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(
        path, train=False, download=True, transform=transform)

    return mnist_train, mnist_test


def wrap_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    """
    Wraps the given dataset into a DataLoader.
    :param dataset: The dataset to wrap.
    :param batch_size: The batch size.
    :param shuffle: Whether to shuffle the dataset.
    :param drop_last: Whether to drop the last incomplete batch.
    :return: The DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last)
