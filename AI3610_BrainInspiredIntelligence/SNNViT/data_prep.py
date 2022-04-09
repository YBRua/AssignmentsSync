from torchvision import transforms, datasets
from torch.utils.data import DataLoader


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_cifar(path: str):
    """
    Loads the CIFAR dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(path, train=False, download=True, transform=transform)
    return train_set, test_set


def load_mnist(path: str):
    """
    Loads the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
    train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(path, train=False, download=True, transform=transform)
    return train_set, test_set


def wrap_dataloaders(train_set, test_set, batch_size=64):
    """
    Wraps the MNIST dataset into PyTorch DataLoaders.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
